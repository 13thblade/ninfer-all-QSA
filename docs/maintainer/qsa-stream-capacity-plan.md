# QSA stream：text 地址空间的窗口化（实施计划）

| 项 | 值 |
| --- | --- |
| 状态 | 已实施（P1–P3 完成，待实测确认） |
| 开始日期 | 2026-11-21 |
| 最后更新 | 2026-11-21 |
| 基线 commit | `8c90f8271`（master；工作树含 a358b4428 serve 层回退） |
| 目标 | 让 QSA stream 的共享 prefill staging 不被每 lane 的整段 entitlement 占满 |
| 复现命令 | `python qsa_longrun_int8auto.py run --tag I8AUTO --host-mib 16384` |

本文件是可迭代施工图：每完成一阶段勾掉 §6 对应项并追加一行进度。§2 的事实都带出处或实测数字，不要重新推导。

---

## 0. 口径

| 代号 | 指代 |
| --- | --- |
| **text 池** | `decoder.text_kv` 的 device 页池，单位 64-token 页（`kPagedKVPageSize=64`） |
| **stream 池** | 每 `(lane, layer)` 的 4-token 页池（`ops::kKvStreamPageTokens=4`），由 `--kv-resident` 定 |
| **resident** | 每 lane 在 device 上常驻的位置数，`--kv-resident 8192` = 8192 token = 128 个 64-token 页 |
| **entitlement** | 一条 sequence 的 text 地址空间页号上限，`text_kv_page_entitlement` |
| **staging** | dense prefill 读整段 KV 的共享 device 副本，就是 text 池本身 |
| **L** | `ceil(--max-context / 64)`，一个 lane 的完整窗口页数 |
| **C** | `--max-concurrency` |

---

## 1. 问题陈述（实测）

本次 6 session / 4 lane、int8 KV、128K、`--kv-resident 8192`、host 16G、ViT+MTP：

```
capacity | KV 131,072 tokens, int8, auto | pages 2,048/8,192 | runtime 5.35 GiB | free 1.09 GiB
WARN out of memory during materialization reserve: usable 2048 pages (capacity 2048, lent 0),
     allocated 1054, reserved 0, requested 1287
WARN req#3 failed | HTTP 429 | server overloaded
```

12 个请求 3 个 429，均为首轮并发的 67K 级 prompt。

**算术**：text 池 = 2048 页 = 128K token = 1 个 `--max-context` 窗口。4 lane × 67K prompt 各按 ~1054 页预留，`1054 + 1287 > 2048`。4 × 32K = 2048 页恰好卡满，所以 4×32K 能跑不是余量，是边界。

---

## 2. 根因

### 2.1 预算归属本身正确

`startup.cpp:270-275` 的注释表与实现一致：

| 项 | 轴 | 归属 | 出处 |
| --- | --- | --- | --- |
| device stream pool | `ceil(--kv-resident/4)` per (lane, layer) | 每 lane | `startup.cpp:346-383` |
| indexer pooled keys | `ceil(max_context/compress)+1` per (lane, layer) | 每 lane | `startup.cpp:311-327` |
| stream identity/map | `max_context` pages | 共享 | `startup.cpp:381-383` |
| **text staging** | `logical_pages = L` | **共享一份** | `startup.cpp:276` |
| MTP pool | `C × L` + extra | 每 lane | `startup.cpp:279-285` |
| GDN/hidden state | state slot per lane | 每 lane | `startup.cpp:390-400` |

`startup.cpp:246-250` 的设计原文：

> Keep exactly one device layer instead of sixteen... the staging layer holds `ceil(F / 64)` pages at frontier F
> Those reserved pages are **bookkeeping placeholders the sparse branch never reads** (the pinned host copy is the authority)

实测字节（`out/I8AUTO.jsonl`）：

```
kv_stream_staging_bytes = 276,824,064 B = 264.0 MiB   ← 2048 页 × 135168 B，1 层
kv_mtp_payload_bytes    = 1,107,836,928 B = 1056.5 MiB ← 8196 页，C×L + 4
kv_payload_bytes        = 1,384,660,992 B = 1320.5 MiB ← 两者之和
```

### 2.2 实现把"页号"算成了"每 lane 的整段物理页"

| 环节 | 代码 | 行为 |
| --- | --- | --- |
| entitlement | `output_budget.h:36` | `reserved = prompt_tokens + output - 1`，按**整段**算页 |
| 请求计划 | `request_plan.cpp:471-474` | `text_kv_page_entitlement = kv_pages_for_tokens(整段)`；67K → **1054 页/lane** |
| 激活预留 | `kv_store.h:948-976` | `activate(entitlement)` 的 `growth = entitlement` → `resize_reservation(1054)` **持久占池** |
| 映射 | `kv_store.h:1630-1665` | `ensure_mapped_to_tokens(frontier)` 从 reservation materialize 整段页 |

### 2.3 这些页零读写

`text.cpp:1901-1905`（QSA prefill）：

```cpp
// Under --kv-resident the pinned host copy is the ONLY authoritative one: the device text-KV
// pool holds a single shared 64-position staging layer (planned in persistent_layout), which
// wide calls fill from the host copy.
if (qsa_stream != nullptr) {
    ops::kv_cache_append(k_span, v_span, call_positions, stream_layer->host, stream);
}
```

prefill 只写 host；dense attention 读 staging（`stream.identity` 映射），decode 读 per-lane pool。materialize 出的 1054 页既不写也不读，只做页号占位。

### 2.4 上游 qwen4_exp 的语义

`qwen4_executor_upstream.cpp` / `qwen4_executor` 的 stream 模型：

| 层 | 位置 | 大小 |
| --- | --- | --- |
| host copy（权威） | pinned host，UVA 直接寻址 | `pages = ceil(max_context/64)` per (sequence, layer) |
| `kv_staging` | device | 同样 `pages` 页，**每 rank 一份，所有 sequence 共享** |
| `pool` | device | `stream_slots` × 4 positions per (sequence, layer) |

- prefill 宽调用：`kv_cache_append` 先写 host，`stage()` 把 `[0,first)` H2D 到 staging，再 `kv_cache_append` 写 staging，最后 `kv_stream_evict` 把写的块从 pool 排除（`qwen4_executor_upstream.cpp:3893-3940`）。
- decode 窄调用：`kv_stream_resolve` 按 indexer 选中的块，从 host 拷进 pool（`include/ninfer/ops/kv_stream.h:28-38`）。
- **没有 per-sequence 的整段 device 页，也没有 prefill 后全量 D2D 搬到 resident。**

结论：fork 的 `KVAddressSpaceStore` 为每 sequence materialize 整段页，是上游没有的多余结构；它既不承载权威数据，也不被读，却在共享 staging 池里按 lane 重复计费。

### 2.5 不是 capacity/headroom 的问题

`kv_capacity` 只决定 `main_page_groups`（stream 下被 `text_page_groups` 绕过）与启动校验，不决定 text 池。a358b4428 的 serve 层改动已回退（`git show a358b4428`）。

---

## 3. 目标语义

1. **text 地址空间的 entitlement = stream 窗口页数**：`ceil(stream_slots × 4 / 64)`，派生自 `plan.stream_slots`（`--kv-resident` 窗口被 indexer block budget 托底），不硬编码。
2. **逻辑 frontier 仍可增长到 `--max-context`**：host 是权威，`commit_frontier` 在 host-resident 模式下允许 frontier 超过 device `page_count`。
3. **text 池 `L` 页作为共享 staging**：每 lane 只占窗口页（128），4 lane = 512 < 2048。
4. **resident/indexer/MTP/GDN 的每 lane 预算不变**。

---

## 4. 修改点（已实施）

### P1：窗口页数派生

- `program_impl.h`：新增 `const std::uint32_t text_stream_window_pages`。
- `program_impl.cpp` 初始化：
  ```cpp
  text_stream_window_pages =
      plan.stream_slots == 0 ? 0U
      : kv_pages_for_tokens(plan.stream_slots * ops::kKvStreamPageTokens);
  ```
  完全派生，无字面量。

### P2：entitlement clamp

- `request_plan.cpp`（两个 entitlement 分支之后）：
  ```cpp
  if (text_stream_window_pages != 0) {
      base->text_kv_page_entitlement =
          std::min(base->text_kv_page_entitlement, text_stream_window_pages);
  }
  ```
- 效果：`activate` 的 `growth` = 窗口页数（128），`reservation` 只占 128 页/lane。

### P3：host-resident 的 store 语义

`kv_store.h` 的 `KVAddressSpaceStore` 构造新增 `host_resident`（`program_impl.cpp` 传 `plan.stream_slots != 0`）：

- `ensure_mapped_to_tokens`：`target = min(target, entitlement(address))`，不按整段 materialize。
- `commit_frontier`：允许 `frontier > page_count × 64`；coverage 只提交到 `min(pages_for_tokens(frontier), page_count)`。
- `destructive_truncate`：`target` clamp 到 entitlement；frontier 在窗口外时跳过 tail coverage 检查。

---

## 5. 验证

- 编译：`ninja -j14 ninfer_tests ninfer-serve` 通过。
- 单测：`ninfer_serve_options_test` → `ok`。
- 实测：`qsa_longrun_int8auto.py` 重跑（tag `I8AUTO3`），期望 12/12 成功、`allocated` 不随 lane 累加超过 2048、turn closure 命中不变。
- README 实测段：参数里 `--kv-headroom-mib 256` 在回退后会启动失败（`--kv-headroom-mib requires --kv-capacity auto`），需改为不带 headroom 或显式 `--kv-capacity auto`。

---

## 6. 进度

- [x] P0 审阅：预算归属、entitlement 口径、上游 qwen4_exp 语义（§2）
- [x] P1 `text_stream_window_pages` 派生
- [x] P2 entitlement clamp
- [x] P3 `host_resident` 的 ensure_mapped / commit_frontier / destructive_truncate
- [x] P4 6session int8 实测（`I8CP`，带 `--concurrent-prefill`）：12/12 成功，0 个 429/500，6 个 turn closure 命中 99.9–100%
- [ ] P5 README 口径（`--kv-headroom-mib 256` 在 serve 回退后会启动失败，需改）

---

## 7. 风险与回退

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| frontier 超过 device 页 | host-resident 放宽了 `commit_frontier` 的检查 | 只有 stream 的 text store 传 `host_resident=true`；backend（MTP）不受影响 |
| trim 越界 | `destructive_truncate` 原按整段算 target | 已 clamp 到 entitlement，并跳过窗口外的 tail 检查 |
| prefix snapshot 依赖整段页号 | stream 下 block publication 走 host store | 实测 turn closure 命中率验证 |
| 回退 | 改动集中在 `program_impl.*`、`request_plan.cpp`、`kv_store.h` | `git revert` 单提交即可 |
