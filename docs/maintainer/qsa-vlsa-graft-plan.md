# vLSA indexer 嫁接：可选 pre-norm 与跟随 trunk 的旋转（实施计划）

| 项 | 值 |
| --- | --- |
| 状态 | 完成（P0–P3、P5、P6 完成；P4 撤销） |
| 开始日期 | 2026-11-21 |
| 最后更新 | 2026-11-21 |
| 基线 commit | `5784f9bc8`（master，工作树干净） |
| 目标产物 | `E:/LLMs/Qwen3.8-27B-QSA-long-16k-1M-heavy-UD-Q4_K_XL.ninfer` |
| 参考实现 | https://github.com/Prefrontal-AI/latent-sparse-attention |

本文件是**可迭代的施工图**：每完成一个阶段就勾掉 `§4` 对应项并追加一行「进度」，换对话窗口时
从 `§4` 的未勾选项继续即可，不需要回溯历史对话。`§2` 的事实都带出处或实测数值，不要再重新推导。

---

## 0. 口径（先读，避免歧义）

| 代号 | 指代 | 路径 |
| --- | --- | --- |
| **v1** | kuraneko1 的第一版 QSA GGUF 张量 | `E:/LLMs/Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M.gguf` |
| **v2** | kuraneko1 的第二版 QSA GGUF 张量（结构同 v1，数值不同） | `E:/LLMs/Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M-v2.gguf` |
| **vLSA** | Prefrontal-AI 的 LSA 蒸馏产物（**不是** kuraneko1 的版本） | `E:/LLMs/qwen38_27b__long-16k-1M-heavy-qsa.pt` |

**v1 与 v2 都是 kuraneko1 的**，二者只差训练权重；**vLSA 与它们不是同源**。
后文一律用 v1 / v2 / vLSA，禁止用「v1/v2」指代 Prefrontal 的产物。

**`vLSA` 只在本文件与对话里出现**。仓库内的文件名、变量名、注释一律用 donor 的运行名
`long-16k-1M-heavy`（资产后缀 `-long-16k-1M-heavy`），因为这里嫁接的仍然是 **QSA indexer**，
与 LSA（latent sparse attention）注意力本身无关，写成 `lsa` 会被误读成另一套注意力。

当前仓库自带资产是 **v2** 的：

```
tools/convert/assets/indexer_tensors.manifest.json
  source_donor = Qwen3.8-27B-QSA-Unsloth-UD-Q4_K_M-v2.gguf
  tensor_count = 80
  asset_sha256 = c0f50809d5f6bdb93c07031f5ad45cc4aa479d2a0a2a448345870ee1be5cc98b
```

磁盘上的 `indexer_tensors.safetensors`（104,889,856 B）与上述 sha256 一致（2026-11-21 核对）。

---

## 1. 目标与非目标

### 目标

1. 引擎能加载**不含 `key_pre_norm`** 的 QSA 产物，并走「先 4-token 均值、后 norm」的路径。
2. 转换器能消费 vLSA 的 4 类张量（16×4=64 个），产出与 5 类资产同构的 `.ninfer`。
3. indexer 的旋转**跟随主注意力**（作者仓库的语义），不新增独立的 rope 设置。
4. 端到端：用 `Qwen3.8-27B-UD-Q4_K_XL.gguf` + vLSA 资产转换、加载、文本与带图推理通过，
   运行期用 QSA 原生配置 `--qsa-top-blocks 512`（= 2048 keys / 4-token block）。
5. 现有 v1/v2 路径的产物**逐字节不变**（默认行为零回归）。

### 非目标

- 不改 trunk（文本塔、ViT、MTP）的任何权重或数值路径。
- 不改 `qsa_indexer` 内核的数学；`PreNorm=false` 分支已存在并被测试覆盖。
- 不给 indexer 单独的 rope：`src/lsa/baselines/qsa.py` 写明它的旋转是「the rotation the
  model's own: its frequencies and its YaRN scale」，`lsa/assets/models/qwen38_27b_qsa.yaml`
  也说 indexer 之外的一切（含旋转）沿用 `qwen38_27b`。YaRN factor 40 是**模型级**的服务配置。
- 不引入新依赖（只用 torch / safetensors / numpy，均在现有 ninfer python env 内）。

---

## 2. 已核实的事实

### 2.1 张量对照

vLSA 的 `.pt` 结构：`{maps: {layer: {...}}, teacher: {...}, run: {...}, step: 48000}`，
`maps` 的 16 个层号 = `3, 7, 11, …, 63`（与 `qsa_gguf.qsa_layers()` 完全一致）。

| vLSA `maps[layer]` | 形状 | 对应 v1/v2 GGUF | 形状 | 关系 |
| --- | --- | --- | --- | --- |
| `index_qk_proj` | `(640, 5120)` | — | — | **按行切分**：`[0:512]` → `q_proj.weight`，`[512:640]` → `k_proj.weight` |
| `q_layernorm` | `(128,)` | `indexer.q_norm.weight` | `(128,)` | 都是零中心 `w`（gain = `1 + w`） |
| `k_layernorm` | `(128,)` | `indexer.k_norm.weight` | `(128,)` | 同上 |
| — | — | `indexer.k_norm.pre_weight` | `(128,)` | **vLSA 没有** |
| `log_tau` | 标量 | — | — | 训练损失温度，见 2.3 |
| `inverse_frequencies` | `(32,)` | — | — | 服务期不需要，见 2.2 |
| `attention_scaling` | 标量 | — | — | 同上 |

出处：`src/lsa/training/distillation/model.py` 第 179-180 行（"`index_qk_proj`
`[(H + 1) head_dim, model_dim]`, the query heads' rows first"）、第 261/277 行（key/query 的行切分）、
`inject.py` 第 55 行（`QSA_WEIGHTS = ("index_qk_proj", "q_layernorm", "k_layernorm")`）。

上游 `inject` 只写 3 个权重；`log_tau` / `inverse_frequencies` / `attention_scaling` 是运行时参数，
不进权重。

### 2.2 vLSA 记录的 rope 表是模型级 YaRN factor 40

`maps[layer]["inverse_frequencies"]` 与 `theta=1e7` 的裸表并不相同：从 pair 15 起偏离，pair 22-31
的相对差恒为 **97.5% = 1 − 1/40**，即 `plain / 40`；中段是 YaRN 的插值坡道。
`attention_scaling` 恒为 `1.368888 = 0.1·ln(40) + 1`。

用本仓库 `src/ops/launcher/rope.cu::yarn_schedule(dim=64, pairs=32, theta=1e7, factor=40,
native=262144)` 重算，与 `.pt` 的实测比对：

```
max|d| = 2.980e-08      （float32 舍入级）
attention_factor = 1.368887945   vs  .pt 的 1.368887901（同一 float32 值）
```

结论（2026-11-21 订正）：这张表是 **donor 蒸馏时模型级的服务旋转**（`qwen38_27b.yaml` 的
`rope_parameters.factor = 40`），不是 indexer 私有属性；提取器用它做 provenance 校验，引擎不读，
运行时 indexer 跟随 trunk（D3）。

### 2.3 `log_tau` 与排序无关

`model.py` 第 184-185 行原文：*"The read is invariant to `tau` — it ranks blocks by the score — and a
scale on the scores is one the query norm's gain already spans, so `tau` changes nothing the indexer can
express."* 因此提取时**丢弃**，只记入 manifest 备查。

### 2.4 两条路线的算子差异只有一处

| | 参考实现（`src/lsa/baselines/qsa.py`） | 本引擎（`src/ops/qsa_indexer/qsa_indexer.cu`） |
| --- | --- | --- |
| key | `k[b] = rotate(norm(mean_{s∈b} x_s W_K), r·b)` | `PreNorm=true` 时：`k[b] = norm_rope(mean_{s∈b}(rmsnorm(x_s W_K)·(1+w_pre)))` |
| | 逐 token 无 norm | **多一次逐 token RMSNorm** |

`PreNorm=false` 时 `pre_norm_key` 直接返回原值（`qsa_indexer.cu:156`），即参考实现的路径。
所以内核不用改，**只需要让这条路径能被选中**。

### 2.5 引擎目前恒为 `PreNorm=true`

| 位置 | 现状 |
| --- | --- |
| `src/ops/qsa_indexer/qsa_indexer.cu:895` | `if (weights.key_pre_norm != nullptr)` 决定模板参数 `PreNorm` |
| `src/models/qwen3_5/execution/text.cpp:1789` | `QsaIndexerWeights{&qsa.query_norm, &qsa.key_norm, &qsa.key_pre_norm}` — 取地址，**永远非空** |
| `src/models/qwen3_5/program/prefill.cpp:1075` | 同上 |
| `src/models/qwen4_exp/executor.cpp:3976`, `7320` | Flash-Next 显式传 `nullptr`（对照正确写法） |

因此 vLSA 产物即使不带 `key_pre_norm` 也进不了 `PreNorm=false`：必须先让它**可以缺席**。

### 2.6 相关基础设施的现状

| 事实 | 位置 |
| --- | --- |
| `Binder::parameter` 缺参数直接抛 `missing logical parameter` | `src/artifact/binder.cpp:41-43` |
| 仓库内**没有**可选绑定先例 | `grep "optional parameter\|absent"` 无结果 |
| `Bindings::direct` = `parameter(name, shape, {}, format, residency)` | `src/models/qwen3_5/load/prepare.cpp:728-731` |
| `layer_weights` 无条件收集 5 个 WeightId | `src/models/qwen3_5/load/text.cpp:438` |
| `Tensor` 默认 `data = nullptr` | `src/core/tensor.h:11-17` |
| `make_rope` 用 `rope_yarn_frequencies(64, 32, 1e7F, yarn, true)` | `src/ops/qsa_indexer/qsa_indexer.cu:874` |
| `rope_yarn_frequencies` **没有** factor 上限 | `src/ops/launcher/rope.cu:332-350` |
| `resolve_rope_scaling` 把 factor 限在 **4× native** | `src/runtime/contract/rope_scaling.h:69-74` |
| YaRN 只能由命令行给（`--rope-yarn`, `--rope-yarn-factor`），产物配置里没有 | `src/serve/serve_options.cpp:1588-1593` |

**关键推论（2026-11-21 订正）**：YaRN factor 40 是**模型级**的服务旋转，不是 indexer 属性。
`assets/models/qwen38_27b.yaml` 把它写在 `rope_parameters`，并注明 40 × 262,144 = 10,485,760，
刚好覆盖 10M 提示与回答；`assets/models/qwen38_27b_qsa.yaml` 说「Everything else is
qwen38_27b's, **the rotation it is served under included**」；`baselines/qsa.py` 的模块注释
同样写明 indexer 的旋转是「the rotation the model's own: its frequencies and its YaRN scale」。
因此 indexer 用 trunk 的 `RopeYarn` 即可，`inverse_frequencies` 只是蒸馏时那份服务旋转的记录。
`--rope-yarn-factor 4`（或 `--rope-yarn --max-context`）会同时作用于 trunk 与 indexer，这正是
作者仓库的语义；反过来，把 indexer 单独钉在 40、trunk 用 2×/4× 才是错的。

---

## 3. 设计决策

### D1 缺席即语义（presence-driven），不加配置布尔

产物里有没有 `attention/indexer/key_pre_norm` 这个 binding，就是 `PreNorm` 的开关。
理由：少一个可能与实际张量不一致的配置位；旧产物天然是 `true`，新产物天然是 `false`。

### D2 资产用 safetensors 的 `__metadata__` 记录 norm 形式

现有资产存的是 GGUF 的 `1 + w`，读取端统一 `-1.0`。vLSA 的 `q_layernorm` 是零中心 `w`，
若也存 `w + 1` 会多一次无谓的浮点往返。改为在资产头部写
`norm_form = "zero-centred"`，读取端据此**跳过** `-1.0`；缺省（旧资产）按 `"one-plus"` 处理。
自描述、不依赖 sidecar 文件名。

### D3 indexer 跟随 trunk 的 rope，不设独立字段（2026-11-21 订正）

不新增 `text_config.indexer_rope`。`qsa_indexer_append` / `qsa_indexer_select` 直接接收
`rope_yarn_`，与 trunk 共用同一份 `planned_rope_yarn`。依据见 §2.6：作者仓库没有
「indexer 自己的 rope」这个语义，factor 40 是模型级 YaRN（10M 窗口）。
产物只在 manifest 的 `distillation_rope` 里记录 donor 蒸馏时的旋转（provenance），
提取时用它校验 `.pt` 的 `inverse_frequencies` / `attention_scaling`；引擎不读这个字段。

### D4 vLSA 资产用运行名命名，通过环境变量选择

`assets/indexer_tensors-long-16k-1M-heavy.safetensors` +
`indexer_tensors-long-16k-1M-heavy.manifest.json`，通过 `NINFER_QSA_INDEXER_ASSET=<path>`
指定；默认路径仍是 v2 资产。`*.safetensors` 被 `.gitignore` 忽略，所以**只有 manifest 进
git**，资产本体作为 release 附件。

### D5 已知遗留（本期不改）

- `make_rope` 把 theta 硬编码为 `1.0e7F`（`qsa_indexer.cu:874`）。vLSA 正好也是 1e7，故不影响；
  若将来嫁接 theta 不同的 donor，需要把 theta 从配置传进来。
- `unit_offset` 对 vLSA 无影响：vLSA 是零中心形式，`unit_offset=true` 正是 `1 + w`。

---

## 4. 阶段与验收

每阶段一个 commit；提交信息用 `feat|fix|test|docs(scope): …`，并在正文写清验收结果。
**未通过验收不得勾选。**

### P0 规划（本文件）— [x]

- [x] 核对 vLSA 的键、形状、层号、`log_tau` 语义
- [x] 核实 vLSA 的 rope 表 == YaRN-40（数值比对）
- [x] 核实引擎恒 `PreNorm=true` 的根因
- [x] 写本文件并提交

### P1 引擎：允许 `key_pre_norm` 缺席 — [x]

改动文件与要点：

1. `src/artifact/binder.h` / `binder.cpp`
   新增 `[[nodiscard]] bool contains(std::string_view name) const;`
   → `reader_.directory().bindings.find(name) != …end()`。
2. `src/models/qwen3_5/load/bindings.h` + `prepare.cpp`
   新增 `std::optional<WeightId> optional_direct(std::string name, artifact::Shape shape,
   QType format = QType::BF16, artifact::Residency residency = artifact::Residency::Device);`
   以及 `bool has(std::string_view name) const { return binder.contains(name); }`。
   `optional_direct` 在缺席时返回 `std::nullopt` 且**不**注册参数、不推 `weights`。
3. `src/models/qwen3_5/weights.h:70`
   `std::optional<WeightId> key_pre_norm;`（其余四个不变）。
4. `src/models/qwen3_5/load/text.cpp`
   - `bind_qsa_indexer`（:48）改用 `optional_direct`；
   - `bind_qsa_indexer_share`（:153）先 `if (!b.has(i + "key_pre_norm")) return std::nullopt;`；
   - `layer_weights`（:438）改为条件 push（`if (q.key_pre_norm) out.push_back(*q.key_pre_norm);`）。
5. `src/models/qwen3_5/execution/parameters.h:72` → `std::optional<Tensor> key_pre_norm;`
   `parameters.cpp:434` → `if (a->qsa->key_pre_norm) { qsa.key_pre_norm = tensor(*a->qsa->key_pre_norm); }`
6. `src/models/qwen3_5/execution/text.cpp:1789` 与 `src/models/qwen3_5/program/prefill.cpp:1075`
   ```cpp
   const Tensor* pre = qsa.key_pre_norm ? &*qsa.key_pre_norm : nullptr;
   const ops::QsaIndexerWeights weights{&qsa.query_norm, &qsa.key_norm, pre};
   ```
   注意 `unit_offset` 仍走默认 `true`（v1/v2/vLSA 都是零中心）。

验收：

- [x] `ninfer_tests.exe ninfer_qsa_indexer_test` PASS（含 8102/8103/8104 的 `PreNorm=false` 分支）
- [x] `ninfer_tests.exe ninfer_sparse_attention_test`、`ninfer_kv_stream_test` PASS
- [x] 用现有 v2 产物（实际用了 `Qwen3.8-27B-QSA-UD-Q4_K_XL.ninfer`）加载并推理，输出正常
- [x] 提交 `30f9a2493`

### P2 转换器：4/5 role 两条路线统一 — [x]

1. `tools/convert/qsa_gguf.py`
   - 新增 `ROLES_NO_PRE_NORM = ("query", "key", "query_norm", "key_norm")`；
   - `register_parameters(model, dtype=DIRECT_FORMAT, pre_norm: bool = True)`；
   - `sources_from_asset(...)`：从 safetensors `__metadata__` 读 `norm_form`；
     `"zero-centred"` 时**不**减 1；并按资产实际键推导 role 集合（4 或 5，必须每层一致）。
2. `tools/convert/qsa_graft_recipe.py`
   - 由 `indexer_sources` 推导 `pre_norm`，把同一个布尔传给 `register_parameters` 与
     `recipe.add_parameter` 的循环；不再硬编码 `qsa_gguf.ROLES`。
3. `tests/convert/test_qsa_graft.py`
   - 新增一个 4-role、`norm_form=zero-centred` 的资产用例，断言注册 4 个参数、
     `recipe.add_parameter` 调 4 次、norm 不被减 1。

验收：

- [x] `pytest tests/convert -q` 串行跑：**425 passed / 3 failed / 1 skipped**，失败仍是 §7 那 3 个
- [x] v2 资产与 v2 donor 的读取结果**逐位一致**：80/80 张量 `np.array_equal`（`fp32` norm 路径）

### P3 vLSA 资产与提取脚本 — [x]

1. 新增 `tools/convert/extract_long_16k_indexer.py`
   - 入参：`.pt` 路径（环境变量 `NINFER_LONG_16K_PT`，无内置路径）；
   - 校验：`maps` 的键集合 == `qsa_gguf.qsa_layers()`；`index_qk_proj` 为 `(640, 5120)`；
     三个向量为 `(128,)`；全部有限；
   - 写出 64 个张量，命名沿用 `qsa_gguf.gguf_tensor(layer, role)`（便于复用现有读取路径）；
     `q = proj[:512]`、`k = proj[512:]` 均 **FP32**；
   - metadata：`norm_form=zero-centred`、`donor=long-16k-1M-heavy`、`attention_scaling`、
     `distillation_rope`（provenance）、逐层 `log_tau`（仅备查）；
   - manifest：`tensor_count=64`、逐张量 `sha256`、`asset_sha256`、`source_donor`、
     `distillation_rope`（引擎不读）。
2. 提交 manifest（`*.safetensors` 由 `.gitignore` 忽略，资产本体不进 git）。
3. 新增 `tests/convert/test_long_16k_asset.py`
   - 断言：资产里的 `q/k` 与 `.pt` 的切分**逐位相同**（BF16 表示下）、norm 与 `.pt` 的 `w`
     **逐位相同**、metadata 的 `norm_form` 为 `zero-centred`、manifest 的 sha256 与文件一致；
   - 再经 `sources_from_asset` 读一遍，确认**没有**执行 `-1.0`（归一化增益不被平移）。
   - rope 表的正确性由提取器的 `_check_rope` 在**提取时**自校验（factor 40 / native 262144
     必须复现 `.pt` 的 `inverse_frequencies`，否则拒绝提取），无需在测试里重复。
   - 通过 `NINFER_LONG_16K_PT` 提供 `.pt` 路径；未设置则整体 skip（CI 无此文件）。

验收：

- [x] `python tools/convert/extract_long_16k_indexer.py` 输出 64 张量、打印 sha256
      （复刻后 `25d756cd…`）
- [x] `NINFER_LONG_16K_PT=… pytest tests/convert/test_long_16k_asset.py -q` → **6 passed**
- [x] manifest 与资产 sha256 一致
- [x] 提交

### P4 indexer 独立 rope — 已撤销（2026-11-21）

原计划给 `text_config.indexer_rope` 单独构造 `RopeYarn`，让 indexer 固定用 YaRN-40。
核实作者仓库后确认这是误读：40 是**模型级** YaRN（10,485,760 = 40 × 262,144，覆盖 10M
提示），`qwen38_27b_qsa.yaml` 明说 indexer 沿用 `qwen38_27b` 的旋转，`baselines/qsa.py`
写的是「the rotation the model's own」。独立字段会让 indexer 在 trunk 用 2×/4× 时仍停在
40，破坏两者的相位一致性，因此**整段改动已回退**：

- [x] 回退 `config.h/.cpp` 的 `IndexerRope`、`startup.h/.cpp` 的 `indexer_rope_yarn`、
      `context.h` / `program_impl.*` / `execution/text.*` / `decode` / `prefill` / `draft` /
      `commit` / `graphs` / `mtp` 的传递，以及 `tests/ops/test_rope.cpp` 的 vLSA 表断言
- [x] `qsa_indexer_append/select` 继续用 trunk 的 `rope_yarn_`，并在调用处写明理由
- [x] `qwen3_5.py` 不再透传 `indexer_rope`（P5 的 base 配置也不需要加）

### P5 端到端 — [ ]

```bash
# 1) 指定 vLSA 资产并转换（base 需要 fp32_direct=true，沿用 F:/temp_scripts/qsa_fp32_base；
#    base 的 config.json 保持原样，不加 indexer_rope）
NINFER_QSA_INDEXER_ASSET=F:/ninfer-all-QSA/tools/convert/assets/indexer_tensors-long-16k-1M-heavy.safetensors \
  <py> -m tools.convert.convert_qsa27b \
    --base-model F:/temp_scripts/qsa_fp32_base \
    --gguf E:/LLMs/Qwen3.8-27B-UD-Q4_K_XL.gguf \
    --vision E:/LLMs/mmproj-BF16.gguf \
    --components text,vision,mtp \
    --out E:/LLMs/Qwen3.8-27B-QSA-long-16k-1M-heavy-UD-Q4_K_XL.ninfer \
    --name Qwen3.8-27B-QSA-long-16k-1M-heavy-UD-Q4_K_XL --device cpu
```

验收（运行期一律 `--qsa-top-blocks 512`，即 QSA 原生 2048 keys / 4-token block）：

- [x] 产物有 **64** 个 indexer 张量（无 `key_pre_norm`），norm 为 fp32（`fp32_direct=true`）
- [x] 产物 config **不含** `indexer_rope`
- [x] 服务加载成功（9.1 s，17.3 GiB），纯文本推理正常
- [x] 带图 + MTP 推理正常：2105 prompt tokens、82.4 tok/s、draft 103/163
- [x] 长上下文不崩：19,835 prompt tokens（~4,950 个块，选 512）、prefill 1726 tok/s、
      decode 72.4 tok/s，回答切题
- [x] indexer 权重与 `.pt` 逐位一致：**41,947,136 / 41,947,136**，`max|delta| = 0`
- [x] 提交

### P6 文档 — [x]

- [x] `MODIFICATIONS.md`：新增 vLSA 资产与可选 pre-norm 两处（不写 `indexer_rope`）
- [x] 本文件勾选完毕，状态行改为「完成」

---

## 5. 全局约束（每次都要遵守）

1. **编译用 14 线程**：
   `cmd.exe //c "F:\temp_scripts\build_ninfer.bat <target>"`
   该脚本已带 `-j 14`。不要另起 `cmake --build` 而不带 `-j`（会吃满所有核）。
2. **测试必须串行**：一次只跑一个 `ninfer_tests.exe <name>`；不要 `ctest -j`、不要并行
   `pytest -n`。GPU 只有一张，并发会互相污染时序与显存。
3. **改动以 git commit 落地**：一个阶段一个 commit；不 amend 已推送的提交；推送前先跑完本阶段验收。
4. **改代码前先停服务**：`taskkill //F //IM ninfer-serve.exe`（否则链接会因文件占用失败）。
5. **临时脚本/日志放 `F:/temp_scripts/`**，不要污染仓库。
6. 只读分析可以随时做；**不要**在未说明的情况下启动长时间基准测试。

---

## 6. 命令手册

```bash
# 仓库与状态
cd /f/ninfer-all-QSA
git log --oneline -6 && git status --short

# 编译（14 线程）
cmd.exe //c "F:\temp_scripts\build_ninfer.bat ninfer-serve"
cmd.exe //c "F:\temp_scripts\build_ninfer.bat ninfer_tests"

# 测试（串行，一次一个）
./build-msvc/tests/ninfer_tests.exe ninfer_qsa_indexer_test
./build-msvc/tests/ninfer_tests.exe ninfer_sparse_attention_test
./build-msvc/tests/ninfer_tests.exe ninfer_kv_stream_test
./build-msvc/tests/ninfer_tests.exe ninfer_qwen3_5_qsa_lane_prefix_test
./build-msvc/tests/ninfer_tests.exe ninfer_qwen3_5_qsa_stream_fallback_test

# 转换器测试（串行）
"C:/Users/bili8/.workbuddy/binaries/python/envs/ninfer/Scripts/python.exe" -m pytest tests/convert -q

# 服务（QSA + MTP + 视觉）
./build-msvc/apps/ninfer-serve.exe "<artifact>" --host 127.0.0.1 --port 8131 \
  --greedy --seed 1234 --qwen38-27b-qsa --qsa-top-blocks 512 --kv-resident 16384 \
  --host-cache-mib 0 --max-context 262144 --kv-capacity 262144 --kv-dtype bf16 \
  --prefill-chunk 2048 --spec mtp --draft-tokens 3 --ngram-draft-tokens 0 \
  --vision --vision-residency overlay --wddm-evictable-budget

# 产物结构核对
"C:/Users/bili8/.workbuddy/binaries/python/envs/ninfer/Scripts/python.exe" \
  -m tools.artifact.inspect "<artifact>" --objects --bindings --json > /f/temp_scripts/x.json
```

---

## 7. 已知坑与既有失败

| 现象 | 处理 |
| --- | --- |
| `vcvars64.bat` 被安全策略拦（它调 `reg.exe`） | 只用 `F:\temp_scripts\build_ninfer.bat`，它手写 INCLUDE/LIB |
| 链接报文件占用 | `taskkill //F //IM ninfer-serve.exe` |
| `git credential fill` 卡住 | 前缀 `GCM_INTERACTIVE=never GIT_TERMINAL_PROMPT=0` |
| `github.com` 直连超时 | `api.github.com` / `raw.githubusercontent.com` 可用 |
| Actions 日志下载慢（~68 KB/s） | 先取 302 的 location 再直连下载，超时给 240 s |
| `pytest tests/convert` 有 3 个既有失败 | `test_deepseek_v41::test_both_gguf_layouts[True]`、两个 `test_*_gdn_controls_round_to_bf16_in_grouped_head_order`；改动前就失败，不作为回归判据 |
| `python -m tools.convert --recipe x.py:fn` 解析失败 | 用驱动脚本 `tools/convert/convert_qsa27b.py`（它给绝对路径） |
| 重新提取资产后 `asset_sha256` 会变 | safetensors 的 `__metadata__` 由 Rust 的 HashMap 序列化，键序不稳定；manifest 必须与资产**同一次**生成，不可混用两次运行的产物 |

---

## 8. 风险与回退

| 风险 | 影响 | 缓解 / 回退 |
| --- | --- | --- |
| 可选绑定漏改 TP 路径 | 张量并行加载崩 | P1 同时改 `bind_qsa_indexer_share`；无 TP 环境下至少编译通过 |
| `std::optional<Tensor>` 改变结构体布局 | 其他翻译单元读到旧布局 | 一次性全量重编（不要增量只编部分目标） |
| indexer 与 trunk 旋转不一致 | 选择质量异常 | 不设独立字段，二者共用 `planned_rope_yarn`（D3）；作者仓库同语义 |
| 资产选错（v2 / vLSA） | 选择质量异常 | manifest 的 `source_donor` + `asset_sha256` 双校验；转换日志打印来源 |
| 需要整体回退 | — | `git revert <commit>`；资产改回 `indexer_tensors.safetensors`（v2，sha256 `c0f50809…`） |

---

## 9. 续行检查清单（新窗口第一次读这里）

1. `cd /f/ninfer-all-QSA && git log --oneline -6 && git status --short`
2. 看 `§4` 里第一个未勾选的阶段，从它的「改动文件」直接开工。
3. 若 `§4` 的勾选与 `git log` 不一致，以 **git log 为准**，并修正本文件。
4. 任何数值判断都先看 `§2`，不要重新推导 rope 表或张量切分。
5. 每完成一个阶段：勾选 → 追加一行进度 → commit（含本文件的更新）。

### 进度

| 日期 | 阶段 | 结果 |
| --- | --- | --- |
| 2026-11-21 | P0 | 规划完成；vLSA 的 rope 表确认为 YaRN-40（3e-8 误差），`PreNorm` 恒真的根因定位到 `text.cpp:1789` / `prefill.cpp:1075` |
| 2026-11-21 | P1 | `30f9a2493`：`optional_direct` + optional 传递链；QSA/attention/KV/loader 7 个测试全 PASS；5-role 产物加载正常 |
| 2026-11-21 | P2 | `ROLES_NO_PRE_NORM` + `norm_form` 元数据 + graft 由来源推导 role 集合；`test_qsa_graft.py` 12 项 PASS，全量 425 passed |
| 2026-11-21 | P3 | 新增 `extract_long_16k_indexer.py`（提取时自校验 rope）与 `test_long_16k_asset.py`（6 项）；资产改名为 `indexer_tensors-long-16k-1M-heavy.safetensors`，64 张量 / 104.88 MB，sha256 `25d756cd…`，manifest 的 rope 记录改称 `distillation_rope`（provenance） |
| 2026-11-21 | P4 | **撤销**：核实 `qwen38_27b.yaml` / `qwen38_27b_qsa.yaml` / `baselines/qsa.py` 后确认 YaRN-40 是模型级服务旋转，indexer 跟随 trunk；独立 `indexer_rope` 的全部改动已回退，改在 `text.cpp` 调用处写明理由 |
| 2026-11-21 | P5 | 资产与产物改用运行名；产物 1208 objects、64 indexer 张量、无 `key_pre_norm`、无 `indexer_rope`；`--qsa-top-blocks 512` 下文本/带图/19,835-token prefill 全通过；indexer 权重逐位 41,947,136/41,947,136 |
| 2026-11-21 | P6 | `MODIFICATIONS.md` 补 4 个新增文件（Total 32 added）与 4-role / `distillation_rope` 说明 |
