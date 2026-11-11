// qsa_indexer_append/select against an FP64 oracle of Qwen3.8-Flash-Next's block indexer: a
// sequence long enough for the selection to bind (650 blocks), fed as a prefill chunk that starts
// mid-block, decode steps across block boundaries and a verify-width chunk; the pooled keys
// pointwise and every selection as a set, up to near-ties of the 512th score. The decode steps
// replay one captured CUDA graph, the position advancing only in the device word the ops read.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim = 128, kHeads = 4, kProjection = 640, kTop = 512;
constexpr float kEps = 1e-6f;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// YaRN over a native window (factor above one), as Hugging Face's `yarn` rope type gives it with
// beta_fast 32 and beta_slow 1; factor one leaves the rotation unscaled.
struct Yarn {
    double factor                = 1.0;
    double native                = 0.0;
    double interpolation_factor  = 1.0;
    std::uint32_t interpolation_threshold = 0;
    [[nodiscard]] ops::RopeYarn op() const {
        return {static_cast<float>(factor), static_cast<std::uint32_t>(native),
                static_cast<float>(interpolation_factor), interpolation_threshold};
    }
    [[nodiscard]] bool active() const { return factor > 1.0 || interpolation_factor > 1.0; }
};

// Pair i's frequency. The unscaled route uses the Op's exp form; the active YaRN
// route uses yarn_schedule's pow form. Both match the engine's two branches.
float pair_frequency(int i, const Yarn& yarn) {
    if (!yarn.active()) return static_cast<float>(std::exp(-(2.0 * i / 64.0) * std::log(1e7)));
    const double base = std::pow(1e7, -2.0 * i / 64.0);
    if (!(yarn.factor > 1.0)) return static_cast<float>(base);
    constexpr double kPi = 3.14159265358979323846;
    const auto correction = [&](double rotations) {
        return 64.0 * std::log(yarn.native / (rotations * 2.0 * kPi)) / (2.0 * std::log(1e7));
    };
    const double lo = std::max(std::floor(correction(32.0)), 0.0);
    double hi       = std::min(std::ceil(correction(1.0)), 63.0);
    if (hi == lo) hi += 0.001;
    const double ramp = std::clamp((i - lo) / (hi - lo), 0.0, 1.0);
    return static_cast<float>(base * ((1.0 - ramp) + ramp / yarn.factor));
}

// Zero-centred RMSNorm then rotate-half. Pair i takes axis i % 3 of pos3 (the
// engine's interleaved MRoPE), with the same per-axis linear interpolation as
// rope.cuh's YaRN kernel. Three equal axes are exactly the scalar rotation.
std::vector<double> norm_rope(std::vector<double> x, const std::vector<float>& w,
                              const std::array<int, 3>& pos3, const Yarn& yarn) {
    double sum = 0;
    for (double v : x) sum += v * v;
    const double scale = 1.0 / std::sqrt(sum / kDim + double(kEps));
    for (int d = 0; d < kDim; ++d) x[d] = x[d] * scale * (1.0 + w[d]);
    const double attention = yarn.factor > 1.0 ? 0.1 * std::log(yarn.factor) + 1.0 : 1.0;
    const float threshold  = static_cast<float>(yarn.interpolation_threshold);
    const float inv_scale  = static_cast<float>(1.0 / yarn.interpolation_factor);
    for (int i = 0; i < 32; ++i) {
        const float frequency = pair_frequency(i, yarn);
        float scaled = static_cast<float>(pos3[i % 3]);
        if (yarn.interpolation_factor > 1.0 && scaled > threshold) {
            scaled = threshold + (scaled - threshold) * inv_scale;
        }
        const float angle = scaled * frequency;
        const double c = std::cos(double(angle)) * attention, s = std::sin(double(angle)) * attention;
        const double x1 = x[i], x2 = x[i + 32];
        x[i]      = x1 * c - x2 * s;
        x[i + 32] = x2 * c + x1 * s;
    }
    return x;
}

std::vector<double> norm_rope(std::vector<double> x, const std::vector<float>& w, int position,
                              const Yarn& yarn) {
    return norm_rope(std::move(x), w, std::array<int, 3>{position, position, position}, yarn);
}

// axis-major 3-axis positions of a token sequence; `distinct` gives an
// image/video-like t/h/w spread, otherwise three equal (text) axes.
std::vector<std::array<int, 3>> make_axes(int length, bool distinct) {
    std::vector<std::array<int, 3>> out(static_cast<std::size_t>(length));
    for (int p = 0; p < length; ++p) {
        out[static_cast<std::size_t>(p)] =
            distinct ? std::array<int, 3>{p, p / 7 + 1, p / 11 + 2}
                     : std::array<int, 3>{p, p, p};
    }
    return out;
}

struct Sequence {
    int length;
    std::vector<float> projection; // [length][640]
    std::vector<float> query_norm, key_norm;
    // QSA-27B's index_k_pre_norm: applied to each raw key before the 4-token mean. Empty selects
    // Flash-Next's path, which pools the raw projection keys.
    std::vector<float> key_pre_norm;
};

std::vector<double> pooled_oracle(const Sequence& seq, int block, const Yarn& yarn,
                                  const std::vector<std::array<int, 3>>* axes = nullptr) {
    std::vector<double> mean(kDim, 0.0);
    for (int i = 0; i < 4; ++i) {
        std::vector<double> key(kDim);
        for (int d = 0; d < kDim; ++d) {
            key[d] = seq.projection[static_cast<std::size_t>(block * 4 + i) * kProjection + kHeads * kDim + d];
        }
        if (!seq.key_pre_norm.empty()) {
            double sum = 0;
            for (double v : key) sum += v * v;
            const double scale = 1.0 / std::sqrt(sum / kDim + double(kEps));
            for (int d = 0; d < kDim; ++d) key[d] = key[d] * scale * (1.0 + seq.key_pre_norm[d]);
        }
        for (int d = 0; d < kDim; ++d) mean[d] += key[d];
    }
    for (double& v : mean) v /= 4.0;
    const int first = block * 4;
    const auto pos3 = axes != nullptr ? (*axes)[static_cast<std::size_t>(first)]
                                      : std::array<int, 3>{first, first, first};
    return norm_rope(mean, seq.key_norm, pos3, yarn);
}

int check_selection(const std::string& label, const Sequence& seq, const std::vector<std::vector<double>>& pooled,
                    int position, const int* got, int got_count, const Yarn& yarn, int column = -1,
                    const std::vector<std::array<int, 3>>* axes = nullptr) {
    const int blocks = std::min((position + 1) / 4, static_cast<int>(pooled.size()));
    std::vector<int> expected;
    std::vector<double> score(blocks);
    if (blocks <= kTop) {
        expected.resize(blocks);
        std::iota(expected.begin(), expected.end(), 0);
    } else {
        std::vector<std::vector<double>> q(kHeads);
        const auto qpos3 = axes != nullptr ? (*axes)[static_cast<std::size_t>(position)]
                                           : std::array<int, 3>{position, position, position};
        for (int h = 0; h < kHeads; ++h) {
            std::vector<double> raw(kDim);
            for (int d = 0; d < kDim; ++d) raw[d] = seq.projection[
                static_cast<std::size_t>(column < 0 ? position : column) * kProjection + h * kDim + d];
            q[h] = norm_rope(raw, seq.query_norm, qpos3, yarn);
        }
        for (int b = 0; b < blocks; ++b) {
            double s = 0;
            for (int h = 0; h < kHeads; ++h) {
                double dot = 0;
                for (int d = 0; d < kDim; ++d) dot += q[h][d] * pooled[b][d];
                s += std::max(dot, 0.0);
            }
            score[b] = s;
        }
        std::vector<int> order(blocks);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
        expected.assign(order.begin(), order.begin() + kTop);
        std::sort(expected.begin(), expected.end());
    }
    if (got_count != static_cast<int>(expected.size())) {
        std::cerr << label << ": selected " << got_count << " blocks, expected " << expected.size() << '\n';
        return 1;
    }
    std::vector<int> actual(got, got + got_count);
    if (!std::is_sorted(actual.begin(), actual.end()) ||
        std::adjacent_find(actual.begin(), actual.end()) != actual.end() ||
        (!actual.empty() && (actual.front() < 0 || actual.back() >= blocks))) {
        std::cerr << label << ": selection not in increasing block order\n";
        return 1;
    }
    if (actual == expected) return 0;
    // Only blocks tied with the 512th score within 1e-5 relative may differ.
    std::vector<double> sorted(score);
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    const double edge = sorted[kTop - 1];
    std::set<int> a(actual.begin(), actual.end()), e(expected.begin(), expected.end());
    for (int b = 0; b < blocks; ++b) {
        if (a.count(b) != e.count(b) && std::abs(score[b] - edge) > 1e-5 * std::abs(edge)) {
            std::cerr << label << ": block " << b << " (score " << score[b] << ", 512th " << edge
                      << ") selected=" << a.count(b) << " expected=" << e.count(b) << '\n';
            return 1;
        }
    }
    return 0;
}

int run(std::uint32_t seed, const Yarn& yarn = {}, bool pre_norm = false,
        const std::vector<std::array<int, 3>>* axes = nullptr) {
    Sequence seq{2600, std::vector<float>(static_cast<std::size_t>(2600) * kProjection), std::vector<float>(kDim), std::vector<float>(kDim),
                 pre_norm ? std::vector<float>(kDim) : std::vector<float>()};
    fill_uniform(seq.projection, seed, -2.0f, 2.0f);
    fill_uniform(seq.query_norm, seed + 1, -0.5f, 0.5f);
    fill_uniform(seq.key_norm, seed + 2, -0.5f, 0.5f);
    round_to_bf16(seq.projection);
    round_to_bf16(seq.query_norm);
    round_to_bf16(seq.key_norm);
    // Only a non-empty key_pre_norm selects the QSA-27B path; the Flash-Next runs must leave it
    // empty so the oracle pools the raw keys, as the op does when the weight is null.
    if (pre_norm) {
        fill_uniform(seq.key_pre_norm, seed + 3, -0.5f, 0.5f);
        round_to_bf16(seq.key_pre_norm);
    }
    std::vector<std::vector<double>> pooled(seq.length / 4);
    for (int b = 0; b < seq.length / 4; ++b) pooled[b] = pooled_oracle(seq, b, yarn, axes);

    const int capacity = 700;
    GuardedDeviceBuffer d_projection(seq.projection.size() * 2), d_qn(kDim * 2), d_kn(kDim * 2),
        d_kpn(kDim * 2), d_pooled(static_cast<std::size_t>(capacity) * kDim * 4), d_tail(3 * kDim * 4),
        d_selected(kProjection * 2 + static_cast<std::size_t>(kTop) * 64 * 4), d_counts(64 * 4),
        d_rope(static_cast<std::size_t>(3) * seq.length * 4), d_tail_axes(9 * 4);
    const auto bits = encode_bf16(seq.projection);
    d_projection.copy_from_host(bits.data(), d_projection.bytes());
    const auto qn = encode_bf16(seq.query_norm), kn = encode_bf16(seq.key_norm),
               kpn = encode_bf16(seq.key_pre_norm);
    d_qn.copy_from_host(qn.data(), qn.size() * sizeof(std::uint16_t));
    d_kn.copy_from_host(kn.data(), kn.size() * sizeof(std::uint16_t));
    if (!kpn.empty()) d_kpn.copy_from_host(kpn.data(), kpn.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemset(d_pooled.data(), 0, d_pooled.bytes()));
    CUDA_CHECK(cudaMemset(d_tail.data(), 0, d_tail.bytes()));
    // tail_axes is persistent across calls: append writes it, the next append
    // reads it when a completed block starts inside the incomplete tail.
    CUDA_CHECK(cudaMemset(d_tail_axes.data(), 0, d_tail_axes.bytes()));
    Tensor t_qn(d_qn.data(), DType::BF16, {kDim}), t_kn(d_kn.data(), DType::BF16, {kDim}),
        t_kpn(d_kpn.data(), DType::BF16, {kDim});
    Tensor t_pooled(d_pooled.data(), DType::FP32, {kDim, capacity});
    Tensor t_tail(d_tail.data(), DType::FP32, {kDim, 3});
    GuardedDeviceBuffer d_first(4);
    const Tensor t_first(d_first.data(), DType::I32, {1});
    const ops::QsaIndexerWeights weights{&t_qn, &t_kn, pre_norm ? &t_kpn : nullptr};
    WorkspaceArena workspace(ops::qsa_indexer_select_workspace_bytes(64, capacity));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraphExec_t decode = nullptr;

    int failures = 0;
    // Chunks: a prefill that ends mid-block, decode steps across block edges, a verify width, the
    // rest as chunks that start and end mid-block.
    std::vector<int> chunks{2050, 1, 1, 1, 1, 1, 9, 37, 64, 64, 64, 64, 64, 64, 64, 64, 2};
    int begin = 0;
    for (const int count : chunks) {
        if (begin + count > seq.length) break;
        // Decode steps read their projection column from one fixed place, as a graph replay does.
        auto* column = static_cast<std::uint16_t*>(d_projection.data()) + static_cast<std::size_t>(begin) * kProjection;
        if (count == 1) {
            CUDA_CHECK(cudaMemcpyAsync(d_selected.data(), column, kProjection * 2, cudaMemcpyDeviceToDevice, stream));
            column = static_cast<std::uint16_t*>(d_selected.data());
        }
        CUDA_CHECK(cudaMemcpyAsync(d_first.data(), &begin, 4, cudaMemcpyHostToDevice, stream));
        Tensor t_rope(d_rope.data(), DType::I32, {count, 3});
        Tensor t_tail_axes(d_tail_axes.data(), DType::I32, {3, 3});
        ops::QsaIndexerPositions positions{};
        if (axes != nullptr) {
            std::vector<int> host(static_cast<std::size_t>(3) * count);
            for (int axis = 0; axis < 3; ++axis) {
                for (int t = 0; t < count; ++t) {
                    host[static_cast<std::size_t>(axis) * count + t] =
                        (*axes)[static_cast<std::size_t>(begin + t)][axis];
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(d_rope.data(), host.data(), host.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            positions.rope_positions = &t_rope;
            positions.tail_axes      = &t_tail_axes;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        Tensor t_projection(column, DType::BF16, {kProjection, count});
        Tensor t_selected(static_cast<std::byte*>(d_selected.data()) + (count == 1 ? kProjection * 2 : 0), DType::I32, {kTop, count});
        Tensor t_counts(d_counts.data(), DType::I32, {count});
        const auto step = [&] {
            ops::qsa_indexer_append(t_projection, t_first, weights, kEps, yarn.op(), t_pooled,
                                    t_tail, stream, positions);
            if (count <= 64) {
                ops::qsa_indexer_select(t_projection, t_first, weights, kEps, yarn.op(), t_pooled,
                                        workspace, t_selected, t_counts, stream, positions);
            }
        };
        if (count == 1) {
            if (decode == nullptr) {
                cudaGraph_t graph = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                step();
                CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&decode, graph, 0));
                CUDA_CHECK(cudaGraphDestroy(graph));
            }
            CUDA_CHECK(cudaGraphLaunch(decode, stream));
        } else {
            step();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (count <= 64) {
            const std::byte* selected_base = static_cast<const std::byte*>(t_selected.data);
            const auto selected = from_device<int>(selected_base, static_cast<std::size_t>(kTop) * count);
            const auto counts   = from_device<int>(d_counts.data(), count);
            for (int t = 0; t < count; ++t) {
                failures += check_selection("qsa select p=" + std::to_string(begin + t), seq, pooled, begin + t,
                                            selected.data() + static_cast<std::size_t>(t) * kTop, counts[t],
                                            yarn, -1, axes);
            }
        }
        begin += count;
    }
    if (decode != nullptr) CUDA_CHECK(cudaGraphExecDestroy(decode));
    CUDA_CHECK(cudaStreamDestroy(stream));
    const int complete = begin / 4;
    const auto got = from_device<float>(d_pooled.data(), static_cast<std::size_t>(complete) * kDim);
    std::vector<double> expected;
    for (int b = 0; b < complete; ++b) expected.insert(expected.end(), pooled[b].begin(), pooled[b].end());
    failures += verify_pointwise(std::string("qsa pooled keys") + (pre_norm ? " pre-norm" : "") +
                                     (yarn.factor > 1.0 ? " yarn" : ""),
                                 std::vector<double>(got.begin(), got.end()), expected, {2e-5, 2e-5});
    for (auto* buffer : {&d_projection, &d_qn, &d_kn, &d_kpn, &d_pooled, &d_tail, &d_selected,
                         &d_counts, &d_first, &d_rope, &d_tail_axes}) {
        failures += buffer->verify_guards("qsa_indexer");
    }
    return failures;
}

int selection_case(int tokens, int capacity, int first, bool ties = false, const Yarn& yarn = {}) {
    Sequence seq{tokens, std::vector<float>(std::size_t(tokens) * kProjection),
                 std::vector<float>(kDim), std::vector<float>(kDim)};
    fill_uniform(seq.projection, 9100 + tokens, -2.0f, 2.0f);
    fill_uniform(seq.query_norm, 9110, -0.5f, 0.5f);
    round_to_bf16(seq.projection);
    round_to_bf16(seq.query_norm);
    if (ties) { std::fill(seq.projection.begin(), seq.projection.end(), 0.0f); }
    std::vector<float> keys(std::size_t(capacity) * kDim);
    fill_uniform(keys, 9111, -2.0f, 2.0f);
    // Public pooled keys are FP32 inputs to select; the oracle uses those exact values.
    std::vector<std::vector<double>> pooled(capacity, std::vector<double>(kDim));
    for (int b = 0; b < capacity; ++b) {
        std::copy_n(keys.data() + std::size_t(b) * kDim, kDim, pooled[b].begin());
    }
    GuardedDeviceBuffer d_projection(seq.projection.size() * 2), d_qn(kDim * 2),
        d_pooled(keys.size() * 4), d_first(4), d_selected(std::size_t(tokens) * kTop * 4),
        d_counts(tokens * 4);
    const auto bits = encode_bf16(seq.projection), norms = encode_bf16(seq.query_norm);
    d_projection.copy_from_host(bits.data(), d_projection.bytes());
    d_qn.copy_from_host(norms.data(), d_qn.bytes());
    d_pooled.copy_from_host(keys.data(), d_pooled.bytes());
    Tensor projection(d_projection.data(), DType::BF16, {kProjection, tokens});
    Tensor norm(d_qn.data(), DType::BF16, {kDim});
    Tensor positions(d_first.data(), DType::I32, {1});
    Tensor pool(d_pooled.data(), DType::FP32, {kDim, capacity});
    Tensor selected(d_selected.data(), DType::I32, {kTop, tokens});
    Tensor counts(d_counts.data(), DType::I32, {tokens});
    const ops::QsaIndexerWeights weights{&norm, nullptr};
    WorkspaceArena workspace(ops::qsa_indexer_select_workspace_bytes(tokens, capacity));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int failures = 0;
    {
        const auto step = [&] {
            ops::qsa_indexer_select(projection, positions, weights, kEps, yarn.op(), pool, workspace,
                                   selected, counts, stream);
        };
        cudaGraphExec_t graph = nullptr;
        for (int replay = 0; replay < 2; ++replay) {
            const int position = first + replay * 3;
            d_first.copy_from_host(&position, 4);
            if (replay == 0) {
                step();
            } else {
                cudaGraph_t captured = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                step();
                CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
                CUDA_CHECK(cudaGraphInstantiate(&graph, captured, 0));
                CUDA_CHECK(cudaGraphDestroy(captured));
                CUDA_CHECK(cudaGraphLaunch(graph, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto got = from_device<int>(d_selected.data(), std::size_t(tokens) * kTop);
            const auto sizes = from_device<int>(d_counts.data(), tokens);
            for (int t = 0; t < tokens; ++t) {
                failures += check_selection("qsa T=" + std::to_string(tokens) +
                    " p=" + std::to_string(position + t) + (yarn.factor > 1.0 ? " yarn" : ""),
                    seq, pooled, position + t, got.data() + std::size_t(t) * kTop, sizes[t], yarn, t);
                if (ties) {
                    for (int i = 0; i < sizes[t]; ++i) {
                        if (got[std::size_t(t) * kTop + i] != i) { ++failures; break; }
                    }
                }
            }
            if (replay == 1) {
                CUDA_CHECK(cudaGraphLaunch(graph, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto same = from_device<int>(d_selected.data(), std::size_t(tokens) * kTop);
                const auto same_sizes = from_device<int>(d_counts.data(), tokens);
                if (same_sizes != sizes) { ++failures; }
                for (int t = 0; t < tokens; ++t) {
                    if (sizes[t] >= 0 && sizes[t] <= kTop &&
                        !std::equal(got.begin() + std::size_t(t) * kTop,
                                    got.begin() + std::size_t(t) * kTop + sizes[t],
                                    same.begin() + std::size_t(t) * kTop)) { ++failures; }
                }
                // Changed device position after capture must change causal visibility and RoPE.
                const int next = position + 2;
                d_first.copy_from_host(&next, 4);
                CUDA_CHECK(cudaGraphLaunch(graph, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto repeated = from_device<int>(d_selected.data(), std::size_t(tokens) * kTop);
                const auto repeated_sizes = from_device<int>(d_counts.data(), tokens);
                for (int t = 0; t < tokens; ++t) {
                    failures += check_selection("qsa changed-position graph", seq, pooled, next + t,
                        repeated.data() + std::size_t(t) * kTop, repeated_sizes[t], yarn, t);
                }
            }
        }
        CUDA_CHECK(cudaGraphExecDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (auto* buffer : {&d_projection, &d_qn, &d_pooled, &d_first, &d_selected, &d_counts}) {
        failures += buffer->verify_guards("qsa selection routes");
    }
    return failures;
}

// Y4: three equal M-RoPE axes must be bit-for-bit identical to the original
// scalar path, for pooled keys, tail, selected blocks and counts. This is the
// self-contained YaRN compatibility check (no external multimodal reference).
int equal_axes_equivalence(const Yarn& yarn, bool pre_norm) {
    constexpr int kTokens   = 64;
    constexpr int kCapacity = 700;
    constexpr int kFirst    = 2048; // block-aligned, so tail_axes is never read
    Sequence seq{kTokens, std::vector<float>(static_cast<std::size_t>(kTokens) * kProjection),
                 std::vector<float>(kDim), std::vector<float>(kDim),
                 pre_norm ? std::vector<float>(kDim) : std::vector<float>()};
    fill_uniform(seq.projection, 9700, -2.0f, 2.0f);
    fill_uniform(seq.query_norm, 9701, -0.5f, 0.5f);
    fill_uniform(seq.key_norm, 9702, -0.5f, 0.5f);
    round_to_bf16(seq.projection);
    round_to_bf16(seq.query_norm);
    round_to_bf16(seq.key_norm);
    if (pre_norm) {
        fill_uniform(seq.key_pre_norm, 9703, -0.5f, 0.5f);
        round_to_bf16(seq.key_pre_norm);
    }
    const std::size_t pooled_bytes = static_cast<std::size_t>(kCapacity) * kDim * 4;
    const std::size_t tail_bytes   = 3 * kDim * 4;
    const std::size_t sel_bytes    = static_cast<std::size_t>(kTop) * kTokens * 4;
    GuardedDeviceBuffer d_projection(seq.projection.size() * 2), d_qn(kDim * 2), d_kn(kDim * 2),
        d_kpn(kDim * 2), d_first(4), d_rope(3 * kTokens * 4), d_tail_axes(9 * 4),
        a_pooled(pooled_bytes), a_tail(tail_bytes), a_sel(sel_bytes), a_cnt(kTokens * 4),
        b_pooled(pooled_bytes), b_tail(tail_bytes), b_sel(sel_bytes), b_cnt(kTokens * 4);
    const auto bits = encode_bf16(seq.projection), qn = encode_bf16(seq.query_norm),
               kn = encode_bf16(seq.key_norm), kpn = encode_bf16(seq.key_pre_norm);
    d_projection.copy_from_host(bits.data(), d_projection.bytes());
    d_qn.copy_from_host(qn.data(), d_qn.bytes());
    d_kn.copy_from_host(kn.data(), d_kn.bytes());
    if (!kpn.empty()) d_kpn.copy_from_host(kpn.data(), d_kpn.bytes());
    std::vector<int> host(static_cast<std::size_t>(3) * kTokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (int t = 0; t < kTokens; ++t) {
            host[static_cast<std::size_t>(axis) * kTokens + t] = kFirst + t;
        }
    }
    d_rope.copy_from_host(host.data(), host.size() * sizeof(int));
    d_first.copy_from_host(&kFirst, 4);
    for (auto* b : {&a_pooled, &b_pooled}) CUDA_CHECK(cudaMemset(b->data(), 0, b->bytes()));
    for (auto* b : {&a_tail, &b_tail, &d_tail_axes}) CUDA_CHECK(cudaMemset(b->data(), 0, b->bytes()));

    Tensor t_proj(d_projection.data(), DType::BF16, {kProjection, kTokens});
    Tensor t_qn(d_qn.data(), DType::BF16, {kDim}), t_kn(d_kn.data(), DType::BF16, {kDim}),
        t_kpn(d_kpn.data(), DType::BF16, {kDim});
    Tensor t_first(d_first.data(), DType::I32, {1});
    Tensor t_rope(d_rope.data(), DType::I32, {kTokens, 3});
    Tensor t_tail_axes(d_tail_axes.data(), DType::I32, {3, 3});
    Tensor a_pool(a_pooled.data(), DType::FP32, {kDim, kCapacity}),
        b_pool(b_pooled.data(), DType::FP32, {kDim, kCapacity});
    Tensor a_t(a_tail.data(), DType::FP32, {kDim, 3}), b_t(b_tail.data(), DType::FP32, {kDim, 3});
    Tensor a_s(a_sel.data(), DType::I32, {kTop, kTokens}), b_s(b_sel.data(), DType::I32, {kTop, kTokens});
    Tensor a_c(a_cnt.data(), DType::I32, {kTokens}), b_c(b_cnt.data(), DType::I32, {kTokens});
    const ops::QsaIndexerWeights weights{&t_qn, &t_kn, pre_norm ? &t_kpn : nullptr};
    const ops::QsaIndexerPositions positions{&t_rope, &t_tail_axes, nullptr};
    WorkspaceArena workspace(ops::qsa_indexer_select_workspace_bytes(kTokens, kCapacity));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ops::qsa_indexer_append(t_proj, t_first, weights, kEps, yarn.op(), a_pool, a_t, stream);
    ops::qsa_indexer_select(t_proj, t_first, weights, kEps, yarn.op(), a_pool, workspace, a_s, a_c, stream);
    ops::qsa_indexer_append(t_proj, t_first, weights, kEps, yarn.op(), b_pool, b_t, stream, positions);
    ops::qsa_indexer_select(t_proj, t_first, weights, kEps, yarn.op(), b_pool, workspace, b_s, b_c,
                            stream, positions);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int failures = 0;
    const auto pa = from_device<float>(a_pooled.data(), kCapacity * kDim);
    const auto pb = from_device<float>(b_pooled.data(), kCapacity * kDim);
    const auto ta = from_device<float>(a_tail.data(), 3 * kDim);
    const auto tb = from_device<float>(b_tail.data(), 3 * kDim);
    const auto sa = from_device<int>(a_sel.data(), static_cast<std::size_t>(kTop) * kTokens);
    const auto sb = from_device<int>(b_sel.data(), static_cast<std::size_t>(kTop) * kTokens);
    const auto ca = from_device<int>(a_cnt.data(), kTokens);
    const auto cb = from_device<int>(b_cnt.data(), kTokens);
    for (const auto& [label, equal] : {std::pair<const char*, bool>{"pooled", pa == pb},
                                       {"tail", ta == tb}, {"selected", sa == sb},
                                       {"counts", ca == cb}}) {
        if (!equal) {
            std::cerr << "equal-axes equivalence: " << label << " differs from the scalar path\n";
            ++failures;
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (auto* b : {&d_projection, &d_qn, &d_kn, &d_kpn, &d_first, &d_rope, &d_tail_axes,
                    &a_pooled, &a_tail, &a_sel, &a_cnt, &b_pooled, &b_tail, &b_sel, &b_cnt}) {
        failures += b->verify_guards("equal axes");
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run(8100u);
    failures += run(8300u, {}, true);
    // Scalar below 64 tokens (80 on an A100), narrow and wide SIMT tiles or the A100's tensor tile
    // above, by the device's SM count: each card meets the routes it selects.
    for (int tokens : {1, 3, 4, 5, 8, 9, 15, 16, 17, 37, 63, 64, 79, 80, 127, 128, 129, 512}) {
        failures += selection_case(tokens, 700, 2050);
    }
    // Nine 64-block columns: narrow tiles on a card of 19 or more SMs (on an A100 at 80 tokens).
    failures += selection_case(64, 576, 2200);
    failures += selection_case(80, 576, 2200);
    failures += selection_case(5, 512, 2047);
    failures += selection_case(5, 700, 2400, true);
    failures += selection_case(128, 700, 2400, true);
    failures += selection_case(128, 8192, 32700);
    failures += selection_case(129, 32768, 130970);
    failures += selection_case(512, 32768, 0);
    failures += selection_case(512, 32768, 1800);
    // YaRN at the factor of Flash-Next's 589,824-token window: the whole prefill / decode / verify
    // flow, the scalar and tiled scoring routes, and queries near that window's end.
    const Yarn yarn{2.25, 262144.0};
    failures += run(8200u, yarn);
    failures += run(8400u, yarn, true);
    failures += selection_case(5, 700, 2050, false, yarn);
    failures += selection_case(129, 32768, 130970, false, yarn);
    failures += selection_case(5, 147500, 589000, false, yarn);
    failures += selection_case(128, 147500, 589000, false, yarn);
    // V2: interleaved 3-axis M-RoPE. Y4 - three equal axes must be bit-for-bit
    // identical to the scalar path, with and without YaRN / interpolation / pre-norm.
    // Y5/Y9 - distinct t/h/w axes against the 3D FP64 oracle, including YaRN x 3D,
    // interpolation x 3D, cross-call tail_axes and CUDA Graph replay (inside run()).
    const auto equal_axes = make_axes(2600, false);
    const auto mrope_axes = make_axes(2600, true);
    for (const auto& cfg : {Yarn{}, Yarn{2.25, 262144.0}, Yarn{4.0, 262144.0},
                            Yarn{2.25, 262144.0, 3.0, 2050}}) {
        failures += equal_axes_equivalence(cfg, false);
        failures += equal_axes_equivalence(cfg, true);
    }
    failures += run(8500u, {}, false, &equal_axes);
    failures += run(8600u, {}, true, &mrope_axes);
    failures += run(8700u, Yarn{2.25, 262144.0}, false, &mrope_axes);
    failures += run(8800u, Yarn{4.0, 262144.0}, true, &mrope_axes);
    failures += run(8900u, Yarn{2.25, 262144.0, 3.0, 2050}, false, &mrope_axes);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " qsa_indexer\n";
    return failures == 0 ? 0 : 1;
}
