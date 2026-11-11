#include "core/layout.h"
#include "models/qwen3_5/state/state_image.h"

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::StateImageSpec base_spec() {
    return q36::StateImageSpec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = 2,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 7,
    };
}

q36::StateImageDeviceLayout plan(bool dflash) {
    q36::StateImageSpec spec = base_spec();
    if (dflash) {
        spec.dflash_local =
            q36::DFlashLocalStateSpec{.layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    ninfer::LayoutBuilder builder;
    return q36::plan_state_image_device_pool(builder, spec);
}

q36::StateImageDeviceLayout plan_qsa() {
    q36::StateImageSpec spec = base_spec();
    spec.qsa                 = q36::QsaStateSpec{.layers = 3, .head_dim = 16, .blocks = 7, .tail = 3};
    ninfer::LayoutBuilder builder;
    q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    // As startup.cpp does: the shared indexer planes live on rank 0 (pooled prefix, live tail
    // and scratch tail in one combined plane per layer, plus the tail axes), and the image's QSA
    // region binds them.
    std::vector<ninfer::TensorRegion> planes;
    std::vector<ninfer::TensorRegion> axes;
    for (std::uint32_t layer = 0; layer < spec.qsa->layers; ++layer) {
        (void)layer;
        planes.push_back(builder.add_tensor(
            ninfer::DType::FP32,
            {spec.qsa->head_dim,
             static_cast<std::int32_t>(spec.qsa->blocks + 2U * spec.qsa->tail)},
            256, "QSA indexer pooled keys, tail and scratch"));
        axes.push_back(builder.add_tensor(ninfer::DType::I32, {3, 3}, 256,
                                          "QSA indexer tail M-RoPE axes"));
    }
    layout.qsa = q36::QsaStateDeviceLayout{.planes    = std::move(planes),
                                           .tail_axes = std::move(axes),
                                           .blocks    = spec.qsa->blocks,
                                           .tail      = spec.qsa->tail};
    return layout;
}

} // namespace

int main() {
    const q36::StateImageDeviceLayout common = plan(false);
    const auto common_work                   = q36::state_image_transfer_work(common.host);
    expect(common_work.payload_bytes == common.host.linear_conv.bytes +
                                                common.host.linear_recurrent.bytes +
                                                common.host.continuation_hidden.bytes,
           "common StateImage work payload includes layout padding");
    expect(common_work.copy_operations == 2 * common.host.spec.linear.layers + 1,
           "common StateImage work does not match physical CUDA copies");

    const q36::StateImageDeviceLayout dflash = plan(true);
    const auto full_work                     = q36::state_image_transfer_work(dflash.host);
    const auto local_work                    = q36::dflash_local_transfer_work(dflash.host);
    const std::uint64_t local_bytes =
        2ULL * dflash.host.dflash_local_layer_bytes * dflash.host.spec.dflash_local->layers;
    expect(local_work.payload_bytes == local_bytes &&
               local_work.copy_operations == 2 * dflash.host.spec.dflash_local->layers,
           "DFlash-local work does not match physical K/V layer copies");
    expect(full_work.payload_bytes == common_work.payload_bytes + local_work.payload_bytes &&
               full_work.copy_operations ==
                   common_work.copy_operations + local_work.copy_operations,
           "full DFlash StateImage work is not common plus local state");

    // QSA: the image carries every full-attention layer's pooled block keys, the live tail and
    // the tail axes, on top of the common state.
    const q36::StateImageDeviceLayout qsa = plan_qsa();
    const auto qsa_work                   = q36::state_image_transfer_work(qsa.host);
    expect(common.qsa == std::nullopt, "a StateImage without QSA carries no QSA region");
    expect(qsa.qsa && qsa.qsa->planes.size() == 3 && qsa.qsa->tail_axes.size() == 3,
           "QSA device layout binds one pooled plane and tail axes per full-attention layer");
    expect(qsa.qsa && qsa.qsa->blocks == 7 && qsa.qsa->tail == 3,
           "QSA device layout carries the pooled block count and tail length");
    expect(qsa.host.spec.qsa && qsa.host.spec.qsa->layers == 3 &&
               qsa.host.spec.qsa->head_dim == 16 && qsa.host.spec.qsa->blocks == 7 &&
               qsa.host.spec.qsa->tail == 3,
           "QSA host layout echoes the QSA spec");
    expect(qsa.host.qsa_pooled && qsa.host.qsa_tail && qsa.host.qsa_tail_axes,
           "QSA host layout is missing a pooled, tail or axes region");
    if (qsa.host.qsa_pooled && qsa.host.qsa_tail && qsa.host.qsa_tail_axes) {
        expect(qsa.host.qsa_pooled_layer_bytes == std::uint64_t(16) * 7 * 4 &&
                   qsa.host.qsa_tail_layer_bytes == std::uint64_t(16) * 3 * 4 &&
                   qsa.host.qsa_tail_axes_layer_bytes == std::uint64_t(3) * 3 * 4,
               "QSA host layer byte counts are not FP32 pooled/tail and I32 axes");
        expect(qsa.host.qsa_pooled->bytes == 3 * qsa.host.qsa_pooled_layer_bytes &&
                   qsa.host.qsa_tail->bytes == 3 * qsa.host.qsa_tail_layer_bytes &&
                   qsa.host.qsa_tail_axes->bytes == 3 * qsa.host.qsa_tail_axes_layer_bytes,
               "QSA host regions are not contiguous by layer");
    }
    const std::uint64_t qsa_bytes =
        std::uint64_t(qsa.host.qsa_pooled_layer_bytes + qsa.host.qsa_tail_layer_bytes +
                      qsa.host.qsa_tail_axes_layer_bytes) * 3;
    expect(qsa_work.payload_bytes == common_work.payload_bytes + qsa_bytes,
           "QSA StateImage work is not common state plus the QSA region");
    expect(qsa_work.copy_operations == common_work.copy_operations + 3 * 3,
           "QSA StateImage work does not add one pooled, tail and axes copy per layer");
    expect(qsa.host.image_bytes >= qsa_work.payload_bytes,
           "QSA StateImage host image is smaller than its payload");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
