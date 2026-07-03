#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>

#include <sst/elements/golem/sfu/sfu.h>

namespace SST {
namespace Golem {

namespace {

using SoftmaxReducerKey = std::pair<uint64_t, uint32_t>;
constexpr uint32_t GOLEM_DTYPE_FP32_VALUE = 1;
constexpr uint32_t GOLEM_SFU_PRIMITIVE_FLAG_REPEAT_CHUNK = 0x1;
constexpr uint32_t GOLEM_SFU_PRIMITIVE_BATCH_MAX_DESCS = 64;

struct SoftmaxReducerRowState {
    double m_acc = -std::numeric_limits<double>::infinity();
    double l_acc = 0.0;
    uint32_t partials_seen = 0;
    uint32_t n_tiles_expected = 0;
    uint32_t normalizes_done = 0;
    bool ready = false;
};

std::map<SoftmaxReducerKey, SoftmaxReducerRowState>& softmaxReducerRows()
{
    static std::map<SoftmaxReducerKey, SoftmaxReducerRowState> rows;
    return rows;
}

template <typename T>
void appendBytes(std::vector<uint8_t>& out, const T& value)
{
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), raw, raw + sizeof(T));
}

size_t tilePackedIndex(const SFUSoftmaxTileDesc& desc, uint32_t row, uint32_t col)
{
    return static_cast<size_t>(col) * desc.block_m + row;
}

uint32_t effectiveStride(uint32_t strideBytes)
{
    return strideBytes == 0 ? static_cast<uint32_t>(sizeof(float)) : strideBytes;
}

} // namespace

SFU::SFU(ComponentId_t id, SST::Params& params)
    : SFUAPI(id, params),
      coreId_(params.find<uint32_t>("core_id", 0)),
      activeWorkerCores_(params.find<uint32_t>("active_worker_cores", 1)),
      maxInflight_(params.find<uint32_t>("max_inflight", 8)),
      statsLatency_(params.find<uint32_t>("stats_latency", 1)),
      mergeLatency_(params.find<uint32_t>("merge_latency", 1)),
      normalizeLatency_(params.find<uint32_t>("normalize_latency", 1)),
      inflight_(0),
      verbose_(params.find<int>("verbose", 0)),
      globalMem_(nullptr),
      output_("Golem::SFU[@p:@l]: ", verbose_, 0, SST::Output::STDOUT)
{
    if (activeWorkerCores_ == 0) {
        activeWorkerCores_ = 1;
    }
    if (maxInflight_ == 0) {
        maxInflight_ = 1;
    }

    statOpsIssued_ = registerStatistic<uint64_t>("sfu_ops_issued");
    statSoftmaxRows_ = registerStatistic<uint64_t>("sfu_softmax_rows");
    statSoftmaxTiles_ = registerStatistic<uint64_t>("sfu_softmax_tiles");
    statPrimitiveElems_ = registerStatistic<uint64_t>("sfu_primitive_elems");
    statPartialSubmits_ = registerStatistic<uint64_t>("sfu_partial_submits");
    statPartialDone_ = registerStatistic<uint64_t>("sfu_partial_done");
    statCreditStalls_ = registerStatistic<uint64_t>("sfu_credit_stalls");
    statCrossTileWaitCycles_ = registerStatistic<uint64_t>("sfu_cross_tile_wait_cycles");
    statRetryEvents_ = registerStatistic<uint64_t>("sfu_retry_events");
}

void SFU::finish()
{
    if (verbose_ > 0) {
        output_.verbose(CALL_INFO, 1, 0,
            "SFU finish core=%" PRIu32 " inflight=%" PRIu32 "\n",
            coreId_, inflight_);
    }
}

bool SFU::issueSoftmaxTile(uint64_t descAddr, uint64_t tag)
{
    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    SoftmaxOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;
    state.normalizeReady = false;

    if (!readSoftmaxDescriptor(descAddr, &state.desc)) {
        state.status = (globalMem_ == nullptr) ? SFUStatus::GlobalMemoryUnavailable
                                               : SFUStatus::InvalidDescriptor;
    } else {
        state.status = validateSoftmaxDescriptor(state.desc);
    }

    if (state.status == SFUStatus::Success && !readInputTile(state.desc, &state.inputTile)) {
        state.status = SFUStatus::InvalidDescriptor;
    }

    if (state.status == SFUStatus::Success) {
        computeTileStats(&state);
        state.normalizeReady = mergeTileStats(&state);
        state.status = SFUStatus::Pending;
    }

    pendingSoftmaxOps_[tag] = state;

    ++inflight_;
    statOpsIssued_->addData(1);
    statSoftmaxTiles_->addData(1);
    if (state.status == SFUStatus::Success || state.status == SFUStatus::Pending) {
        statPartialSubmits_->addData(state.rowStats.size());
    }
    return true;
}

bool SFU::issuePrimitive(uint64_t descAddr, uint64_t tag)
{
    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    PrimitiveOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;

    if (!readPrimitiveDescriptor(descAddr, &state.desc)) {
        state.status = (globalMem_ == nullptr) ? SFUStatus::GlobalMemoryUnavailable
                                               : SFUStatus::InvalidDescriptor;
    } else {
        state.status = validatePrimitiveDescriptor(state.desc);
    }

    if (state.status == SFUStatus::Success && !readPrimitiveInput(state.desc, &state.input0)) {
        state.status = SFUStatus::InvalidDescriptor;
    }
    if (state.status == SFUStatus::Success && !executePrimitive(&state)) {
        state.status = SFUStatus::InvalidDescriptor;
    }
    if (state.status == SFUStatus::Success && !writePrimitiveOutput(state.desc, state.output)) {
        state.status = SFUStatus::InvalidDescriptor;
    }

    pendingPrimitiveOps_[tag] = state;

    ++inflight_;
    statOpsIssued_->addData(1);
    if (state.status == SFUStatus::Success) {
        statPrimitiveElems_->addData(primitiveProcessedElems(state.desc));
    } else {
        statPrimitiveElems_->addData(0);
    }
    return true;
}

bool SFU::issuePrimitiveBatch(uint64_t descAddr, uint64_t tag)
{
    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    PrimitiveBatchOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;
    state.processedElems = 0;

    if (!readPrimitiveBatchDescriptor(descAddr, &state.desc)) {
        state.status = (globalMem_ == nullptr) ? SFUStatus::GlobalMemoryUnavailable
                                               : SFUStatus::InvalidDescriptor;
    } else {
        state.status = validatePrimitiveBatchDescriptor(state.desc);
    }

    if (state.status == SFUStatus::Success) {
        for (uint32_t i = 0; i < state.desc.desc_count; ++i) {
            const uint64_t childDescAddr =
                state.desc.desc_array_gm_addr +
                static_cast<uint64_t>(i) * sizeof(SFUPrimitiveDesc);
            uint64_t childElems = 0;
            if (!executePrimitiveDesc(childDescAddr, tag + i + 1, &childElems)) {
                state.status = SFUStatus::InvalidDescriptor;
                break;
            }
            state.processedElems += childElems;
        }
    }

    pendingPrimitiveBatchOps_[tag] = state;

    ++inflight_;
    statOpsIssued_->addData(1);
    statPrimitiveElems_->addData(state.processedElems);
    return true;
}

bool SFU::wait(uint64_t tag, uint64_t* status)
{
    SFUStatus opStatus = SFUStatus::Success;
    auto it = pendingSoftmaxOps_.find(tag);
    if (it != pendingSoftmaxOps_.end()) {
        SoftmaxOpState& state = it->second;
        if (state.status == SFUStatus::Pending && tileGlobalStatsReady(state)) {
            state.normalizeReady = true;
            state.status = normalizeTile(&state) ? SFUStatus::Success : SFUStatus::InvalidDescriptor;
            if (state.status == SFUStatus::Success) {
                statPartialDone_->addData(state.rowStats.size());
            }
        }

        opStatus = state.status;
        if (opStatus == SFUStatus::Pending) {
            statCrossTileWaitCycles_->addData(1);
            if (status != nullptr) {
                *status = static_cast<uint64_t>(SFUStatus::Pending);
            }
            return false;
        }

        pendingSoftmaxOps_.erase(it);
    } else {
        auto primitiveIt = pendingPrimitiveOps_.find(tag);
        if (primitiveIt != pendingPrimitiveOps_.end()) {
            opStatus = primitiveIt->second.status;
            pendingPrimitiveOps_.erase(primitiveIt);
        } else {
            auto batchIt = pendingPrimitiveBatchOps_.find(tag);
            if (batchIt != pendingPrimitiveBatchOps_.end()) {
                opStatus = batchIt->second.status;
                pendingPrimitiveBatchOps_.erase(batchIt);
            }
        }
    }

    if (inflight_ > 0) {
        --inflight_;
    }
    if (status != nullptr) {
        *status = static_cast<uint64_t>(opStatus);
    }
    return true;
}

bool SFU::readPrimitiveDescriptor(uint64_t descAddr, SFUPrimitiveDesc* desc)
{
    if (globalMem_ == nullptr || desc == nullptr || descAddr == 0) {
        return false;
    }

    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(descAddr, sizeof(SFUPrimitiveDesc), raw);
    if (raw.size() != sizeof(SFUPrimitiveDesc)) {
        return false;
    }

    std::memcpy(desc, raw.data(), sizeof(SFUPrimitiveDesc));
    return true;
}

bool SFU::readPrimitiveBatchDescriptor(uint64_t descAddr, SFUPrimitiveBatchDesc* desc)
{
    if (globalMem_ == nullptr || desc == nullptr || descAddr == 0) {
        return false;
    }

    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(descAddr, sizeof(SFUPrimitiveBatchDesc), raw);
    if (raw.size() != sizeof(SFUPrimitiveBatchDesc)) {
        return false;
    }

    std::memcpy(desc, raw.data(), sizeof(SFUPrimitiveBatchDesc));
    return true;
}

SFUStatus SFU::validatePrimitiveDescriptor(const SFUPrimitiveDesc& desc) const
{
    if (desc.dtype != GOLEM_DTYPE_FP32_VALUE) {
        return SFUStatus::UnsupportedElemBytes;
    }
    if (desc.input0_gm_addr == 0 || desc.output_gm_addr == 0 || desc.elem_count == 0) {
        return SFUStatus::InvalidShape;
    }
    if (effectiveStride(desc.input0_stride_bytes) < sizeof(float) ||
        effectiveStride(desc.output_stride_bytes) < sizeof(float)) {
        return SFUStatus::InvalidShape;
    }

    switch (desc.op) {
        case static_cast<uint32_t>(SFUPrimitiveOp::EXP):
        case static_cast<uint32_t>(SFUPrimitiveOp::LOG):
        case static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL):
        case static_cast<uint32_t>(SFUPrimitiveOp::RSQRT):
        case static_cast<uint32_t>(SFUPrimitiveOp::TANH):
        case static_cast<uint32_t>(SFUPrimitiveOp::SIGMOID):
            return SFUStatus::Success;
        default:
            return SFUStatus::InvalidDescriptor;
    }
}

SFUStatus SFU::validatePrimitiveBatchDescriptor(const SFUPrimitiveBatchDesc& desc) const
{
    if (desc.desc_array_gm_addr == 0 || desc.desc_count == 0 ||
        desc.desc_count > GOLEM_SFU_PRIMITIVE_BATCH_MAX_DESCS) {
        return SFUStatus::InvalidShape;
    }
    return SFUStatus::Success;
}

bool SFU::executePrimitiveDesc(uint64_t descAddr, uint64_t tag, uint64_t* processedElems)
{
    PrimitiveOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;

    if (!readPrimitiveDescriptor(descAddr, &state.desc)) {
        return false;
    }
    state.status = validatePrimitiveDescriptor(state.desc);
    if (state.status != SFUStatus::Success) {
        return false;
    }
    if (!readPrimitiveInput(state.desc, &state.input0)) {
        return false;
    }
    if (!executePrimitive(&state)) {
        return false;
    }
    if (!writePrimitiveOutput(state.desc, state.output)) {
        return false;
    }
    if (processedElems != nullptr) {
        *processedElems = primitiveProcessedElems(state.desc);
    }
    return true;
}

bool SFU::readPrimitiveInput(const SFUPrimitiveDesc& desc, std::vector<float>* values)
{
    if (globalMem_ == nullptr || values == nullptr) {
        return false;
    }

    values->assign(desc.elem_count, 0.0f);
    const uint32_t stride = effectiveStride(desc.input0_stride_bytes);
    if (stride == sizeof(float)) {
        std::vector<uint8_t> raw;
        const size_t byteCount = static_cast<size_t>(desc.elem_count) * sizeof(float);
        globalMem_->rd_from_globalmem(desc.input0_gm_addr, byteCount, raw);
        if (raw.size() != byteCount) {
            return false;
        }
        std::memcpy(values->data(), raw.data(), byteCount);
        return true;
    }

    for (uint32_t i = 0; i < desc.elem_count; ++i) {
        std::vector<uint8_t> raw;
        globalMem_->rd_from_globalmem(desc.input0_gm_addr + static_cast<uint64_t>(i) * stride,
                                      sizeof(float), raw);
        if (raw.size() != sizeof(float)) {
            return false;
        }
        float value = 0.0f;
        std::memcpy(&value, raw.data(), sizeof(float));
        (*values)[i] = value;
    }
    return true;
}

bool SFU::executePrimitive(PrimitiveOpState* state)
{
    if (state == nullptr) {
        return false;
    }

    state->output.assign(state->input0.size(), 0.0f);
    for (size_t i = 0; i < state->input0.size(); ++i) {
        const float value = state->input0[i];
        switch (state->desc.op) {
            case static_cast<uint32_t>(SFUPrimitiveOp::EXP):
                state->output[i] = static_cast<float>(std::exp(static_cast<double>(value)));
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::LOG):
                state->output[i] = static_cast<float>(std::log(static_cast<double>(value)));
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL):
                state->output[i] = 1.0f / value;
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::RSQRT):
                state->output[i] = 1.0f / static_cast<float>(std::sqrt(static_cast<double>(value)));
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::TANH):
                state->output[i] = static_cast<float>(std::tanh(static_cast<double>(value)));
                break;
            case static_cast<uint32_t>(SFUPrimitiveOp::SIGMOID):
                state->output[i] = 1.0f / (1.0f + static_cast<float>(std::exp(-static_cast<double>(value))));
                break;
            default:
                return false;
        }
    }
    return true;
}

bool SFU::writePrimitiveOutput(const SFUPrimitiveDesc& desc, const std::vector<float>& values)
{
    if (globalMem_ == nullptr || values.size() != desc.elem_count) {
        return false;
    }

    const uint32_t stride = effectiveStride(desc.output_stride_bytes);
    if (stride == sizeof(float)) {
        std::vector<uint8_t> raw(values.size() * sizeof(float));
        std::memcpy(raw.data(), values.data(), raw.size());
        globalMem_->wr_to_globalmem(desc.output_gm_addr, raw.size(), raw);
        return true;
    }

    for (uint32_t i = 0; i < desc.elem_count; ++i) {
        std::vector<uint8_t> raw;
        appendBytes(raw, values[i]);
        globalMem_->wr_to_globalmem(desc.output_gm_addr + static_cast<uint64_t>(i) * stride,
                                    raw.size(), raw);
    }
    return true;
}

uint64_t SFU::primitiveProcessedElems(const SFUPrimitiveDesc& desc) const
{
    if ((desc.flags & GOLEM_SFU_PRIMITIVE_FLAG_REPEAT_CHUNK) == 0) {
        return desc.elem_count;
    }
    return desc.input1_gm_addr > desc.elem_count ? desc.input1_gm_addr : desc.elem_count;
}

bool SFU::readSoftmaxDescriptor(uint64_t descAddr, SFUSoftmaxTileDesc* desc)
{
    if (globalMem_ == nullptr || desc == nullptr || descAddr == 0) {
        return false;
    }

    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(descAddr, sizeof(SFUSoftmaxTileDesc), raw);
    if (raw.size() != sizeof(SFUSoftmaxTileDesc)) {
        return false;
    }

    std::memcpy(desc, raw.data(), sizeof(SFUSoftmaxTileDesc));
    return true;
}

SFUStatus SFU::validateSoftmaxDescriptor(const SFUSoftmaxTileDesc& desc) const
{
    if (desc.elem_bytes != sizeof(float)) {
        return SFUStatus::UnsupportedElemBytes;
    }
    if (desc.local_input_gm_addr == 0 || desc.local_output_gm_addr == 0 ||
        desc.global_m == 0 || desc.global_n == 0 ||
        desc.block_m == 0 || desc.block_n == 0 ||
        desc.valid_m == 0 || desc.valid_n == 0 ||
        desc.n_tiles_per_row == 0 ||
        desc.valid_m > desc.block_m || desc.valid_n > desc.block_n) {
        return SFUStatus::InvalidShape;
    }

    const uint64_t firstRow = static_cast<uint64_t>(desc.m_tile) * desc.block_m;
    const uint64_t firstCol = static_cast<uint64_t>(desc.n_tile) * desc.block_n;
    if (firstRow + desc.valid_m > desc.global_m || firstCol + desc.valid_n > desc.global_n) {
        return SFUStatus::InvalidShape;
    }
    return SFUStatus::Success;
}

bool SFU::readInputTile(const SFUSoftmaxTileDesc& desc, std::vector<float>* values)
{
    if (globalMem_ == nullptr || values == nullptr) {
        return false;
    }

    const size_t elemCount = static_cast<size_t>(desc.block_m) * desc.block_n;
    const size_t byteCount = elemCount * desc.elem_bytes;
    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(desc.local_input_gm_addr, byteCount, raw);
    if (raw.size() != byteCount) {
        return false;
    }

    values->assign(elemCount, 0.0f);
    for (size_t i = 0; i < elemCount; ++i) {
        float value = 0.0f;
        std::memcpy(&value, raw.data() + i * sizeof(float), sizeof(float));
        (*values)[i] = value;
    }
    return true;
}

void SFU::computeTileStats(SoftmaxOpState* state)
{
    if (state == nullptr) {
        return;
    }

    const SFUSoftmaxTileDesc& desc = state->desc;
    state->rowStats.clear();
    state->rowStats.reserve(desc.valid_m);
    for (uint32_t row = 0; row < desc.valid_m; ++row) {
        double tile_m = -std::numeric_limits<double>::infinity();
        for (uint32_t col = 0; col < desc.valid_n; ++col) {
            const size_t idx = tilePackedIndex(desc, row, col);
            tile_m = std::max(tile_m, static_cast<double>(state->inputTile[idx]));
        }

        double tile_l = 0.0;
        for (uint32_t col = 0; col < desc.valid_n; ++col) {
            const size_t idx = tilePackedIndex(desc, row, col);
            tile_l += std::exp(static_cast<double>(state->inputTile[idx]) - tile_m);
        }

        state->rowStats.push_back(SFUTileRowStats{
            static_cast<uint32_t>(desc.m_tile * desc.block_m + row),
            tile_m,
            tile_l,
        });
        statSoftmaxRows_->addData(1);
    }
}

bool SFU::mergeTileStats(SoftmaxOpState* state)
{
    if (state == nullptr) {
        return false;
    }

    bool allReady = true;
    auto& rows = softmaxReducerRows();
    for (const auto& row : state->rowStats) {
        const SoftmaxReducerKey key(state->desc.job_id, row.global_row);
        auto& rowState = rows[key];
        if (rowState.n_tiles_expected == 0) {
            rowState.n_tiles_expected = state->desc.n_tiles_per_row;
        }

        const double m_new = std::max(rowState.m_acc, row.tile_m);
        const double l_new = rowState.l_acc * std::exp(rowState.m_acc - m_new) +
                             row.tile_l * std::exp(row.tile_m - m_new);
        rowState.m_acc = m_new;
        rowState.l_acc = l_new;
        rowState.partials_seen += 1;
        rowState.ready = rowState.partials_seen >= rowState.n_tiles_expected;
        allReady = allReady && rowState.ready;
    }
    return allReady;
}

bool SFU::tileGlobalStatsReady(const SoftmaxOpState& state) const
{
    const auto& rows = softmaxReducerRows();
    for (const auto& row : state.rowStats) {
        const SoftmaxReducerKey key(state.desc.job_id, row.global_row);
        auto it = rows.find(key);
        if (it == rows.end() || !it->second.ready) {
            return false;
        }
    }
    return true;
}

bool SFU::normalizeTile(SoftmaxOpState* state)
{
    if (state == nullptr || globalMem_ == nullptr) {
        return false;
    }

    const SFUSoftmaxTileDesc& desc = state->desc;
    const size_t elemCount = static_cast<size_t>(desc.block_m) * desc.block_n;
    std::vector<float> outputTile(elemCount, 0.0f);

    auto& rows = softmaxReducerRows();
    for (uint32_t row = 0; row < desc.block_m; ++row) {
        const uint32_t globalRow = static_cast<uint32_t>(desc.m_tile * desc.block_m + row);
        const SoftmaxReducerKey key(desc.job_id, globalRow);
        auto rowIt = rows.find(key);
        if (row < desc.valid_m && (rowIt == rows.end() || !rowIt->second.ready || rowIt->second.l_acc == 0.0)) {
            return false;
        }

        for (uint32_t col = 0; col < desc.block_n; ++col) {
            if (row < desc.valid_m && col < desc.valid_n) {
                const auto& rowState = rowIt->second;
                const size_t idx = tilePackedIndex(desc, row, col);
                const double normalized =
                    std::exp(static_cast<double>(state->inputTile[idx]) - rowState.m_acc) / rowState.l_acc;
                outputTile[idx] = static_cast<float>(normalized);
            }
        }
    }

    std::vector<uint8_t> out;
    out.reserve(elemCount * sizeof(float));
    for (const float value : outputTile) {
        appendBytes(out, value);
    }
    globalMem_->wr_to_globalmem(desc.local_output_gm_addr, out.size(), out);

    for (const auto& row : state->rowStats) {
        const SoftmaxReducerKey key(desc.job_id, row.global_row);
        auto it = rows.find(key);
        if (it != rows.end()) {
            it->second.normalizes_done += 1;
            if (it->second.normalizes_done >= it->second.n_tiles_expected) {
                rows.erase(it);
            }
        }
    }
    return true;
}

void SFU::bindGlobalMemory(GlobalMemoryAPI* globalMem)
{
    globalMem_ = globalMem;
}

void SFU::setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores)
{
    coreId_ = coreId;
    activeWorkerCores_ = activeWorkerCores == 0 ? 1 : activeWorkerCores;
}

} // namespace Golem
} // namespace SST
