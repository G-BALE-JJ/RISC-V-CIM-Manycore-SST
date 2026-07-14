#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>

#include <sst/elements/golem/sfu/sfu.h>

namespace SST {
namespace Golem {

namespace {

using SoftmaxReducerKey = std::pair<uint64_t, uint32_t>;
using DistributedSoftmaxReducerKey =
    std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>;
using DistributedReductionResponseFanoutKey =
    std::tuple<uint64_t,
               uint64_t,
               uint32_t,
               uint32_t,
               ReductionTransportMessageKind>;
constexpr uint32_t GOLEM_DTYPE_FP32_VALUE = 1;
constexpr uint32_t GOLEM_SFU_PRIMITIVE_FLAG_REPEAT_CHUNK = 0x1;
constexpr uint32_t GOLEM_SFU_PRIMITIVE_BATCH_MAX_DESCS = 64;
constexpr uint32_t GOLEM_SFU_JOB_SOFTMAX_ROW_BAND_ROWS = 4;

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

struct DistributedSoftmaxReducerRowState {
    uint32_t expectedWorkers = 0;
    uint32_t expectedRows = 0;
    uint32_t expectedCols = 0;
    std::vector<uint8_t> maxSeen;
    std::vector<uint8_t> sumSeen;
    std::vector<uint8_t> normalizeSeen;
    std::vector<uint8_t> abortSeen;
    double globalMax = -std::numeric_limits<double>::infinity();
    double globalSum = 0.0;
    bool aborted = false;
};

enum class DistributedReducerResult : uint8_t {
    Accepted,
    Pending,
    Ready,
    Aborted,
    Invalid,
};

std::map<DistributedSoftmaxReducerKey, DistributedSoftmaxReducerRowState>&
distributedSoftmaxReducerRows()
{
    static std::map<DistributedSoftmaxReducerKey, DistributedSoftmaxReducerRowState> rows;
    return rows;
}

std::set<DistributedSoftmaxReducerKey>& distributedSoftmaxAbortedRows()
{
    static std::set<DistributedSoftmaxReducerKey> rows;
    return rows;
}

DistributedSoftmaxReducerKey distributedSoftmaxReducerKey(uint64_t jobId,
                                                           uint64_t tag,
                                                           uint32_t ownerCore,
                                                           uint32_t row)
{
    return DistributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
}

void markDistributedSoftmaxRowAborted(const DistributedSoftmaxReducerKey& key)
{
    distributedSoftmaxAbortedRows().insert(key);
}

bool distributedSoftmaxRowAborted(const DistributedSoftmaxReducerKey& key)
{
    return distributedSoftmaxAbortedRows().find(key) != distributedSoftmaxAbortedRows().end();
}

std::set<DistributedReductionResponseFanoutKey>& distributedReductionResponseFanoutRows()
{
    static std::set<DistributedReductionResponseFanoutKey> rows;
    return rows;
}

void clearDistributedReductionResponseFanout(const DistributedSoftmaxReducerKey& key)
{
    auto& rows = distributedReductionResponseFanoutRows();
    const uint64_t jobId = std::get<0>(key);
    const uint64_t tag = std::get<1>(key);
    const uint32_t ownerCore = std::get<2>(key);
    const uint32_t row = std::get<3>(key);
    rows.erase(DistributedReductionResponseFanoutKey(
        jobId, tag, ownerCore, row, ReductionTransportMessageKind::MaxResponse));
    rows.erase(DistributedReductionResponseFanoutKey(
        jobId, tag, ownerCore, row, ReductionTransportMessageKind::SumResponse));
}

bool distributedSoftmaxJobMatchesKey(const DistributedSoftmaxReducerKey& key,
                                     uint64_t jobId,
                                     uint64_t tag,
                                     uint32_t ownerCore)
{
    return std::get<0>(key) == jobId && std::get<1>(key) == tag &&
           std::get<2>(key) == ownerCore;
}

std::vector<DistributedSoftmaxReducerKey> collectDistributedSoftmaxJobKeys(
    uint64_t jobId,
    uint64_t tag,
    uint32_t ownerCore)
{
    std::vector<DistributedSoftmaxReducerKey> keys;
    for (const auto& entry : distributedSoftmaxReducerRows()) {
        if (distributedSoftmaxJobMatchesKey(entry.first, jobId, tag, ownerCore)) {
            keys.push_back(entry.first);
        }
    }
    return keys;
}

bool initializeDistributedSoftmaxRow(DistributedSoftmaxReducerRowState* row,
                                     uint32_t expectedWorkers,
                                     uint32_t expectedRows,
                                     uint32_t expectedCols)
{
    if (row == nullptr || expectedWorkers == 0 || expectedRows == 0 || expectedCols == 0) {
        return false;
    }
    if (row->expectedWorkers == 0) {
        row->expectedWorkers = expectedWorkers;
        row->expectedRows = expectedRows;
        row->expectedCols = expectedCols;
        row->maxSeen.assign(expectedWorkers, 0);
        row->sumSeen.assign(expectedWorkers, 0);
        row->normalizeSeen.assign(expectedWorkers, 0);
        row->abortSeen.assign(expectedWorkers, 0);
    }
    return row->expectedWorkers == expectedWorkers &&
           row->expectedRows == expectedRows && row->expectedCols == expectedCols;
}

bool allWorkerSlotsSeen(const std::vector<uint8_t>& seen)
{
    return !seen.empty() &&
           std::all_of(seen.begin(), seen.end(), [](uint8_t value) { return value != 0; });
}

DistributedReducerResult observeDistributedSoftmaxAbort(
    const DistributedSoftmaxReducerKey& key,
    DistributedSoftmaxReducerRowState* row,
    uint32_t workerSlot)
{
    if (row == nullptr || !row->aborted || workerSlot >= row->expectedWorkers) {
        return DistributedReducerResult::Invalid;
    }
    row->abortSeen[workerSlot] = 1;
    if (allWorkerSlotsSeen(row->abortSeen)) {
        clearDistributedReductionResponseFanout(key);
        distributedSoftmaxReducerRows().erase(key);
    }
    return DistributedReducerResult::Aborted;
}

void abortDistributedSoftmaxRow(uint64_t jobId,
                                uint64_t tag,
                                uint32_t ownerCore,
                                uint32_t row,
                                uint32_t workerSlot,
                                uint32_t expectedWorkers,
                                uint32_t expectedRows,
                                uint32_t expectedCols)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    markDistributedSoftmaxRowAborted(key);
    auto& rowState = rows[key];
    clearDistributedReductionResponseFanout(key);
    if (!initializeDistributedSoftmaxRow(
            &rowState, expectedWorkers, expectedRows, expectedCols)) {
        rowState.aborted = true;
        if (workerSlot < rowState.expectedWorkers) {
            (void)observeDistributedSoftmaxAbort(key, &rowState, workerSlot);
        }
        return;
    }
    rowState.aborted = true;
    (void)observeDistributedSoftmaxAbort(key, &rowState, workerSlot);
}

DistributedReducerResult submitDistributedSoftmaxMax(uint64_t jobId,
                                                      uint64_t tag,
                                                      uint32_t ownerCore,
                                                      uint32_t row,
                                                      uint32_t workerSlot,
                                                      uint32_t expectedWorkers,
                                                      uint32_t expectedRows,
                                                      uint32_t expectedCols,
                                                      double localMax)
{
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    if (distributedSoftmaxRowAborted(key)) {
        return DistributedReducerResult::Aborted;
    }
    auto& rows = distributedSoftmaxReducerRows();
    auto& rowState = rows[key];
    if (!initializeDistributedSoftmaxRow(
            &rowState, expectedWorkers, expectedRows, expectedCols) ||
        workerSlot >= expectedWorkers) {
        return DistributedReducerResult::Invalid;
    }
    if (rowState.aborted) {
        return observeDistributedSoftmaxAbort(key, &rowState, workerSlot);
    }
    if (rowState.maxSeen[workerSlot] != 0) {
        return DistributedReducerResult::Invalid;
    }
    rowState.maxSeen[workerSlot] = 1;
    rowState.globalMax = std::max(rowState.globalMax, localMax);
    return DistributedReducerResult::Accepted;
}

DistributedReducerResult distributedSoftmaxMaxReady(uint64_t jobId,
                                                     uint64_t tag,
                                                     uint32_t ownerCore,
                                                     uint32_t row,
                                                     uint32_t workerSlot,
                                                     double* globalMax)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    if (distributedSoftmaxRowAborted(key)) {
        return DistributedReducerResult::Aborted;
    }
    auto it = rows.find(key);
    if (it == rows.end()) {
        return DistributedReducerResult::Pending;
    }
    if (it->second.aborted) {
        return observeDistributedSoftmaxAbort(key, &it->second, workerSlot);
    }
    if (!allWorkerSlotsSeen(it->second.maxSeen)) {
        return DistributedReducerResult::Pending;
    }
    if (globalMax != nullptr) {
        *globalMax = it->second.globalMax;
    }
    return DistributedReducerResult::Ready;
}

DistributedReducerResult submitDistributedSoftmaxSum(uint64_t jobId,
                                                      uint64_t tag,
                                                      uint32_t ownerCore,
                                                      uint32_t row,
                                                      uint32_t workerSlot,
                                                      uint32_t expectedWorkers,
                                                      uint32_t expectedRows,
                                                      uint32_t expectedCols,
                                                      double localSum)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    if (distributedSoftmaxRowAborted(key)) {
        return DistributedReducerResult::Aborted;
    }
    auto it = rows.find(key);
    if (it == rows.end() ||
        !initializeDistributedSoftmaxRow(
            &it->second, expectedWorkers, expectedRows, expectedCols) ||
        workerSlot >= expectedWorkers) {
        return DistributedReducerResult::Invalid;
    }
    if (it->second.aborted) {
        return observeDistributedSoftmaxAbort(key, &it->second, workerSlot);
    }
    if (it->second.sumSeen[workerSlot] != 0 ||
        !allWorkerSlotsSeen(it->second.maxSeen)) {
        return DistributedReducerResult::Invalid;
    }
    it->second.sumSeen[workerSlot] = 1;
    it->second.globalSum += localSum;
    return DistributedReducerResult::Accepted;
}

DistributedReducerResult distributedSoftmaxSumReady(uint64_t jobId,
                                                     uint64_t tag,
                                                     uint32_t ownerCore,
                                                     uint32_t row,
                                                     uint32_t workerSlot,
                                                     double* globalSum)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    if (distributedSoftmaxRowAborted(key)) {
        return DistributedReducerResult::Aborted;
    }
    auto it = rows.find(key);
    if (it == rows.end()) {
        return DistributedReducerResult::Pending;
    }
    if (it->second.aborted) {
        return observeDistributedSoftmaxAbort(key, &it->second, workerSlot);
    }
    if (!allWorkerSlotsSeen(it->second.sumSeen)) {
        return DistributedReducerResult::Pending;
    }
    if (globalSum != nullptr) {
        *globalSum = it->second.globalSum;
    }
    return DistributedReducerResult::Ready;
}

DistributedReducerResult markDistributedSoftmaxNormalized(uint64_t jobId,
                                                           uint64_t tag,
                                                           uint32_t ownerCore,
                                                           uint32_t row,
                                                           uint32_t workerSlot,
                                                           uint32_t expectedWorkers,
                                                           uint32_t expectedRows,
                                                           uint32_t expectedCols)
{
    auto& rows = distributedSoftmaxReducerRows();
    const auto key = distributedSoftmaxReducerKey(jobId, tag, ownerCore, row);
    auto it = rows.find(key);
    if (it == rows.end() || it->second.expectedWorkers != expectedWorkers ||
        it->second.expectedRows != expectedRows || it->second.expectedCols != expectedCols ||
        workerSlot >= expectedWorkers) {
        return DistributedReducerResult::Invalid;
    }
    if (it->second.aborted) {
        return observeDistributedSoftmaxAbort(key, &it->second, workerSlot);
    }
    if (it->second.normalizeSeen[workerSlot] != 0 ||
        !allWorkerSlotsSeen(it->second.sumSeen)) {
        return DistributedReducerResult::Invalid;
    }
    it->second.normalizeSeen[workerSlot] = 1;
    if (allWorkerSlotsSeen(it->second.normalizeSeen)) {
        clearDistributedReductionResponseFanout(key);
        distributedSoftmaxReducerRows().erase(key);
    }
    return DistributedReducerResult::Accepted;
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

uint32_t workerColumnBegin(uint32_t cols, uint32_t worker, uint32_t workerCores)
{
    return static_cast<uint32_t>(static_cast<uint64_t>(cols) * worker / workerCores);
}

uint32_t workerColumnEnd(uint32_t cols, uint32_t worker, uint32_t workerCores)
{
    return static_cast<uint32_t>(static_cast<uint64_t>(cols) * (worker + 1) / workerCores);
}

struct SoftmaxJobRowBandState {
    uint32_t rowBegin = 0;
    uint32_t rowEnd = 0;
    uint32_t bandRows = 0;
    std::vector<double> localMax;
    std::vector<double> localSum;
    std::vector<double> globalMax;
    std::vector<double> globalSum;
};

bool readSoftmaxJobChunk(GlobalMemoryAPI* globalMem,
                         const SFUJobDesc& desc,
                         uint32_t row,
                         uint32_t colBegin,
                         uint32_t colEnd,
                         std::vector<float>* chunk)
{
    if (globalMem == nullptr || chunk == nullptr || colEnd < colBegin || colEnd > desc.cols) {
        return false;
    }

    const uint32_t elemCount = colEnd - colBegin;
    const uint64_t offsetElems = static_cast<uint64_t>(row) * desc.cols + colBegin;
    std::vector<uint8_t> raw;
    globalMem->rd_from_globalmem(desc.input0_addr + offsetElems * sizeof(float),
                                 static_cast<uint64_t>(elemCount) * sizeof(float),
                                 raw);
    if (raw.size() != static_cast<size_t>(elemCount) * sizeof(float)) {
        return false;
    }

    chunk->assign(elemCount, 0.0f);
    if (elemCount > 0) {
        std::memcpy(chunk->data(), raw.data(), raw.size());
    }
    return true;
}

bool readSoftmaxJobOutputChunk(GlobalMemoryAPI* globalMem,
                               const SFUJobDesc& desc,
                               uint32_t row,
                               uint32_t colBegin,
                               uint32_t colEnd,
                               std::vector<float>* chunk)
{
    if (globalMem == nullptr || chunk == nullptr || colEnd < colBegin || colEnd > desc.cols) {
        return false;
    }

    const uint32_t elemCount = colEnd - colBegin;
    const uint64_t offsetElems = static_cast<uint64_t>(row) * desc.cols + colBegin;
    std::vector<uint8_t> raw;
    globalMem->rd_from_globalmem(desc.output_addr + offsetElems * sizeof(float),
                                 static_cast<uint64_t>(elemCount) * sizeof(float),
                                 raw);
    if (raw.size() != static_cast<size_t>(elemCount) * sizeof(float)) {
        return false;
    }

    chunk->assign(elemCount, 0.0f);
    if (elemCount > 0) {
        std::memcpy(chunk->data(), raw.data(), raw.size());
    }
    return true;
}

bool writeSoftmaxJobChunk(GlobalMemoryAPI* globalMem,
                          const SFUJobDesc& desc,
                          uint32_t row,
                          uint32_t colBegin,
                          const std::vector<float>& chunk)
{
    if (globalMem == nullptr || colBegin > desc.cols || chunk.empty()) {
        return false;
    }
    if (static_cast<uint64_t>(colBegin) + chunk.size() > desc.cols) {
        return false;
    }

    std::vector<uint8_t> raw(chunk.size() * sizeof(float));
    std::memcpy(raw.data(), chunk.data(), raw.size());
    const uint64_t offsetElems = static_cast<uint64_t>(row) * desc.cols + colBegin;
    globalMem->wr_to_globalmem(desc.output_addr + offsetElems * sizeof(float),
                               raw.size(),
                               raw);
    return true;
}

bool readDistributedSoftmaxJobChunk(GlobalMemoryAPI* globalMem,
                                    uint64_t baseAddr,
                                    uint32_t rows,
                                    uint32_t localCols,
                                    uint32_t row,
                                    uint32_t localColBegin,
                                    uint32_t localColEnd,
                                    std::vector<float>* chunk)
{
    if (globalMem == nullptr || baseAddr == 0 || chunk == nullptr || row >= rows ||
        localColEnd <= localColBegin || localColEnd > localCols) {
        return false;
    }
    const uint32_t elemCount = localColEnd - localColBegin;
    const uint64_t offsetElems = static_cast<uint64_t>(row) * localCols + localColBegin;
    std::vector<uint8_t> raw;
    globalMem->rd_from_globalmem(baseAddr + offsetElems * sizeof(float),
                                 static_cast<uint64_t>(elemCount) * sizeof(float),
                                 raw);
    if (raw.size() != static_cast<size_t>(elemCount) * sizeof(float)) {
        return false;
    }
    chunk->resize(elemCount);
    std::memcpy(chunk->data(), raw.data(), raw.size());
    return true;
}

bool writeDistributedSoftmaxJobChunk(GlobalMemoryAPI* globalMem,
                                     uint64_t baseAddr,
                                     uint32_t rows,
                                     uint32_t localCols,
                                     uint32_t row,
                                     uint32_t localColBegin,
                                     const std::vector<float>& chunk)
{
    if (globalMem == nullptr || baseAddr == 0 || row >= rows || chunk.empty() ||
        static_cast<uint64_t>(localColBegin) + chunk.size() > localCols) {
        return false;
    }
    std::vector<uint8_t> raw(chunk.size() * sizeof(float));
    std::memcpy(raw.data(), chunk.data(), raw.size());
    const uint64_t offsetElems = static_cast<uint64_t>(row) * localCols + localColBegin;
    globalMem->wr_to_globalmem(baseAddr + offsetElems * sizeof(float), raw.size(), raw);
    return true;
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
      distributedReductionTransport_(DistributedReductionTransport::Shared),
      globalMem_(nullptr),
      output_("Golem::SFU[@p:@l]: ", verbose_, 0, SST::Output::STDOUT)
{
    if (activeWorkerCores_ == 0) {
        activeWorkerCores_ = 1;
    }
    if (maxInflight_ == 0) {
        maxInflight_ = 1;
    }
    const std::string transport =
        params.find<std::string>("distributed_reduction_transport", "shared");
    if (transport == "shared") {
        distributedReductionTransport_ = DistributedReductionTransport::Shared;
    } else if (transport == "modeled_noc" || transport == "noc_model" ||
        transport == "noc") {
        distributedReductionTransport_ = DistributedReductionTransport::ModeledNoC;
    } else if (transport == "explicit_noc") {
        distributedReductionTransport_ = DistributedReductionTransport::ExplicitNoC;
    } else {
        output_.fatal(CALL_INFO,
                      -1,
                      "invalid distributed_reduction_transport='%s' "
                      "(expected shared, modeled_noc, or explicit_noc)\n",
                      transport.c_str());
    }

    statOpsIssued_ = registerStatistic<uint64_t>("sfu_ops_issued");
    statSoftmaxRows_ = registerStatistic<uint64_t>("sfu_softmax_rows");
    statSoftmaxTiles_ = registerStatistic<uint64_t>("sfu_softmax_tiles");
    statJobSoftmaxMaxChunks_ = registerStatistic<uint64_t>("sfu_job_softmax_max_chunks");
    statJobSoftmaxSumChunks_ = registerStatistic<uint64_t>("sfu_job_softmax_sum_chunks");
    statJobSoftmaxNormChunks_ = registerStatistic<uint64_t>("sfu_job_softmax_norm_chunks");
    statPrimitiveElems_ = registerStatistic<uint64_t>("sfu_primitive_elems");
    statPartialSubmits_ = registerStatistic<uint64_t>("sfu_partial_submits");
    statPartialDone_ = registerStatistic<uint64_t>("sfu_partial_done");
    statReductionMaxRequests_ = registerStatistic<uint64_t>("sfu_reduction_max_requests");
    statReductionMaxResponses_ = registerStatistic<uint64_t>("sfu_reduction_max_responses");
    statReductionSumRequests_ = registerStatistic<uint64_t>("sfu_reduction_sum_requests");
    statReductionSumResponses_ = registerStatistic<uint64_t>("sfu_reduction_sum_responses");
    statReductionTransportReceived_ = registerStatistic<uint64_t>("sfu_reduction_transport_received");
    statReductionTransportStaleDropped_ = registerStatistic<uint64_t>("sfu_reduction_transport_stale_dropped");
    statReductionTransportInboxHighWater_ = registerStatistic<uint64_t>("sfu_reduction_transport_inbox_high_water");
    statReductionTransportLatencyCycles_ = registerStatistic<uint64_t>("sfu_reduction_transport_latency_cycles");
    statCreditStalls_ = registerStatistic<uint64_t>("sfu_credit_stalls");
    statCrossTileWaitCycles_ = registerStatistic<uint64_t>("sfu_cross_tile_wait_cycles");
    statRetryEvents_ = registerStatistic<uint64_t>("sfu_retry_events");
}

bool SFU::modeledDistributedReductionEnabled() const
{
    return distributedReductionTransport_ == DistributedReductionTransport::ModeledNoC;
}

bool SFU::explicitDistributedReductionEnabled() const
{
    return distributedReductionTransport_ == DistributedReductionTransport::ExplicitNoC;
}

void SFU::handleReductionTransportMessage(const ReductionTransportMessage& message)
{
    if (!explicitDistributedReductionEnabled() || globalMem_ == nullptr) {
        return;
    }

    statReductionTransportReceived_->addData(1);
    const uint64_t receiveCycle = getCurrentSimCycle();
    if (receiveCycle >= message.sendCycle) {
        statReductionTransportLatencyCycles_->addData(receiveCycle - message.sendCycle);
    }
    const auto staleDrop = [this]() {
        statReductionTransportStaleDropped_->addData(1);
        statRetryEvents_->addData(1);
    };

    const bool isMaxRequest = message.kind == ReductionTransportMessageKind::MaxRequest;
    const bool isSumRequest = message.kind == ReductionTransportMessageKind::SumRequest;
    const bool isMaxResponse = message.kind == ReductionTransportMessageKind::MaxResponse;
    const bool isSumResponse = message.kind == ReductionTransportMessageKind::SumResponse;

    if (isMaxResponse || isSumResponse) {
        auto job = pendingJobOps_.find(message.tag);
        if (job == pendingJobOps_.end()) {
            staleDrop();
            return;
        }
        const SoftmaxJobStage expectedStage = isMaxResponse
            ? SoftmaxJobStage::MaxSubmitted
            : SoftmaxJobStage::SumSubmitted;
        const std::vector<uint8_t>& responseSeen = isMaxResponse
            ? job->second.maxResponseSeen
            : job->second.sumResponseSeen;
        if ((job->second.desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) == 0 ||
            job->second.distributedAbortObserved ||
            job->second.desc.job_id != message.jobId ||
            job->second.tag != message.tag ||
            job->second.desc.owner_core != message.ownerCore ||
            job->second.workerSlot != message.workerSlot ||
            job->second.desc.worker_cores != message.expectedWorkers ||
            job->second.desc.rows != message.expectedRows ||
            job->second.desc.cols != message.expectedCols ||
            message.row >= message.expectedRows ||
            job->second.stage != expectedStage ||
            message.row >= responseSeen.size() ||
            responseSeen[message.row] != 0 ||
            message.ownerCore > std::numeric_limits<uint32_t>::max() - message.workerSlot ||
            coreId_ != message.ownerCore + message.workerSlot) {
            staleDrop();
            return;
        }

        const DistributedReductionResponseInboxKey key(message.jobId,
                                                        message.tag,
                                                        message.ownerCore,
                                                        message.row,
                                                        message.workerSlot,
                                                        message.kind);
        if (!distributedReductionResponseInbox_.emplace(key, message).second) {
            staleDrop();
            return;
        }
        const uint64_t inboxSize = distributedReductionResponseInbox_.size();
        if (inboxSize > distributedReductionResponseInboxHighWater_) {
            statReductionTransportInboxHighWater_->addData(
                inboxSize - distributedReductionResponseInboxHighWater_);
            distributedReductionResponseInboxHighWater_ = inboxSize;
        }
        recordDistributedReductionResponse(isMaxResponse);
        return;
    }

    if ((!isMaxRequest && !isSumRequest) || coreId_ != message.ownerCore ||
        message.expectedWorkers == 0 || message.expectedRows == 0 ||
        message.expectedCols == 0 || message.row >= message.expectedRows ||
        message.workerSlot >= message.expectedWorkers) {
        staleDrop();
        return;
    }

    DistributedReducerResult result = DistributedReducerResult::Invalid;
    if (isMaxRequest) {
        result = submitDistributedSoftmaxMax(message.jobId,
                                             message.tag,
                                             message.ownerCore,
                                             message.row,
                                             message.workerSlot,
                                             message.expectedWorkers,
                                             message.expectedRows,
                                             message.expectedCols,
                                             message.value);
    } else {
        result = submitDistributedSoftmaxSum(message.jobId,
                                             message.tag,
                                             message.ownerCore,
                                             message.row,
                                             message.workerSlot,
                                             message.expectedWorkers,
                                             message.expectedRows,
                                             message.expectedCols,
                                             message.value);
    }
    if (result != DistributedReducerResult::Accepted) {
        staleDrop();
        return;
    }

    double reducedValue = 0.0;
    const DistributedReducerResult ready = isMaxRequest
        ? distributedSoftmaxMaxReady(message.jobId,
                                     message.tag,
                                     message.ownerCore,
                                     message.row,
                                     message.workerSlot,
                                     &reducedValue)
        : distributedSoftmaxSumReady(message.jobId,
                                     message.tag,
                                     message.ownerCore,
                                     message.row,
                                     message.workerSlot,
                                     &reducedValue);
    if (ready == DistributedReducerResult::Pending) {
        return;
    }
    if (ready != DistributedReducerResult::Ready) {
        staleDrop();
        return;
    }

    const ReductionTransportMessageKind responseKind = isMaxRequest
        ? ReductionTransportMessageKind::MaxResponse
        : ReductionTransportMessageKind::SumResponse;
    const DistributedReductionResponseFanoutKey fanoutKey(message.jobId,
                                                           message.tag,
                                                           message.ownerCore,
                                                           message.row,
                                                           responseKind);
    if (!distributedReductionResponseFanoutRows().insert(fanoutKey).second) {
        return;
    }

    for (uint32_t workerSlot = 0; workerSlot < message.expectedWorkers; ++workerSlot) {
        ReductionTransportMessage response = message;
        response.kind = responseKind;
        response.workerSlot = workerSlot;
        response.value = reducedValue;
        if (!globalMem_->sendReductionMessage(message.ownerCore + workerSlot, response)) {
            statRetryEvents_->addData(1);
            auto ownerJob = pendingJobOps_.find(message.tag);
            if (ownerJob != pendingJobOps_.end() &&
                ownerJob->second.desc.job_id == message.jobId &&
                ownerJob->second.desc.owner_core == message.ownerCore) {
                abortDistributedSoftmaxJob(&ownerJob->second);
            } else {
                abortDistributedSoftmaxRow(message.jobId,
                                           message.tag,
                                           message.ownerCore,
                                           message.row,
                                           workerSlot,
                                           message.expectedWorkers,
                                           message.expectedRows,
                                           message.expectedCols);
            }
            return;
        }
    }
}

void SFU::recordDistributedReductionRequest(bool maxStage)
{
    if (!modeledDistributedReductionEnabled() && !explicitDistributedReductionEnabled()) {
        return;
    }
    (maxStage ? statReductionMaxRequests_ : statReductionSumRequests_)->addData(1);
}

void SFU::recordDistributedReductionResponse(bool maxStage)
{
    if (!modeledDistributedReductionEnabled() && !explicitDistributedReductionEnabled()) {
        return;
    }
    (maxStage ? statReductionMaxResponses_ : statReductionSumResponses_)->addData(1);
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

bool SFU::issueJob(uint64_t descAddr, uint64_t tag)
{
    auto existing = pendingJobOps_.find(tag);
    if (existing != pendingJobOps_.end()) {
        // Duplicate tags poison the shared tag identity; preserving either job would be ambiguous.
        abortDistributedSoftmaxJob(&existing->second);
        existing->second.status = SFUStatus::InvalidDescriptor;
        existing->second.stage = SoftmaxJobStage::Complete;
        statRetryEvents_->addData(1);
        return true;
    }
    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    JobOpState state = {};
    state.descAddr = descAddr;
    state.tag = tag;
    state.status = SFUStatus::Success;
    state.processedElems = 0;

    if (!readJobDescriptor(descAddr, &state.desc)) {
        state.status = (globalMem_ == nullptr) ? SFUStatus::GlobalMemoryUnavailable
                                               : SFUStatus::InvalidDescriptor;
    } else {
        state.status = validateJobDescriptor(state.desc);
    }

    if (state.status == SFUStatus::Success &&
        (state.desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0) {
        for (uint32_t row = 0; row < state.desc.rows; ++row) {
            if (distributedSoftmaxRowAborted(distributedSoftmaxReducerKey(
                    state.desc.job_id, state.tag, state.desc.owner_core, row))) {
                // Abort tombstones reserve this transport identity until a generation exists.
                state.status = SFUStatus::InvalidDescriptor;
                statRetryEvents_->addData(1);
                break;
            }
        }
    }

    if (state.status == SFUStatus::Success && !executeJob(&state)) {
        abortDistributedSoftmaxJob(&state);
        state.status = SFUStatus::InvalidDescriptor;
    }
    if (state.status != SFUStatus::Success && state.status != SFUStatus::Pending && verbose_ > 0) {
        output_.verbose(
            CALL_INFO,
            1,
            0,
            "SFU job rejected core=%" PRIu32 " status=%" PRIu64
            " input=0x%" PRIx64 " output=0x%" PRIx64
            " op=%" PRIu32 " dtype=%" PRIu32 " rows=%" PRIu32 " cols=%" PRIu32
            " chunk=%" PRIu32 " workers=%" PRIu32 " slot=%" PRIu32
            " active_workers=%" PRIu32 " flags=0x%" PRIx32 "\n",
            coreId_,
            static_cast<uint64_t>(state.status),
            state.desc.input0_addr,
            state.desc.output_addr,
            state.desc.op_type,
            state.desc.dtype,
            state.desc.rows,
            state.desc.cols,
            state.desc.chunk_elems,
            state.desc.worker_cores,
            state.desc.reserved0,
            activeWorkerCores_,
            state.desc.flags);
    }

    pendingJobOps_[tag] = state;

    ++inflight_;
    statOpsIssued_->addData(1);
    if (state.status == SFUStatus::Success || state.status == SFUStatus::Pending) {
        statPrimitiveElems_->addData(state.processedElems);
    }
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
            } else {
                auto jobIt = pendingJobOps_.find(tag);
                if (jobIt != pendingJobOps_.end()) {
                    JobOpState& state = jobIt->second;
                    if (state.status == SFUStatus::Pending &&
                        (state.desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0 &&
                        !advanceDistributedSoftmaxJob(&state)) {
                        abortDistributedSoftmaxJob(&state);
                        state.status = SFUStatus::InvalidDescriptor;
                    }
                    opStatus = state.status;
                    if (opStatus == SFUStatus::Pending) {
                        statCrossTileWaitCycles_->addData(1);
                        if (status != nullptr) {
                            *status = static_cast<uint64_t>(SFUStatus::Pending);
                        }
                        return false;
                    }
                    clearDistributedReductionResponseInbox(state);
                    pendingJobOps_.erase(jobIt);
                }
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

bool SFU::readJobDescriptor(uint64_t descAddr, SFUJobDesc* desc)
{
    if (globalMem_ == nullptr || desc == nullptr || descAddr == 0) {
        return false;
    }

    std::vector<uint8_t> raw;
    globalMem_->rd_from_globalmem(descAddr, sizeof(SFUJobDesc), raw);
    if (raw.size() != sizeof(SFUJobDesc)) {
        return false;
    }

    std::memcpy(desc, raw.data(), sizeof(SFUJobDesc));
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
        case static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX):
        case static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM):
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

SFUStatus SFU::validateJobDescriptor(const SFUJobDesc& desc) const
{
    constexpr uint32_t supportedFlags =
        SFU_JOB_FLAG_DISTRIBUTED_COLUMNS | SFU_JOB_FLAG_DISTRIBUTED_ABORT;
    if ((desc.flags & ~supportedFlags) != 0 ||
        ((desc.flags & SFU_JOB_FLAG_DISTRIBUTED_ABORT) != 0 &&
         (desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) == 0)) {
        return SFUStatus::InvalidDescriptor;
    }
    if (desc.dtype != GOLEM_DTYPE_FP32_VALUE) {
        return SFUStatus::UnsupportedElemBytes;
    }
    if (desc.input0_addr == 0 || desc.output_addr == 0) {
        return SFUStatus::InvalidShape;
    }
    if (desc.worker_cores == 0 || desc.chunk_elems == 0) {
        return SFUStatus::InvalidShape;
    }
    if ((desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0) {
        if (desc.reserved0 >= desc.worker_cores ||
            desc.worker_cores > activeWorkerCores_ || desc.owner_core > coreId_ ||
            coreId_ - desc.owner_core >= desc.worker_cores ||
            coreId_ - desc.owner_core != desc.reserved0) {
            return SFUStatus::InvalidShape;
        }
    }

    switch (desc.op_type) {
        case static_cast<uint32_t>(SFUJobOp::SOFTMAX_ROW):
            if (desc.rows == 0 || desc.cols == 0) {
                return SFUStatus::InvalidShape;
            }
            return SFUStatus::Success;
        case static_cast<uint32_t>(SFUJobOp::ELEMENTWISE):
        case static_cast<uint32_t>(SFUJobOp::REDUCE):
            if (desc.elem_count == 0) {
                return SFUStatus::InvalidShape;
            }
            return SFUStatus::Success;
        default:
            return SFUStatus::InvalidDescriptor;
    }
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

    if (state->desc.op == static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX)) {
        auto maxIt = std::max_element(state->input0.begin(), state->input0.end());
        if (maxIt == state->input0.end()) {
            return false;
        }
        state->output.assign(1, 0.0f);
        state->output[0] = *maxIt;
        return true;
    }
    if (state->desc.op == static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM)) {
        double sum = 0.0;
        for (float value : state->input0) {
            sum += static_cast<double>(value);
        }
        state->output.assign(1, 0.0f);
        state->output[0] = static_cast<float>(sum);
        return true;
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
    const bool reductionOutput =
        (desc.op == static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX) ||
         desc.op == static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM)) &&
        values.size() == 1;
    if (globalMem_ == nullptr || (values.size() != desc.elem_count && !reductionOutput)) {
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

bool SFU::executeJob(JobOpState* state)
{
    if (state == nullptr) {
        return false;
    }

    switch (state->desc.op_type) {
        case static_cast<uint32_t>(SFUJobOp::SOFTMAX_ROW):
            if ((state->desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) != 0) {
                if ((state->desc.flags & SFU_JOB_FLAG_DISTRIBUTED_ABORT) != 0) {
                    abortDistributedSoftmaxJob(state);
                    state->stage = SoftmaxJobStage::Complete;
                    state->status = SFUStatus::InvalidDescriptor;
                    return true;
                }
                return executeDistributedSoftmaxRowJob(state);
            }
            return executeSoftmaxRowJob(state);
        default:
            return false;
    }
}

bool SFU::executeSoftmaxRowJob(JobOpState* state)
{
    if (state == nullptr || globalMem_ == nullptr) {
        return false;
    }

    const SFUJobDesc& desc = state->desc;
    const uint32_t workerCores = state->desc.worker_cores;
    const uint32_t chunkElems = state->desc.chunk_elems;

    for (uint32_t rowBandBegin = 0; rowBandBegin < desc.rows;
         rowBandBegin += GOLEM_SFU_JOB_SOFTMAX_ROW_BAND_ROWS) {
        SoftmaxJobRowBandState band = {};
        band.rowBegin = rowBandBegin;
        band.rowEnd = std::min(desc.rows, rowBandBegin + GOLEM_SFU_JOB_SOFTMAX_ROW_BAND_ROWS);
        band.bandRows = band.rowEnd - band.rowBegin;
        const uint32_t bandRows = band.bandRows;
        band.localMax.assign(static_cast<size_t>(workerCores) * bandRows,
                             -std::numeric_limits<double>::infinity());
        band.localSum.assign(static_cast<size_t>(workerCores) * bandRows, 0.0);
        band.globalMax.assign(bandRows, -std::numeric_limits<double>::infinity());
        band.globalSum.assign(bandRows, 0.0);

        for (uint32_t worker = 0; worker < workerCores; ++worker) {
            const uint32_t begin = workerColumnBegin(desc.cols, worker, workerCores);
            const uint32_t end = workerColumnEnd(desc.cols, worker, workerCores);
            for (uint32_t col = begin; col < end; col += chunkElems) {
                const uint32_t chunkEnd = std::min(end, col + chunkElems);
                for (uint32_t rowIdx = band.rowBegin; rowIdx < band.rowEnd; ++rowIdx) {
                    std::vector<float> chunk;
                    if (!readSoftmaxJobChunk(globalMem_, desc, rowIdx, col, chunkEnd, &chunk)) {
                        return false;
                    }
                    statJobSoftmaxMaxChunks_->addData(1);
                    const uint32_t bandRow = rowIdx - band.rowBegin;
                    double& localMax = band.localMax[worker * bandRows + bandRow];
                    for (float value : chunk) {
                        localMax = std::max(localMax, static_cast<double>(value));
                    }
                }
            }
        }

        for (uint32_t bandRow = 0; bandRow < bandRows; ++bandRow) {
            for (uint32_t worker = 0; worker < workerCores; ++worker) {
                band.globalMax[bandRow] =
                    std::max(band.globalMax[bandRow],
                             band.localMax[worker * bandRows + bandRow]);
            }
        }

        for (uint32_t worker = 0; worker < workerCores; ++worker) {
            const uint32_t begin = workerColumnBegin(desc.cols, worker, workerCores);
            const uint32_t end = workerColumnEnd(desc.cols, worker, workerCores);
            for (uint32_t col = begin; col < end; col += chunkElems) {
                const uint32_t chunkEnd = std::min(end, col + chunkElems);
                for (uint32_t rowIdx = band.rowBegin; rowIdx < band.rowEnd; ++rowIdx) {
                    std::vector<float> chunk;
                    if (!readSoftmaxJobChunk(globalMem_, desc, rowIdx, col, chunkEnd, &chunk)) {
                        return false;
                    }
                    const uint32_t bandRow = rowIdx - band.rowBegin;
                    double& localSum = band.localSum[worker * bandRows + bandRow];
                    for (float& value : chunk) {
                        const double expValue =
                            std::exp(static_cast<double>(value) - band.globalMax[bandRow]);
                        value = static_cast<float>(expValue);
                        localSum += expValue;
                    }
                    if (!writeSoftmaxJobChunk(globalMem_, desc, rowIdx, col, chunk)) {
                        return false;
                    }
                    statJobSoftmaxSumChunks_->addData(1);
                }
            }
        }

        for (uint32_t bandRow = 0; bandRow < bandRows; ++bandRow) {
            for (uint32_t worker = 0; worker < workerCores; ++worker) {
                band.globalSum[bandRow] += band.localSum[worker * bandRows + bandRow];
            }
            if (band.globalSum[bandRow] == 0.0) {
                return false;
            }
        }

        for (uint32_t worker = 0; worker < workerCores; ++worker) {
            const uint32_t begin = workerColumnBegin(desc.cols, worker, workerCores);
            const uint32_t end = workerColumnEnd(desc.cols, worker, workerCores);
            for (uint32_t col = begin; col < end; col += chunkElems) {
                const uint32_t chunkEnd = std::min(end, col + chunkElems);
                for (uint32_t rowIdx = band.rowBegin; rowIdx < band.rowEnd; ++rowIdx) {
                    std::vector<float> chunk;
                    if (!readSoftmaxJobOutputChunk(globalMem_, desc, rowIdx, col, chunkEnd, &chunk)) {
                        return false;
                    }
                    const uint32_t bandRow = rowIdx - band.rowBegin;
                    const float invSum = static_cast<float>(1.0 / band.globalSum[bandRow]);
                    for (float& value : chunk) {
                        value *= invSum;
                    }
                    if (!writeSoftmaxJobChunk(globalMem_, desc, rowIdx, col, chunk)) {
                        return false;
                    }
                    statJobSoftmaxNormChunks_->addData(1);
                }
            }
        }

        for (uint32_t rowIdx = band.rowBegin; rowIdx < band.rowEnd; ++rowIdx) {
            statSoftmaxRows_->addData(1);
        }
    }

    state->processedElems = static_cast<uint64_t>(desc.rows) * desc.cols;
    return true;
}

bool SFU::executeDistributedSoftmaxRowJob(JobOpState* state)
{
    if (state == nullptr || globalMem_ == nullptr) {
        return false;
    }
    if (explicitDistributedReductionEnabled() && !globalMem_->reductionNetworkAvailable()) {
        return false;
    }

    const SFUJobDesc& desc = state->desc;
    state->workerSlot = desc.reserved0;
    state->colBegin = workerColumnBegin(desc.cols, state->workerSlot, desc.worker_cores);
    state->colEnd = workerColumnEnd(desc.cols, state->workerSlot, desc.worker_cores);
    const uint32_t localCols = state->colEnd - state->colBegin;
    if (localCols == 0) {
        return false;
    }

    state->localMax.assign(desc.rows, -std::numeric_limits<double>::infinity());
    state->maxResponseSeen.assign(desc.rows, 0);
    state->sumResponseSeen.assign(desc.rows, 0);
    for (uint32_t col = state->colBegin; col < state->colEnd; col += desc.chunk_elems) {
        const uint32_t chunkEnd = std::min(state->colEnd, col + desc.chunk_elems);
        for (uint32_t row = 0; row < desc.rows; ++row) {
            std::vector<float> chunk;
            if (!readDistributedSoftmaxJobChunk(globalMem_,
                                                 desc.input0_addr,
                                                 desc.rows,
                                                 localCols,
                                                 row,
                                                 col - state->colBegin,
                                                 chunkEnd - state->colBegin,
                                                 &chunk)) {
                return false;
            }
            statJobSoftmaxMaxChunks_->addData(1);
            for (float value : chunk) {
                state->localMax[row] =
                    std::max(state->localMax[row], static_cast<double>(value));
            }
        }
    }

    for (uint32_t row = 0; row < desc.rows; ++row) {
        DistributedReducerResult result = DistributedReducerResult::Invalid;
        if (explicitDistributedReductionEnabled()) {
            ReductionTransportMessage message;
            message.kind = ReductionTransportMessageKind::MaxRequest;
            message.jobId = desc.job_id;
            message.tag = state->tag;
            message.ownerCore = desc.owner_core;
            message.workerSlot = state->workerSlot;
            message.row = row;
            message.expectedWorkers = desc.worker_cores;
            message.expectedRows = desc.rows;
            message.expectedCols = desc.cols;
            message.value = state->localMax[row];
            if (!globalMem_->sendReductionMessage(desc.owner_core, message)) {
                abortDistributedSoftmaxJob(state);
                return false;
            }
            recordDistributedReductionRequest(true);
            result = DistributedReducerResult::Accepted;
        } else {
            result = submitDistributedSoftmaxMax(desc.job_id,
                                                 state->tag,
                                                 desc.owner_core,
                                                 row,
                                                 state->workerSlot,
                                                 desc.worker_cores,
                                                 desc.rows,
                                                 desc.cols,
                                                 state->localMax[row]);
        }
        if (result == DistributedReducerResult::Aborted) {
            observeDistributedSoftmaxJobAbort(state);
            return false;
        }
        if (result != DistributedReducerResult::Accepted) {
            return false;
        }
        if (!explicitDistributedReductionEnabled()) {
            recordDistributedReductionRequest(true);
        }
    }
    statPartialSubmits_->addData(desc.rows);
    state->processedElems = static_cast<uint64_t>(desc.rows) * localCols;
    state->stage = SoftmaxJobStage::MaxSubmitted;
    state->status = SFUStatus::Pending;
    return true;
}

bool SFU::advanceDistributedSoftmaxJob(JobOpState* state)
{
    if (state == nullptr || globalMem_ == nullptr) {
        return false;
    }

    const SFUJobDesc& desc = state->desc;
    const uint32_t localCols = state->colEnd - state->colBegin;
    if (state->stage == SoftmaxJobStage::MaxSubmitted) {
        std::vector<double> globalMax(desc.rows, 0.0);
        if (explicitDistributedReductionEnabled()) {
            std::vector<DistributedReductionResponseInboxKey> responseKeys;
            responseKeys.reserve(desc.rows);
            for (uint32_t row = 0; row < desc.rows; ++row) {
                if (distributedSoftmaxRowAborted(distributedSoftmaxReducerKey(
                        desc.job_id, state->tag, desc.owner_core, row))) {
                    observeDistributedSoftmaxJobAbort(state);
                    return false;
                }
                const DistributedReductionResponseInboxKey key(
                    desc.job_id,
                    state->tag,
                    desc.owner_core,
                    row,
                    state->workerSlot,
                    ReductionTransportMessageKind::MaxResponse);
                if (distributedReductionResponseInbox_.find(key) ==
                    distributedReductionResponseInbox_.end()) {
                    return true;
                }
                responseKeys.push_back(key);
            }
            for (uint32_t row = 0; row < desc.rows; ++row) {
                auto response = distributedReductionResponseInbox_.find(responseKeys[row]);
                globalMax[row] = response->second.value;
                distributedReductionResponseInbox_.erase(response);
                state->maxResponseSeen[row] = 1;
            }
        } else {
            for (uint32_t row = 0; row < desc.rows; ++row) {
                double reducedMax = 0.0;
                const auto result = distributedSoftmaxMaxReady(desc.job_id,
                                                               state->tag,
                                                               desc.owner_core,
                                                               row,
                                                               state->workerSlot,
                                                               &reducedMax);
                if (result == DistributedReducerResult::Pending) {
                    return true;
                }
                if (result == DistributedReducerResult::Aborted) {
                    observeDistributedSoftmaxJobAbort(state);
                    return false;
                }
                if (result != DistributedReducerResult::Ready) {
                    return false;
                }
                globalMax[row] = reducedMax;
                if (row < state->maxResponseSeen.size() && state->maxResponseSeen[row] == 0) {
                    recordDistributedReductionResponse(true);
                    state->maxResponseSeen[row] = 1;
                }
            }
        }

        state->localSum.assign(desc.rows, 0.0);
        for (uint32_t col = state->colBegin; col < state->colEnd; col += desc.chunk_elems) {
            const uint32_t chunkEnd = std::min(state->colEnd, col + desc.chunk_elems);
            for (uint32_t row = 0; row < desc.rows; ++row) {
                std::vector<float> chunk;
                if (!readDistributedSoftmaxJobChunk(globalMem_,
                                                     desc.input0_addr,
                                                     desc.rows,
                                                     localCols,
                                                     row,
                                                     col - state->colBegin,
                                                     chunkEnd - state->colBegin,
                                                     &chunk)) {
                    return false;
                }
                for (float& value : chunk) {
                    const double expValue =
                        std::exp(static_cast<double>(value) - globalMax[row]);
                    value = static_cast<float>(expValue);
                    state->localSum[row] += expValue;
                }
                if (!writeDistributedSoftmaxJobChunk(globalMem_,
                                                      desc.output_addr,
                                                      desc.rows,
                                                      localCols,
                                                      row,
                                                      col - state->colBegin,
                                                      chunk)) {
                    return false;
                }
                statJobSoftmaxSumChunks_->addData(1);
            }
        }
        for (uint32_t row = 0; row < desc.rows; ++row) {
            DistributedReducerResult result = DistributedReducerResult::Invalid;
            if (explicitDistributedReductionEnabled()) {
                ReductionTransportMessage message;
                message.kind = ReductionTransportMessageKind::SumRequest;
                message.jobId = desc.job_id;
                message.tag = state->tag;
                message.ownerCore = desc.owner_core;
                message.workerSlot = state->workerSlot;
                message.row = row;
                message.expectedWorkers = desc.worker_cores;
                message.expectedRows = desc.rows;
                message.expectedCols = desc.cols;
                message.value = state->localSum[row];
                if (!globalMem_->sendReductionMessage(desc.owner_core, message)) {
                    abortDistributedSoftmaxJob(state);
                    return false;
                }
                recordDistributedReductionRequest(false);
                result = DistributedReducerResult::Accepted;
            } else {
                result = submitDistributedSoftmaxSum(desc.job_id,
                                                     state->tag,
                                                     desc.owner_core,
                                                     row,
                                                     state->workerSlot,
                                                     desc.worker_cores,
                                                     desc.rows,
                                                     desc.cols,
                                                     state->localSum[row]);
            }
            if (result == DistributedReducerResult::Aborted) {
                observeDistributedSoftmaxJobAbort(state);
                return false;
            }
            if (result != DistributedReducerResult::Accepted) {
                return false;
            }
            if (!explicitDistributedReductionEnabled()) {
                recordDistributedReductionRequest(false);
            }
        }
        statPartialSubmits_->addData(desc.rows);
        state->stage = SoftmaxJobStage::SumSubmitted;
        return true;
    }

    if (state->stage == SoftmaxJobStage::SumSubmitted) {
        std::vector<double> globalSum(desc.rows, 0.0);
        if (explicitDistributedReductionEnabled()) {
            std::vector<DistributedReductionResponseInboxKey> responseKeys;
            responseKeys.reserve(desc.rows);
            for (uint32_t row = 0; row < desc.rows; ++row) {
                if (distributedSoftmaxRowAborted(distributedSoftmaxReducerKey(
                        desc.job_id, state->tag, desc.owner_core, row))) {
                    observeDistributedSoftmaxJobAbort(state);
                    return false;
                }
                const DistributedReductionResponseInboxKey key(
                    desc.job_id,
                    state->tag,
                    desc.owner_core,
                    row,
                    state->workerSlot,
                    ReductionTransportMessageKind::SumResponse);
                if (distributedReductionResponseInbox_.find(key) ==
                    distributedReductionResponseInbox_.end()) {
                    return true;
                }
                responseKeys.push_back(key);
            }
            for (uint32_t row = 0; row < desc.rows; ++row) {
                auto response = distributedReductionResponseInbox_.find(responseKeys[row]);
                globalSum[row] = response->second.value;
                distributedReductionResponseInbox_.erase(response);
                state->sumResponseSeen[row] = 1;
                if (globalSum[row] == 0.0) {
                    return false;
                }
            }
        } else {
            for (uint32_t row = 0; row < desc.rows; ++row) {
                double reducedSum = 0.0;
                const auto result = distributedSoftmaxSumReady(desc.job_id,
                                                               state->tag,
                                                               desc.owner_core,
                                                               row,
                                                               state->workerSlot,
                                                               &reducedSum);
                if (result == DistributedReducerResult::Pending) {
                    return true;
                }
                if (result == DistributedReducerResult::Aborted) {
                    observeDistributedSoftmaxJobAbort(state);
                    return false;
                }
                if (result != DistributedReducerResult::Ready) {
                    return false;
                }
                globalSum[row] = reducedSum;
                if (row < state->sumResponseSeen.size() && state->sumResponseSeen[row] == 0) {
                    recordDistributedReductionResponse(false);
                    state->sumResponseSeen[row] = 1;
                }
                if (globalSum[row] == 0.0) {
                    return false;
                }
            }
        }

        for (uint32_t col = state->colBegin; col < state->colEnd; col += desc.chunk_elems) {
            const uint32_t chunkEnd = std::min(state->colEnd, col + desc.chunk_elems);
            for (uint32_t row = 0; row < desc.rows; ++row) {
                std::vector<float> chunk;
                if (!readDistributedSoftmaxJobChunk(globalMem_,
                                                     desc.output_addr,
                                                     desc.rows,
                                                     localCols,
                                                     row,
                                                     col - state->colBegin,
                                                     chunkEnd - state->colBegin,
                                                     &chunk)) {
                    return false;
                }
                const float invSum = static_cast<float>(1.0 / globalSum[row]);
                for (float& value : chunk) {
                    value *= invSum;
                }
                if (!writeDistributedSoftmaxJobChunk(globalMem_,
                                                      desc.output_addr,
                                                      desc.rows,
                                                      localCols,
                                                      row,
                                                      col - state->colBegin,
                                                      chunk)) {
                    return false;
                }
                statJobSoftmaxNormChunks_->addData(1);
            }
        }
        for (uint32_t row = 0; row < desc.rows; ++row) {
            const auto result = markDistributedSoftmaxNormalized(desc.job_id,
                                                                 state->tag,
                                                                 desc.owner_core,
                                                                 row,
                                                                 state->workerSlot,
                                                                 desc.worker_cores,
                                                                 desc.rows,
                                                                 desc.cols);
            if (result == DistributedReducerResult::Aborted) {
                observeDistributedSoftmaxJobAbort(state);
                return false;
            }
            if (result != DistributedReducerResult::Accepted) {
                return false;
            }
            statSoftmaxRows_->addData(1);
        }
        statPartialDone_->addData(desc.rows);
        state->stage = SoftmaxJobStage::Complete;
        state->status = SFUStatus::Success;
        return true;
    }

    return state->stage == SoftmaxJobStage::Complete;
}

void SFU::abortDistributedSoftmaxJob(JobOpState* state)
{
    if (state == nullptr ||
        (state->desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) == 0) {
        return;
    }
    clearDistributedReductionResponseInbox(*state);
    if (state->distributedAbortObserved ||
        state->desc.worker_cores == 0 || state->desc.reserved0 >= state->desc.worker_cores) {
        return;
    }
    auto& rows = distributedSoftmaxReducerRows();
    const auto existingKeys = collectDistributedSoftmaxJobKeys(state->desc.job_id,
                                                               state->tag,
                                                               state->desc.owner_core);
    for (const auto& key : existingKeys) {
        auto it = rows.find(key);
        if (it == rows.end()) {
            continue;
        }
        it->second.aborted = true;
        markDistributedSoftmaxRowAborted(key);
        clearDistributedReductionResponseFanout(key);
        if (state->desc.reserved0 < it->second.expectedWorkers) {
            (void)observeDistributedSoftmaxAbort(key, &it->second, state->desc.reserved0);
        }
    }
    for (uint32_t row = 0; row < state->desc.rows; ++row) {
        const auto key = distributedSoftmaxReducerKey(state->desc.job_id,
                                                      state->tag,
                                                      state->desc.owner_core,
                                                      row);
        if (std::find(existingKeys.begin(), existingKeys.end(), key) != existingKeys.end()) {
            continue;
        }
        abortDistributedSoftmaxRow(state->desc.job_id,
                                   state->tag,
                                   state->desc.owner_core,
                                   row,
                                   state->desc.reserved0,
                                   state->desc.worker_cores,
                                   state->desc.rows,
                                   state->desc.cols);
    }
}

void SFU::clearDistributedReductionResponseInbox(const JobOpState& state)
{
    for (auto it = distributedReductionResponseInbox_.begin();
         it != distributedReductionResponseInbox_.end();) {
        const DistributedReductionResponseInboxKey& key = it->first;
        if (std::get<0>(key) == state.desc.job_id && std::get<1>(key) == state.tag &&
            std::get<2>(key) == state.desc.owner_core) {
            it = distributedReductionResponseInbox_.erase(it);
        } else {
            ++it;
        }
    }
}

void SFU::observeDistributedSoftmaxJobAbort(JobOpState* state)
{
    if (state == nullptr || state->distributedAbortObserved ||
        (state->desc.flags & SFU_JOB_FLAG_DISTRIBUTED_COLUMNS) == 0) {
        return;
    }
    auto& rows = distributedSoftmaxReducerRows();
    const auto keys = collectDistributedSoftmaxJobKeys(state->desc.job_id,
                                                       state->tag,
                                                       state->desc.owner_core);
    for (const auto& key : keys) {
        auto it = rows.find(key);
        if (it != rows.end() && it->second.aborted) {
            (void)observeDistributedSoftmaxAbort(key, &it->second, state->desc.reserved0);
        }
    }
    state->distributedAbortObserved = true;
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
    if (globalMem_ != nullptr && globalMem_ != globalMem) {
        globalMem_->setReductionMessageHandler(GlobalMemoryAPI::ReductionMessageHandler{});
    }
    globalMem_ = globalMem;
    if (globalMem_ != nullptr) {
        globalMem_->setReductionMessageHandler(
            [this](const ReductionTransportMessage& message) {
                handleReductionTransportMessage(message);
            });
    }
}

void SFU::setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores)
{
    coreId_ = coreId;
    activeWorkerCores_ = activeWorkerCores == 0 ? 1 : activeWorkerCores;
}

} // namespace Golem
} // namespace SST
