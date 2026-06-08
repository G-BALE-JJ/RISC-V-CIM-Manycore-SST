#include <sst/elements/golem/requestscheduler/requestscheduler.h>

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>

namespace {
constexpr uint64_t SCHED_LOCAL_MAILBOX_BASE = 0x1A00;
constexpr uint64_t SCHED_LOCAL_SUBMIT_HEAD_OFF = 0x00;
constexpr uint64_t SCHED_LOCAL_SUBMIT_TAIL_OFF = 0x08;
constexpr uint64_t SCHED_LOCAL_DONE_HEAD_OFF = 0x10;
constexpr uint64_t SCHED_LOCAL_DONE_TAIL_OFF = 0x18;

constexpr uint64_t SCHED_LOCAL_SUBMIT_RING_BASE = 0x40;
constexpr uint64_t SCHED_LOCAL_SUBMIT_ENTRY_SIZE = 0x30;
constexpr uint64_t SCHED_LOCAL_SUBMIT_ID_OFF = 0x00;
constexpr uint64_t SCHED_LOCAL_SUBMIT_SRC_OFF = 0x08;
constexpr uint64_t SCHED_LOCAL_SUBMIT_DST_OFF = 0x10;
constexpr uint64_t SCHED_LOCAL_SUBMIT_BYTES_OFF = 0x18;
constexpr uint64_t SCHED_LOCAL_SUBMIT_AUX0_OFF = 0x20;
constexpr uint64_t SCHED_LOCAL_SUBMIT_AUX1_OFF = 0x28;

constexpr uint64_t SCHED_LOCAL_DONE_RING_BASE = 0x280;
constexpr uint64_t SCHED_LOCAL_DONE_ENTRY_SIZE = 0x08;
constexpr uint64_t SCHED_LOCAL_DONE_ID_OFF = 0x00;

constexpr uint64_t SCHED_LOCAL_SUBMIT_RING_DEPTH = 8;
constexpr uint64_t SCHED_LOCAL_DONE_RING_DEPTH = 8;
constexpr uint8_t SCHED_PAIR_SLOT_MARKER = 0xFE;

inline uint64_t submitEntryOff(uint64_t ringIdx, uint64_t fieldOff) {
    const uint64_t slot = ringIdx % SCHED_LOCAL_SUBMIT_RING_DEPTH;
    return SCHED_LOCAL_SUBMIT_RING_BASE + slot * SCHED_LOCAL_SUBMIT_ENTRY_SIZE + fieldOff;
}

inline uint64_t doneEntryOff(uint64_t ringIdx, uint64_t fieldOff = SCHED_LOCAL_DONE_ID_OFF) {
    const uint64_t slot = ringIdx % SCHED_LOCAL_DONE_RING_DEPTH;
    return SCHED_LOCAL_DONE_RING_BASE + slot * SCHED_LOCAL_DONE_ENTRY_SIZE + fieldOff;
}

struct TraceStats {
    size_t count = 0;
    double mean = 0.0;
    uint64_t p50 = 0;
    uint64_t p95 = 0;
    uint64_t min = 0;
    uint64_t max = 0;
};

TraceStats buildTraceStats(const std::vector<uint64_t>& samples) {
    TraceStats out{};
    if (samples.empty()) {
        return out;
    }
    out.count = samples.size();
    long double sum = 0.0;
    for (uint64_t v : samples) {
        sum += static_cast<long double>(v);
    }
    out.mean = static_cast<double>(sum / static_cast<long double>(samples.size()));
    std::vector<uint64_t> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    out.min = sorted.front();
    out.max = sorted.back();
    const size_t idx50 = static_cast<size_t>(std::llround(0.50 * static_cast<double>(sorted.size() - 1)));
    const size_t idx95 = static_cast<size_t>(std::llround(0.95 * static_cast<double>(sorted.size() - 1)));
    out.p50 = sorted[idx50];
    out.p95 = sorted[idx95];
    return out;
}

int envFlagDefault(const char* name, int defaultValue) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return defaultValue;
    }
    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return (value == "1" || value == "true" || value == "yes" || value == "on") ? 1 : 0;
}
}

namespace SST {
namespace Golem {

RequestSchedulerEndpoint::RequestSchedulerEndpoint(SST::ComponentId_t id, SST::Params& params)
    : RequestSchedulerAPI(id, params),
      role_(parseRole(params.find<std::string>("role", "worker"))),
      coreId_(parseU32Param(params, "core_id", 0)),
      groupId_(parseU32Param(params, "group_id", 0)),
      workerSlot_(parseI32Param(params, "worker_slot", -1)),
      queueDepth_(parseU32Param(params, "queue_depth", 64)),
      initialNodeCredit_(parseU32Param(params, "initial_node_chunk_credit",
                                       parseU32Param(params, "initial_node_credit", 2))),
      nodeCreditChunkBytes_(parseU32Param(params, "node_credit_chunk_bytes", 512)),
      dmaPrefetchDepth_(parseU32Param(params, "dma_prefetch_depth", 2)),
      workerCreditCap_(0),
      submitBatchSize_(parseU32Param(params, "submit_batch_size", 4)),
      doneBatchSize_(parseU32Param(params, "done_batch_size", 4)),
      issueBudgetPerTick_(parseU32Param(params, "issue_budget_per_tick", 0)),
      smoothStreamEnable_(parseI32Param(params, "smooth_stream_enable", 0) != 0),
      smoothReadBwBytesPerCycle_(static_cast<double>(parseU32Param(params, "smooth_read_bw_bytes_per_cycle", 0))),
      smoothWriteBwBytesPerCycle_(static_cast<double>(parseU32Param(params, "smooth_write_bw_bytes_per_cycle", 0))),
      smoothTokenBurstBytes_(parseU32Param(params, "smooth_token_burst_bytes", 65536)),
      windowKtiles_(parseU32Param(params, "window_k_tiles", 1)),
      numMemoryNodes_(parseU32Param(params, "num_memory_nodes", 5)),
      inferSubmitBytes_(parseI32Param(params, "infer_submit_bytes", 0) != 0),
      traceEvents_(parseI32Param(params, "trace_events", envFlagDefault("GOLEM_REQUEST_SCHEDULER_TRACE", 0)) != 0),
      slot0Bytes_(parseU32Param(params, "slot0_bytes", 0)),
      slot1Bytes_(parseU32Param(params, "slot1_bytes", 0)),
      ctrlLatency_(params.find<std::string>("ctrl_latency", "2ns")),
      gmBaseAddr_(parseU64Param(params, "gm_base_addr", 0)),
      gmSize_(parseU64Param(params, "gm_size", 0)),
      verbose_(parseI32Param(params, "verbose", 0)),
      output_("RequestScheduler[@p:@l]: ", verbose_, 0, SST::Output::STDOUT),
      reqOut_(nullptr),
      linkControl_(nullptr),
      networkId_(-1),
      gm_(nullptr),
      scheduleCursor_(0),
      submitSeen_(false),
      doneSeen_(false),
      nextWindowTxnId_(1),
      lastTickCycle_(0)
{
    if (dmaPrefetchDepth_ == 0) {
        dmaPrefetchDepth_ = 1;
    }
    if (submitBatchSize_ == 0) {
        submitBatchSize_ = 1;
    }
    if (doneBatchSize_ == 0) {
        doneBatchSize_ = 1;
    }
    if (windowKtiles_ == 0) {
        windowKtiles_ = 1;
    }
    if (nodeCreditChunkBytes_ == 0) {
        nodeCreditChunkBytes_ = 512;
    }
    workerCreditCap_ = dmaPrefetchDepth_ * 2;
    reqIn_.resize(4, nullptr);
    nodeCredits_.resize(numMemoryNodes_, initialNodeCredit_ == 0 ? 1 : initialNodeCredit_);
    smoothReadTokens_.resize(numMemoryNodes_, static_cast<double>(smoothTokenBurstBytes_));
    smoothNodeWorkerCursor_.resize(numMemoryNodes_, 0);
    // Keep dma_prefetch_depth as a tile-pair knob. Node credits may be
    // chunk-based, but worker credits track logical mat/vec tile requests.
    workerCredits_.resize(reqIn_.size(), workerCreditCap_);
    configureLinks(params);
    registerClock("1GHz", new SST::Clock::Handler<RequestSchedulerEndpoint>(this, &RequestSchedulerEndpoint::tick));
}

uint32_t RequestSchedulerEndpoint::parseU32Param(SST::Params& params, const std::string& key, uint32_t def) {
    bool found = false;
    std::string value = params.find<std::string>(key, "", found);
    if (!found || value.empty()) return def;
    return static_cast<uint32_t>(std::stoull(value, nullptr, 0));
}

int32_t RequestSchedulerEndpoint::parseI32Param(SST::Params& params, const std::string& key, int32_t def) {
    bool found = false;
    std::string value = params.find<std::string>(key, "", found);
    if (!found || value.empty()) return def;
    return static_cast<int32_t>(std::stoll(value, nullptr, 0));
}

uint64_t RequestSchedulerEndpoint::parseU64Param(SST::Params& params, const std::string& key, uint64_t def) {
    bool found = false;
    std::string value = params.find<std::string>(key, "", found);
    if (!found || value.empty()) return def;
    return static_cast<uint64_t>(std::stoull(value, nullptr, 0));
}

RequestSchedulerRole RequestSchedulerEndpoint::parseRole(const std::string& role) const {
    return role == "manager" ? RequestSchedulerRole::MANAGER : RequestSchedulerRole::WORKER;
}

uint32_t RequestSchedulerEndpoint::inferBytesForSlot(uint8_t slot, uint32_t mailboxBytes) const {
    if (!inferSubmitBytes_) {
        return mailboxBytes;
    }
    if (slot == 0 && slot0Bytes_ > 0) {
        return slot0Bytes_;
    }
    if (slot == 1 && slot1Bytes_ > 0) {
        return slot1Bytes_;
    }
    return mailboxBytes;
}

void RequestSchedulerEndpoint::configureLinks(SST::Params& params) {
    auto* tc = getTimeConverter(ctrlLatency_);
    if (role_ == RequestSchedulerRole::WORKER) {
        reqOut_ = configureLink("req_out", tc);
        return;
    }
    for (int slot = 0; slot < 4; ++slot) {
        SST::Event::HandlerBase* handler = nullptr;
        if (slot == 0) handler = new SST::Event::Handler<RequestSchedulerEndpoint>(this, &RequestSchedulerEndpoint::handleReq0);
        else if (slot == 1) handler = new SST::Event::Handler<RequestSchedulerEndpoint>(this, &RequestSchedulerEndpoint::handleReq1);
        else if (slot == 2) handler = new SST::Event::Handler<RequestSchedulerEndpoint>(this, &RequestSchedulerEndpoint::handleReq2);
        else handler = new SST::Event::Handler<RequestSchedulerEndpoint>(this, &RequestSchedulerEndpoint::handleReq3);
        reqIn_[slot] = configureLink("req_in_" + std::to_string(slot), tc, handler);
    }

    linkControl_ = loadUserSubComponent<SST::Interfaces::SimpleNetwork>("networkIF", ComponentInfo::SHARE_NONE, 1);
    if (!linkControl_) {
        SST::Params ifParams;
        ifParams.insert("link_bw", params.find<std::string>("link_bw", "50GB/s"));
        ifParams.insert("input_buf_size", params.find<std::string>("buffer_length", "64KB"));
        ifParams.insert("output_buf_size", params.find<std::string>("buffer_length", "64KB"));
        ifParams.insert("port_name", "rtr");
        linkControl_ = loadAnonymousSubComponent<SST::Interfaces::SimpleNetwork>(
            "merlin.linkcontrol", "networkIF", 0,
            ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS,
            ifParams, 1);
    }
    linkControl_->setNotifyOnReceive(new SST::Interfaces::SimpleNetwork::Handler<RequestSchedulerEndpoint>(this, &RequestSchedulerEndpoint::handleRecv));
    linkControl_->setNotifyOnSend(new SST::Interfaces::SimpleNetwork::Handler2<RequestSchedulerEndpoint, &RequestSchedulerEndpoint::handleSendAvailable>(this));
}

void RequestSchedulerEndpoint::init(unsigned int phase) {
    if (linkControl_) linkControl_->init(phase);
}

void RequestSchedulerEndpoint::setup() {
    gm_ = GlobalMemoryImplement::lookupByCoreId(static_cast<int>(coreId_));
    if (role_ == RequestSchedulerRole::WORKER) {
        if (reqOut_ == nullptr) {
            output_.fatal(CALL_INFO, -1, "worker core=%u missing req_out\n", coreId_);
        }
        if (gm_) {
            writeMailbox(SCHED_LOCAL_SUBMIT_HEAD_OFF, 0);
            writeMailbox(SCHED_LOCAL_SUBMIT_TAIL_OFF, 0);
            writeMailbox(SCHED_LOCAL_DONE_HEAD_OFF, 0);
            writeMailbox(SCHED_LOCAL_DONE_TAIL_OFF, 0);
        }
        return;
    }
    if (linkControl_) {
        networkId_ = linkControl_->getEndpointID();
        linkControl_->setup();
    }
}

void RequestSchedulerEndpoint::finish() {
    if (role_ == RequestSchedulerRole::MANAGER) {
        output_.verbose(CALL_INFO, 1, 0, "manager core=%u pending=%zu\n", coreId_, pendingQ_.size());
        emitManagerTraceSummary();
        emitManagerPressureSummary();
    }
}

uint8_t RequestSchedulerEndpoint::decodeRequestSlot(uint64_t requestId) {
    return static_cast<uint8_t>((requestId >> 48) & 0xffULL);
}

uint64_t RequestSchedulerEndpoint::decodeSiblingRequestId(uint64_t requestId) {
    return requestId ^ (1ULL << 48);
}

uint32_t RequestSchedulerEndpoint::nodeCreditUnitsForBytes(uint32_t bytes) const {
    const uint32_t chunkBytes = nodeCreditChunkBytes_ == 0 ? 512u : nodeCreditChunkBytes_;
    return std::max<uint32_t>(1u, (bytes + chunkBytes - 1u) / chunkBytes);
}

void RequestSchedulerEndpoint::traceSubmitAtManager(uint64_t requestId, uint8_t workerSlot, uint16_t targetNode) {
    if (role_ != RequestSchedulerRole::MANAGER) {
        return;
    }
    RequestTraceState& state = requestTrace_[requestId];
    if (state.submitSeen) {
        return;
    }
    state.submitSeen = true;
    state.submitRecvCycle = lastTickCycle_;
    state.requestSlot = decodeRequestSlot(requestId);
    state.workerSlot = workerSlot;
    state.targetNode = targetNode;
}

void RequestSchedulerEndpoint::traceIssueAtManager(uint64_t requestId) {
    if (role_ != RequestSchedulerRole::MANAGER) {
        return;
    }
    RequestTraceState& state = requestTrace_[requestId];
    if (state.issueSeen) {
        return;
    }
    state.issueSeen = true;
    state.issueCycle = lastTickCycle_;
    state.requestSlot = decodeRequestSlot(requestId);

    if (traceEvents_) {
        output_.output(
            "[RequestScheduler][core=%u] TRACE_REQ_ISSUE cycle=%" PRIu64 " req=%" PRIu64
            " slot=%u worker_slot=%u target_node=%u\n",
            coreId_,
            state.issueCycle,
            requestId,
            static_cast<unsigned>(state.requestSlot),
            static_cast<unsigned>(state.workerSlot),
            static_cast<unsigned>(state.targetNode));
    }

    if (state.submitSeen && state.issueCycle >= state.submitRecvCycle) {
        const uint64_t delta = state.issueCycle - state.submitRecvCycle;
        if (state.requestSlot == 0) {
            traceSubmitToIssueMat_.push_back(delta);
        } else if (state.requestSlot == 1) {
            traceSubmitToIssueVec_.push_back(delta);
        }
    }

    const uint64_t baseId = requestId & ~(1ULL << 48);
    if (tracePairIssueCounted_.find(baseId) == tracePairIssueCounted_.end()) {
        auto itMat = requestTrace_.find(baseId);
        auto itVec = requestTrace_.find(baseId | (1ULL << 48));
        if (itMat != requestTrace_.end() && itVec != requestTrace_.end() &&
            itMat->second.issueSeen && itVec->second.issueSeen) {
            const uint64_t matIssue = itMat->second.issueCycle;
            const uint64_t vecIssue = itVec->second.issueCycle;
            tracePairIssueGap_.push_back(vecIssue >= matIssue ? (vecIssue - matIssue) : (matIssue - vecIssue));
            tracePairIssueCounted_.insert(baseId);
        }
    }
}

void RequestSchedulerEndpoint::traceDoneAtManager(uint64_t requestId, uint64_t pendingClearCycle) {
    if (role_ != RequestSchedulerRole::MANAGER) {
        return;
    }
    RequestTraceState& state = requestTrace_[requestId];
    if (state.doneSeen) {
        return;
    }
    state.doneSeen = true;
    state.doneAckCycle = lastTickCycle_;
    state.requestSlot = decodeRequestSlot(requestId);
    if (pendingClearCycle > 0) {
        state.pendingClearSeen = true;
        state.pendingClearCycle = pendingClearCycle;
    }

    if (state.pendingClearSeen && state.issueSeen && state.pendingClearCycle >= state.issueCycle) {
        const uint64_t delta = state.pendingClearCycle - state.issueCycle;
        if (state.requestSlot == 0) {
            traceIssueToPendingClearMat_.push_back(delta);
        } else if (state.requestSlot == 1) {
            traceIssueToPendingClearVec_.push_back(delta);
        }
    }
    if (state.pendingClearSeen && state.doneAckCycle >= state.pendingClearCycle) {
        const uint64_t delta = state.doneAckCycle - state.pendingClearCycle;
        if (state.requestSlot == 0) {
            tracePendingClearToDoneAckMat_.push_back(delta);
        } else if (state.requestSlot == 1) {
            tracePendingClearToDoneAckVec_.push_back(delta);
        }
    }

    if (traceEvents_) {
        output_.output(
            "[RequestScheduler][core=%u] TRACE_REQ_DONE cycle=%" PRIu64 " req=%" PRIu64
            " slot=%u issue_cycle=%" PRIu64 " pending_clear_cycle=%" PRIu64 " done_ack_cycle=%" PRIu64 "\n",
            coreId_,
            state.doneAckCycle,
            requestId,
            static_cast<unsigned>(state.requestSlot),
            state.issueSeen ? state.issueCycle : 0,
            state.pendingClearSeen ? state.pendingClearCycle : 0,
            state.doneAckCycle);
    }

    if (state.issueSeen && state.doneAckCycle >= state.issueCycle) {
        const uint64_t delta = state.doneAckCycle - state.issueCycle;
        if (state.requestSlot == 0) {
            traceIssueToDoneMat_.push_back(delta);
        } else if (state.requestSlot == 1) {
            traceIssueToDoneVec_.push_back(delta);
        }
    }
    if (state.submitSeen && state.doneAckCycle >= state.submitRecvCycle) {
        const uint64_t delta = state.doneAckCycle - state.submitRecvCycle;
        if (state.requestSlot == 0) {
            traceSubmitToDoneMat_.push_back(delta);
        } else if (state.requestSlot == 1) {
            traceSubmitToDoneVec_.push_back(delta);
        }
    }

    const uint64_t baseId = requestId & ~(1ULL << 48);
    if (tracePairDoneCounted_.find(baseId) == tracePairDoneCounted_.end()) {
        auto itMat = requestTrace_.find(baseId);
        auto itVec = requestTrace_.find(baseId | (1ULL << 48));
        if (itMat != requestTrace_.end() && itVec != requestTrace_.end() &&
            itMat->second.doneSeen && itVec->second.doneSeen) {
            const uint64_t matDone = itMat->second.doneAckCycle;
            const uint64_t vecDone = itVec->second.doneAckCycle;
            tracePairDoneGap_.push_back(vecDone >= matDone ? (vecDone - matDone) : (matDone - vecDone));
            tracePairDoneCounted_.insert(baseId);
        }
    }

    const uint64_t siblingId = decodeSiblingRequestId(requestId);
    auto sibIt = requestTrace_.find(siblingId);
    if (sibIt != requestTrace_.end() && sibIt->second.doneSeen) {
        requestTrace_.erase(requestId);
        requestTrace_.erase(siblingId);
    }
}

void RequestSchedulerEndpoint::emitManagerTraceSummary() const {
    if (role_ != RequestSchedulerRole::MANAGER) {
        return;
    }

    const TraceStats s2iMat = buildTraceStats(traceSubmitToIssueMat_);
    const TraceStats s2iVec = buildTraceStats(traceSubmitToIssueVec_);
    const TraceStats i2pcMat = buildTraceStats(traceIssueToPendingClearMat_);
    const TraceStats i2pcVec = buildTraceStats(traceIssueToPendingClearVec_);
    const TraceStats pc2dMat = buildTraceStats(tracePendingClearToDoneAckMat_);
    const TraceStats pc2dVec = buildTraceStats(tracePendingClearToDoneAckVec_);
    const TraceStats i2dMat = buildTraceStats(traceIssueToDoneMat_);
    const TraceStats i2dVec = buildTraceStats(traceIssueToDoneVec_);
    const TraceStats s2dMat = buildTraceStats(traceSubmitToDoneMat_);
    const TraceStats s2dVec = buildTraceStats(traceSubmitToDoneVec_);
    const TraceStats pairIssue = buildTraceStats(tracePairIssueGap_);
    const TraceStats pairDone = buildTraceStats(tracePairDoneGap_);

    output_.output(
        "[RequestScheduler][core=%u] TRACE_SUBMIT_READY_BREAKDOWN(cycles): "
        "submit_to_issue_mat(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "submit_to_issue_vec(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "issue_to_pending_clear_mat(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "issue_to_pending_clear_vec(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "pending_clear_to_done_ack_mat(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "pending_clear_to_done_ack_vec(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "issue_to_done_mat(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "issue_to_done_vec(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "submit_to_done_mat(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "submit_to_done_vec(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "pair_issue_gap(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ") "
        "pair_done_gap(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " min=%" PRIu64 " max=%" PRIu64 ")\n",
        coreId_,
        s2iMat.count, s2iMat.mean, s2iMat.p50, s2iMat.p95, s2iMat.min, s2iMat.max,
        s2iVec.count, s2iVec.mean, s2iVec.p50, s2iVec.p95, s2iVec.min, s2iVec.max,
        i2pcMat.count, i2pcMat.mean, i2pcMat.p50, i2pcMat.p95, i2pcMat.min, i2pcMat.max,
        i2pcVec.count, i2pcVec.mean, i2pcVec.p50, i2pcVec.p95, i2pcVec.min, i2pcVec.max,
        pc2dMat.count, pc2dMat.mean, pc2dMat.p50, pc2dMat.p95, pc2dMat.min, pc2dMat.max,
        pc2dVec.count, pc2dVec.mean, pc2dVec.p50, pc2dVec.p95, pc2dVec.min, pc2dVec.max,
        i2dMat.count, i2dMat.mean, i2dMat.p50, i2dMat.p95, i2dMat.min, i2dMat.max,
        i2dVec.count, i2dVec.mean, i2dVec.p50, i2dVec.p95, i2dVec.min, i2dVec.max,
        s2dMat.count, s2dMat.mean, s2dMat.p50, s2dMat.p95, s2dMat.min, s2dMat.max,
        s2dVec.count, s2dVec.mean, s2dVec.p50, s2dVec.p95, s2dVec.min, s2dVec.max,
        pairIssue.count, pairIssue.mean, pairIssue.p50, pairIssue.p95, pairIssue.min, pairIssue.max,
        pairDone.count, pairDone.mean, pairDone.p50, pairDone.p95, pairDone.min, pairDone.max);
}

void RequestSchedulerEndpoint::sampleManagerPressure() {
    if (role_ != RequestSchedulerRole::MANAGER) {
        return;
    }
    pressureTicks_++;
    pressurePendingQSamples_.push_back(static_cast<uint64_t>(pendingQ_.size()));
    if (!pendingQ_.empty()) {
        pressurePendingNonEmptyTicks_++;
    }

    uint64_t maxWorkerUsed = 0;
    for (uint32_t credit : workerCredits_) {
        if (workerCreditCap_ > credit) {
            maxWorkerUsed = std::max<uint64_t>(maxWorkerUsed, static_cast<uint64_t>(workerCreditCap_ - credit));
        }
    }
    pressureWorkerUsedMaxSamples_.push_back(maxWorkerUsed);

    uint64_t maxNodeUsed = 0;
    const uint32_t nodeCreditCap = initialNodeCredit_ == 0 ? 1 : initialNodeCredit_;
    for (uint32_t credit : nodeCredits_) {
        if (nodeCreditCap > credit) {
            maxNodeUsed = std::max<uint64_t>(maxNodeUsed, static_cast<uint64_t>(nodeCreditCap - credit));
        }
    }
    pressureNodeUsedMaxSamples_.push_back(maxNodeUsed);
}

void RequestSchedulerEndpoint::emitManagerPressureSummary() const {
    if (role_ != RequestSchedulerRole::MANAGER) {
        return;
    }
    const TraceStats pending = buildTraceStats(pressurePendingQSamples_);
    const TraceStats workerUsed = buildTraceStats(pressureWorkerUsedMaxSamples_);
    const TraceStats nodeUsed = buildTraceStats(pressureNodeUsedMaxSamples_);
    output_.output(
        "[RequestScheduler][core=%u] SCHED_PRESSURE: "
        "ticks=%" PRIu64 " pending_nonempty_ticks=%" PRIu64 " no_issue_ticks=%" PRIu64
        " issued_requests=%" PRIu64 " issued_pairs=%" PRIu64
        " pending_q(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " max=%" PRIu64 ")"
        " worker_used_max(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " max=%" PRIu64 " cap=%u)"
        " node_used_max(n=%zu mean=%.2f p50=%" PRIu64 " p95=%" PRIu64 " max=%" PRIu64 " cap=%u chunk_bytes=%u)"
        " blocked(worker_credit=%" PRIu64 " node_credit=%" PRIu64 " issue_budget=%" PRIu64
        " smooth=%" PRIu64 " no_sibling=%" PRIu64 ")\n",
        coreId_,
        pressureTicks_, pressurePendingNonEmptyTicks_, pressureNoIssueTicks_,
        pressureIssuedRequests_, pressureIssuedPairs_,
        pending.count, pending.mean, pending.p50, pending.p95, pending.max,
        workerUsed.count, workerUsed.mean, workerUsed.p50, workerUsed.p95, workerUsed.max, workerCreditCap_,
        nodeUsed.count, nodeUsed.mean, nodeUsed.p50, nodeUsed.p95, nodeUsed.max,
        initialNodeCredit_ == 0 ? 1 : initialNodeCredit_, nodeCreditChunkBytes_,
        pressureWorkerCreditBlocked_, pressureNodeCreditBlocked_, pressureIssueBudgetBlocked_,
        pressureSmoothBlocked_, pressureNoSiblingBlocked_);
}

uint64_t RequestSchedulerEndpoint::submitWindowTransaction(const WcpWindowTransaction& txn)
{
    if (role_ != RequestSchedulerRole::WORKER || gm_ == nullptr) {
        return 0;
    }

    WorkerWindowTxnState state;
    state.txnId = nextWindowTxnId_++;
    state.txn = txn;
    state.tiles.resize(txn.kTiles);

    for (uint32_t i = 0; i < txn.kTiles; ++i) {
        state.tiles[i].issued = false;
    }

    workerWindowTxns_[state.txnId] = state;
    enqueueWindowTiles(workerWindowTxns_[state.txnId]);
    return state.txnId;
}

void RequestSchedulerEndpoint::enqueueWindowTiles(WorkerWindowTxnState& state)
{
    if (gm_ == nullptr) {
        return;
    }
    uint32_t inFlightTiles = 0;
    for (const auto& tile : state.tiles) {
        if (tile.issued && !tile.retired) {
            inFlightTiles++;
        }
    }
    for (uint32_t i = 0; i < state.tiles.size() && inFlightTiles < dmaPrefetchDepth_; ++i) {
        auto& tile = state.tiles[i];
        if (tile.issued) {
            continue;
        }
        const uint32_t slotCount = std::max<uint32_t>(state.txn.slotCount, 1u);
        const uint32_t slotIdx = i % slotCount;
        const uint32_t kWindowTiles = std::max<uint32_t>(state.txn.kWindowTiles, 1u);
        const uint32_t totalKTileCount = std::max<uint32_t>(state.txn.totalKTileCount, kWindowTiles);
        const bool matValid = !state.txn.skipMatRead &&
                              (!state.txn.useIndependentMatVecTiles || i < state.txn.matTileCount);
        const bool vecValid = !state.txn.skipVecRead &&
                              (!state.txn.useIndependentMatVecTiles || i < state.txn.vecTileCount);
        const uint32_t matGroupIdx = state.txn.useIndependentMatVecTiles ? (i / kWindowTiles) : 0;
        const uint32_t matKIdx = state.txn.useIndependentMatVecTiles ? (i % kWindowTiles) : i;
        const uint32_t vecGroupIdx = state.txn.useIndependentMatVecTiles ? (i / kWindowTiles) : 0;
        const uint32_t vecKIdx = state.txn.useIndependentMatVecTiles ? (i % kWindowTiles) : i;

        const uint64_t matSrcBase = state.txn.matBaseAddr +
            static_cast<uint64_t>(state.txn.useIndependentMatVecTiles
                                      ? (matGroupIdx * totalKTileCount + matKIdx)
                                      : i) * state.txn.matStrideBytes;
        const uint64_t vecSrcBase = state.txn.vecBaseAddr +
            static_cast<uint64_t>(state.txn.useIndependentMatVecTiles
                                      ? (vecGroupIdx * totalKTileCount + vecKIdx)
                                      : i) * state.txn.vecStrideBytes;
        const uint64_t matDstBase = state.txn.localMatBaseAddr + static_cast<uint64_t>(slotIdx) * state.txn.localMatSlotStrideBytes;
        const uint64_t vecDstBase = state.txn.localVecBaseAddr + static_cast<uint64_t>(slotIdx) * state.txn.localVecSlotStrideBytes;

        const uint16_t matTargetNode = (state.txn.memNodeSize > 0)
            ? static_cast<uint16_t>((matSrcBase / state.txn.memNodeSize) % std::max<uint32_t>(numMemoryNodes_, 1u))
            : 0;
        const uint16_t vecTargetNode = (state.txn.memNodeSize > 0)
            ? static_cast<uint16_t>((vecSrcBase / state.txn.memNodeSize) % std::max<uint32_t>(numMemoryNodes_, 1u))
            : 0;
        const uint64_t seqBase = (state.txnId << 20) | (static_cast<uint64_t>(i) & 0xfffffULL);
        const uint64_t matRequestId = !matValid
            ? 0
            : ((static_cast<uint64_t>(coreId_ & 0xff) << 56) |
               (static_cast<uint64_t>(0) << 48) |
               (static_cast<uint64_t>(matTargetNode & 0xffff) << 32) |
               (seqBase & 0xffffffffULL));
        const uint64_t vecRequestId = !vecValid
            ? 0
            : ((static_cast<uint64_t>(coreId_ & 0xff) << 56) |
               (static_cast<uint64_t>(1) << 48) |
               (static_cast<uint64_t>(vecTargetNode & 0xffff) << 32) |
               (seqBase & 0xffffffffULL));

        tile.matRequestIds.assign(matValid ? 1 : 0, matRequestId);
        tile.vecRequestIds.assign(vecValid ? 1 : 0, vecRequestId);
        tile.matDoneSentKStep.assign(matValid ? 1 : 0, 0);
        tile.vecDoneSentKStep.assign(vecValid ? 1 : 0, 0);

        if (matValid) {
            gm_->ctrlRegisterPendingReadRequest(matRequestId, matDstBase, gm_->ctrlGetReadFlagAddr(0),
                                                matRequestId & 0xffffffffULL, state.txn.matStrideBytes);
            pendingQ_.push_back({static_cast<uint8_t>(workerSlot_), static_cast<uint16_t>(coreId_), matRequestId,
                                 matSrcBase, matDstBase, static_cast<uint32_t>(state.txn.matStrideBytes), matTargetNode,
                                 gm_->ctrlGetReadFlagAddr(0), matRequestId & 0xffffffffULL});
        }
        if (vecValid) {
            gm_->ctrlRegisterPendingReadRequest(vecRequestId, vecDstBase, gm_->ctrlGetReadFlagAddr(1),
                                                vecRequestId & 0xffffffffULL, state.txn.vecStrideBytes);
            pendingQ_.push_back({static_cast<uint8_t>(workerSlot_), static_cast<uint16_t>(coreId_), vecRequestId,
                                 vecSrcBase, vecDstBase, static_cast<uint32_t>(state.txn.vecStrideBytes), vecTargetNode,
                                 gm_->ctrlGetReadFlagAddr(1), vecRequestId & 0xffffffffULL});
        }

        tile.kStepCount = 1;
        tile.matRequestId = tile.matRequestIds.empty() ? 0 : tile.matRequestIds.front();
        tile.vecRequestId = tile.vecRequestIds.empty() ? 0 : tile.vecRequestIds.front();
        tile.submitCycle = lastTickCycle_;
        tile.matDoneCycle = matValid ? 0 : lastTickCycle_;
        tile.vecDoneCycle = vecValid ? 0 : lastTickCycle_;
        tile.readyCycle = 0;
        tile.issued = true;
        tile.matDoneSent = !matValid;
        tile.vecDoneSent = !vecValid;
        tile.retired = false;
        inFlightTiles++;
    }
}

bool RequestSchedulerEndpoint::isTileReady(uint64_t txnId, uint32_t localTileIdx) const
{
    auto txnIt = workerWindowTxns_.find(txnId);
    if (txnIt == workerWindowTxns_.end() || localTileIdx >= txnIt->second.tiles.size() || gm_ == nullptr) {
        return false;
    }
    const auto& tile = txnIt->second.tiles[localTileIdx];
    if (!tile.matRequestIds.empty() || !tile.vecRequestIds.empty()) {
        const size_t kSteps = std::max(tile.matRequestIds.size(), tile.vecRequestIds.size());
        for (size_t k = 0; k < kSteps; ++k) {
            const bool matPending = k < tile.matRequestIds.size() && gm_->ctrlIsReadRequestPending(tile.matRequestIds[k]);
            const bool vecPending = k < tile.vecRequestIds.size() && gm_->ctrlIsReadRequestPending(tile.vecRequestIds[k]);
            if (matPending || vecPending) {
                return false;
            }
        }
        return true;
    }
    return !gm_->ctrlIsReadRequestPending(tile.matRequestId) &&
           !gm_->ctrlIsReadRequestPending(tile.vecRequestId);
}

bool RequestSchedulerEndpoint::getTileKStepReadiness(
    uint64_t txnId,
    uint32_t localTileIdx,
    uint32_t kStep,
    bool& matReady,
    bool& vecReady) const
{
    matReady = false;
    vecReady = false;
    auto txnIt = workerWindowTxns_.find(txnId);
    if (txnIt == workerWindowTxns_.end() || localTileIdx >= txnIt->second.tiles.size() || gm_ == nullptr) {
        return false;
    }
    const auto& tile = txnIt->second.tiles[localTileIdx];
    if (!tile.matRequestIds.empty() || !tile.vecRequestIds.empty()) {
        const size_t kSteps = std::max(tile.matRequestIds.size(), tile.vecRequestIds.size());
        if (kStep >= kSteps) {
            return false;
        }
        matReady = kStep >= tile.matRequestIds.size() || !gm_->ctrlIsReadRequestPending(tile.matRequestIds[kStep]);
        vecReady = kStep >= tile.vecRequestIds.size() || !gm_->ctrlIsReadRequestPending(tile.vecRequestIds[kStep]);
        return true;
    }
    if (kStep > 0) {
        return false;
    }
    matReady = !gm_->ctrlIsReadRequestPending(tile.matRequestId);
    vecReady = !gm_->ctrlIsReadRequestPending(tile.vecRequestId);
    return true;
}

bool RequestSchedulerEndpoint::getTileTimeline(uint64_t txnId, uint32_t localTileIdx, WcpTileTimelineDebug& out) const
{
    auto txnIt = workerWindowTxns_.find(txnId);
    if (txnIt == workerWindowTxns_.end() || localTileIdx >= txnIt->second.tiles.size()) {
        return false;
    }
    const auto& tile = txnIt->second.tiles[localTileIdx];
    out.submitCycle = tile.submitCycle;
    out.matDoneCycle = tile.matDoneCycle;
    out.vecDoneCycle = tile.vecDoneCycle;
    out.readyCycle = tile.readyCycle;
    return true;
}

uint64_t RequestSchedulerEndpoint::getTimelineCycle() const
{
    return lastTickCycle_;
}

void RequestSchedulerEndpoint::retireTileReady(uint64_t txnId, uint32_t localTileIdx)
{
    auto txnIt = workerWindowTxns_.find(txnId);
    if (txnIt == workerWindowTxns_.end() || localTileIdx >= txnIt->second.tiles.size()) {
        return;
    }
    txnIt->second.tiles[localTileIdx].retired = true;
    enqueueWindowTiles(txnIt->second);
}

bool RequestSchedulerEndpoint::isTransactionDone(uint64_t txnId) const
{
    auto txnIt = workerWindowTxns_.find(txnId);
    if (txnIt == workerWindowTxns_.end()) {
        return false;
    }
    for (size_t i = 0; i < txnIt->second.tiles.size(); ++i) {
        if (!isTileReady(txnId, static_cast<uint32_t>(i))) {
            return false;
        }
    }
    return true;
}

void RequestSchedulerEndpoint::retireTransaction(uint64_t txnId)
{
    auto txnIt = workerWindowTxns_.find(txnId);
    if (txnIt != workerWindowTxns_.end()) {
        flushDoneForWindowTransaction(txnIt->second);
        workerWindowTxns_.erase(txnIt);
    }
}

void RequestSchedulerEndpoint::handleReq0(SST::Event* ev) { handleReq(ev, 0); }
void RequestSchedulerEndpoint::handleReq1(SST::Event* ev) { handleReq(ev, 1); }
void RequestSchedulerEndpoint::handleReq2(SST::Event* ev) { handleReq(ev, 2); }
void RequestSchedulerEndpoint::handleReq3(SST::Event* ev) { handleReq(ev, 3); }

bool RequestSchedulerEndpoint::handleRecv(int vn) {
    bool handled = false;
    SST::Interfaces::SimpleNetwork::Request* req = nullptr;
    while ((req = linkControl_->recv(vn)) != nullptr) {
        handled = true;
        Event* ev = req->takePayload();
        auto* nd = dynamic_cast<NetworkDataEvent*>(ev);
        if (nd) {
            output_.output("RequestScheduler manager core=%u unexpected net recv type=%d addr=0x%" PRIx64
                           " len=%zu from=%" PRIu64 " req=%" PRIu64 "\n",
                           coreId_, static_cast<int>(nd->getType()), nd->getAddr(), nd->getLength(), req->src,
                           nd->getRequestId());
        }
        delete ev;
        delete req;
    }
    return handled;
}

void RequestSchedulerEndpoint::handleReq(SST::Event* ev, int slot) {
    auto* msg = dynamic_cast<RequestSchedulerMsg*>(ev);
    if (!msg) {
        output_.fatal(CALL_INFO, -1, "manager core=%u received invalid scheduler msg\n", coreId_);
    }
    if (msg->type == RequestSchedulerMsgType::SUBMIT) {
        auto enqueue_submit = [&](uint64_t requestId,
                                  uint64_t srcAddr,
                                  uint64_t dstAddr,
                                  uint32_t bytes,
                                  uint16_t targetNode,
                                  uint64_t completionFlagAddr,
                                  uint64_t completionValue) {
            traceSubmitAtManager(requestId, static_cast<uint8_t>(slot), targetNode);
            if (pendingQ_.size() < queueDepth_) {
                pendingQ_.push_back({static_cast<uint8_t>(slot), msg->srcCoreId, requestId, srcAddr, dstAddr,
                                     bytes, targetNode, completionFlagAddr, completionValue});
            }
        };

        if (msg->batchCount > 0) {
            const size_t count = static_cast<size_t>(msg->batchCount);
            const size_t validCount = std::min(
                count,
                std::min(
                    std::min(msg->batchRequestIds.size(), msg->batchSrcAddrs.size()),
                    std::min(
                        std::min(msg->batchDstAddrs.size(), msg->batchBytes.size()),
                        std::min(
                            msg->batchTargetNodes.size(),
                            std::min(msg->batchCompletionFlagAddrs.size(), msg->batchCompletionValues.size())))));
            for (size_t i = 0; i < validCount; ++i) {
                enqueue_submit(msg->batchRequestIds[i],
                               msg->batchSrcAddrs[i],
                               msg->batchDstAddrs[i],
                               msg->batchBytes[i],
                               msg->batchTargetNodes[i],
                               msg->batchCompletionFlagAddrs[i],
                               msg->batchCompletionValues[i]);
            }
        } else {
            enqueue_submit(msg->requestId,
                           msg->srcAddr,
                           msg->dstAddr,
                           msg->bytes,
                           msg->targetNode,
                           msg->completionFlagAddr,
                           msg->completionValue);
        }
    } else if (msg->type == RequestSchedulerMsgType::PARTIAL_CREDIT) {
        auto recycle_partial = [&](uint64_t requestId, uint32_t units) {
            if (requestId == 0 || units == 0) {
                return;
            }
            const uint16_t node = static_cast<uint16_t>((requestId >> 32) & 0xffffULL);
            if (node >= nodeCredits_.size()) {
                return;
            }
            auto unitsIt = issuedNodeCreditUnits_.find(requestId);
            if (unitsIt == issuedNodeCreditUnits_.end() || unitsIt->second == 0) {
                return;
            }
            const uint32_t returned = std::min<uint32_t>(units, unitsIt->second);
            unitsIt->second -= returned;
            const uint32_t cap = initialNodeCredit_ == 0 ? 1 : initialNodeCredit_;
            nodeCredits_[node] = std::min<uint32_t>(cap, nodeCredits_[node] + returned);
        };

        if (msg->batchCount > 0 && !msg->batchRequestIds.empty()) {
            const size_t validCount = std::min(
                static_cast<size_t>(msg->batchCount),
                std::min(msg->batchRequestIds.size(), msg->batchBytes.size()));
            for (size_t i = 0; i < validCount; ++i) {
                recycle_partial(msg->batchRequestIds[i], msg->batchBytes[i]);
            }
        } else {
            recycle_partial(msg->requestId, msg->bytes);
        }
    } else if (msg->type == RequestSchedulerMsgType::DONE) {
        auto recycle_done = [&](uint64_t requestId, uint16_t targetNode, uint64_t pendingClearCycle) {
            uint16_t node = targetNode;
            if (requestId != 0) {
                node = static_cast<uint16_t>((requestId >> 32) & 0xffffULL);
            }
            if (node < nodeCredits_.size()) {
                uint32_t units = 1;
                auto unitsIt = issuedNodeCreditUnits_.find(requestId);
                if (unitsIt != issuedNodeCreditUnits_.end()) {
                    units = unitsIt->second;
                    issuedNodeCreditUnits_.erase(unitsIt);
                }
                const uint32_t cap = initialNodeCredit_ == 0 ? 1 : initialNodeCredit_;
                nodeCredits_[node] = std::min<uint32_t>(cap, nodeCredits_[node] + units);
            }
            if (slot >= 0 && static_cast<size_t>(slot) < workerCredits_.size()) {
                uint32_t& credit = workerCredits_[static_cast<size_t>(slot)];
                if (credit < workerCreditCap_) {
                    credit++;
                }
            }
            traceDoneAtManager(requestId, pendingClearCycle);
        };

        if (msg->batchCount > 0 && !msg->batchRequestIds.empty()) {
            const size_t validCount = std::min(static_cast<size_t>(msg->batchCount), msg->batchRequestIds.size());
            for (size_t i = 0; i < validCount; ++i) {
                const uint64_t pendingClearCycle =
                    i < msg->batchPendingClearCycles.size() ? msg->batchPendingClearCycles[i] : 0;
                recycle_done(msg->batchRequestIds[i], msg->targetNode, pendingClearCycle);
            }
        } else {
            recycle_done(msg->requestId, msg->targetNode, msg->pendingClearCycle);
        }
    }
    delete msg;
}

bool RequestSchedulerEndpoint::handleSendAvailable(int vn) {
    (void)vn;
    tryIssue();
    return true;
}

uint64_t RequestSchedulerEndpoint::mailboxAddr(uint64_t off) const {
    return gmBaseAddr_ + SCHED_LOCAL_MAILBOX_BASE + off;
}

uint64_t RequestSchedulerEndpoint::readMailbox(uint64_t off) const {
    return gm_ ? gm_->ctrlReadLocalU64(mailboxAddr(off)) : 0;
}

void RequestSchedulerEndpoint::writeMailbox(uint64_t off, uint64_t value) {
    if (gm_) gm_->ctrlWriteLocalU64(mailboxAddr(off), value);
}

void RequestSchedulerEndpoint::pollWorkerMailbox() {
    if (!gm_) return;
    struct SubmitPayload {
        uint64_t requestId;
        uint64_t srcAddr;
        uint64_t dstAddr;
        uint32_t bytes;
        uint16_t targetNode;
        uint64_t completionFlagAddr;
        uint64_t completionValue;
    };

    std::vector<SubmitPayload> pendingSubmits;
    pendingSubmits.reserve(submitBatchSize_);
    auto flushSubmitBatch = [&]() {
        if (pendingSubmits.empty()) {
            return;
        }
        auto* msg = new RequestSchedulerMsg(RequestSchedulerMsgType::SUBMIT);
        msg->groupId = static_cast<uint8_t>(groupId_);
        msg->workerSlot = static_cast<uint8_t>(workerSlot_);
        msg->srcCoreId = static_cast<uint16_t>(coreId_);
        msg->batchCount = static_cast<uint16_t>(pendingSubmits.size());
        msg->batchRequestIds.reserve(pendingSubmits.size());
        msg->batchSrcAddrs.reserve(pendingSubmits.size());
        msg->batchDstAddrs.reserve(pendingSubmits.size());
        msg->batchBytes.reserve(pendingSubmits.size());
        msg->batchTargetNodes.reserve(pendingSubmits.size());
        msg->batchCompletionFlagAddrs.reserve(pendingSubmits.size());
        msg->batchCompletionValues.reserve(pendingSubmits.size());
        for (const SubmitPayload& entry : pendingSubmits) {
            msg->batchRequestIds.push_back(entry.requestId);
            msg->batchSrcAddrs.push_back(entry.srcAddr);
            msg->batchDstAddrs.push_back(entry.dstAddr);
            msg->batchBytes.push_back(entry.bytes);
            msg->batchTargetNodes.push_back(entry.targetNode);
            msg->batchCompletionFlagAddrs.push_back(entry.completionFlagAddr);
            msg->batchCompletionValues.push_back(entry.completionValue);
        }
        const SubmitPayload& first = pendingSubmits.front();
        msg->requestId = first.requestId;
        msg->srcAddr = first.srcAddr;
        msg->dstAddr = first.dstAddr;
        msg->bytes = first.bytes;
        msg->targetNode = first.targetNode;
        msg->completionFlagAddr = first.completionFlagAddr;
        msg->completionValue = first.completionValue;
        if (reqOut_) {
            reqOut_->send(msg);
        } else {
            delete msg;
        }
        pendingSubmits.clear();
    };

    uint64_t submitHead = readMailbox(SCHED_LOCAL_SUBMIT_HEAD_OFF);
    const uint64_t submitTail = readMailbox(SCHED_LOCAL_SUBMIT_TAIL_OFF);
    while (submitHead < submitTail) {
        const uint64_t requestId = readMailbox(submitEntryOff(submitHead, SCHED_LOCAL_SUBMIT_ID_OFF));
        const uint64_t srcAddr = readMailbox(submitEntryOff(submitHead, SCHED_LOCAL_SUBMIT_SRC_OFF));
        const uint64_t dstAddr = readMailbox(submitEntryOff(submitHead, SCHED_LOCAL_SUBMIT_DST_OFF));
        const uint8_t requestSlot = static_cast<uint8_t>((requestId >> 48) & 0xffULL);
        const uint16_t targetNode = static_cast<uint16_t>((requestId >> 32) & 0xffffULL);
        const uint32_t mailboxBytes = static_cast<uint32_t>(
            readMailbox(submitEntryOff(submitHead, SCHED_LOCAL_SUBMIT_BYTES_OFF)));

        auto emit_submit = [&](uint64_t emitRequestId, uint8_t emitSlot, uint64_t emitSrc, uint64_t emitDst) {
            const uint32_t emitBytes = inferBytesForSlot(emitSlot, mailboxBytes);
            const uint64_t completionFlagAddr = gm_->ctrlGetReadFlagAddr(emitSlot);
            const uint64_t completionValue = emitRequestId & 0xffffffffULL;

            const uint16_t emitNode = static_cast<uint16_t>((emitRequestId >> 32) & 0xffffULL);

            gm_->ctrlRegisterPendingReadRequest(
                emitRequestId,
                emitDst,
                completionFlagAddr,
                completionValue,
                emitBytes);

            pendingSubmits.push_back({emitRequestId,
                                      emitSrc,
                                      emitDst,
                                      emitBytes,
                                      emitNode,
                                      completionFlagAddr,
                                      completionValue});
            if (pendingSubmits.size() >= submitBatchSize_) {
                flushSubmitBatch();
            }

            output_.verbose(CALL_INFO, 1, 0,
                            "worker core=%u submit req=%" PRIu64 " slot=%u node=%u src=0x%" PRIx64 " dst=0x%" PRIx64 " bytes=%u\n",
                            coreId_, emitRequestId, emitSlot, emitNode, emitSrc, emitDst, emitBytes);
        };

        if (requestSlot == SCHED_PAIR_SLOT_MARKER) {
            const uint64_t vecSrcAddr = readMailbox(submitEntryOff(submitHead, SCHED_LOCAL_SUBMIT_AUX0_OFF));
            const uint64_t vecDstAddr = readMailbox(submitEntryOff(submitHead, SCHED_LOCAL_SUBMIT_AUX1_OFF));
            const uint64_t seq = requestId & 0xffffffffULL;
            const uint64_t requestIdMat =
                (static_cast<uint64_t>(coreId_ & 0xff) << 56) |
                (static_cast<uint64_t>(0) << 48) |
                (static_cast<uint64_t>(targetNode & 0xffff) << 32) |
                seq;
            const uint64_t requestIdVec =
                (static_cast<uint64_t>(coreId_ & 0xff) << 56) |
                (static_cast<uint64_t>(1) << 48) |
                (static_cast<uint64_t>(targetNode & 0xffff) << 32) |
                seq;
            emit_submit(requestIdMat, 0, srcAddr, dstAddr);
            emit_submit(requestIdVec, 1, vecSrcAddr, vecDstAddr);
        } else {
            emit_submit(requestId, requestSlot, srcAddr, dstAddr);
        }

        submitHead++;
        submitSeen_ = true;
    }
    flushSubmitBatch();
    if (submitHead != readMailbox(SCHED_LOCAL_SUBMIT_HEAD_OFF)) {
        writeMailbox(SCHED_LOCAL_SUBMIT_HEAD_OFF, submitHead);
    }

    uint64_t doneHead = readMailbox(SCHED_LOCAL_DONE_HEAD_OFF);
    const uint64_t doneTail = readMailbox(SCHED_LOCAL_DONE_TAIL_OFF);
    std::vector<uint64_t> doneBatch;
    doneBatch.reserve(doneBatchSize_ > 0 ? doneBatchSize_ : 1);
    auto flush_done_batch = [&]() {
        if (doneBatch.empty()) {
            return;
        }
        auto* msg = new RequestSchedulerMsg(RequestSchedulerMsgType::DONE);
        msg->groupId = static_cast<uint8_t>(groupId_);
        msg->workerSlot = static_cast<uint8_t>(workerSlot_);
        msg->srcCoreId = static_cast<uint16_t>(coreId_);
        msg->requestId = doneBatch.front();
        msg->targetNode = static_cast<uint16_t>((msg->requestId >> 32) & 0xffffULL);
        msg->batchCount = static_cast<uint16_t>(doneBatch.size());
        msg->batchRequestIds = doneBatch;
        if (reqOut_) reqOut_->send(msg);
        output_.verbose(CALL_INFO, 1, 0, "worker core=%u done batch count=%u first_req=%" PRIu64 " node=%u\n",
                        coreId_, msg->batchCount, msg->requestId, msg->targetNode);
        doneBatch.clear();
    };
    while (doneHead < doneTail) {
        doneBatch.push_back(readMailbox(doneEntryOff(doneHead, SCHED_LOCAL_DONE_ID_OFF)));
        doneHead++;
        doneSeen_ = true;
        if (doneBatch.size() >= doneBatchSize_) {
            flush_done_batch();
        }
    }
    flush_done_batch();
    if (doneHead != readMailbox(SCHED_LOCAL_DONE_HEAD_OFF)) {
        writeMailbox(SCHED_LOCAL_DONE_HEAD_OFF, doneHead);
    }
}

void RequestSchedulerEndpoint::flushWorkerDirectSubmits() {
    if (role_ != RequestSchedulerRole::WORKER || reqOut_ == nullptr || pendingQ_.empty()) {
        return;
    }
    while (!pendingQ_.empty()) {
        auto* msg = new RequestSchedulerMsg(RequestSchedulerMsgType::SUBMIT);
        msg->groupId = static_cast<uint8_t>(groupId_);
        msg->workerSlot = static_cast<uint8_t>(workerSlot_);
        msg->srcCoreId = static_cast<uint16_t>(coreId_);
        msg->batchCount = 0;
        msg->batchRequestIds.reserve(submitBatchSize_);
        msg->batchSrcAddrs.reserve(submitBatchSize_);
        msg->batchDstAddrs.reserve(submitBatchSize_);
        msg->batchBytes.reserve(submitBatchSize_);
        msg->batchTargetNodes.reserve(submitBatchSize_);
        msg->batchCompletionFlagAddrs.reserve(submitBatchSize_);
        msg->batchCompletionValues.reserve(submitBatchSize_);
        for (uint32_t count = 0; count < submitBatchSize_ && !pendingQ_.empty(); ++count) {
            PendingTransfer req = pendingQ_.front();
            pendingQ_.pop_front();
            msg->batchRequestIds.push_back(req.requestId);
            msg->batchSrcAddrs.push_back(req.srcAddr);
            msg->batchDstAddrs.push_back(req.dstAddr);
            msg->batchBytes.push_back(req.bytes);
            msg->batchTargetNodes.push_back(req.targetNode);
            msg->batchCompletionFlagAddrs.push_back(req.completionFlagAddr);
            msg->batchCompletionValues.push_back(req.completionValue);
        }
        msg->batchCount = static_cast<uint16_t>(msg->batchRequestIds.size());
        msg->requestId = msg->batchRequestIds.front();
        msg->srcAddr = msg->batchSrcAddrs.front();
        msg->dstAddr = msg->batchDstAddrs.front();
        msg->bytes = msg->batchBytes.front();
        msg->targetNode = msg->batchTargetNodes.front();
        msg->completionFlagAddr = msg->batchCompletionFlagAddrs.front();
        msg->completionValue = msg->batchCompletionValues.front();
        reqOut_->send(msg);
    }
}

void RequestSchedulerEndpoint::flushWorkerPartialCreditReturns() {
    if (role_ != RequestSchedulerRole::WORKER || reqOut_ == nullptr || gm_ == nullptr) {
        return;
    }

    const auto returns = gm_->ctrlDrainReadCreditReturns();
    if (returns.empty()) {
        return;
    }

    const size_t batchLimit = doneBatchSize_ > 0 ? doneBatchSize_ : 1;
    for (size_t idx = 0; idx < returns.size();) {
        auto* msg = new RequestSchedulerMsg(RequestSchedulerMsgType::PARTIAL_CREDIT);
        msg->groupId = static_cast<uint8_t>(groupId_);
        msg->workerSlot = static_cast<uint8_t>(workerSlot_);
        msg->srcCoreId = static_cast<uint16_t>(coreId_);
        msg->batchRequestIds.reserve(batchLimit);
        msg->batchBytes.reserve(batchLimit);
        for (; idx < returns.size() && msg->batchRequestIds.size() < batchLimit; ++idx) {
            if (returns[idx].first == 0 || returns[idx].second == 0) {
                continue;
            }
            msg->batchRequestIds.push_back(returns[idx].first);
            msg->batchBytes.push_back(returns[idx].second);
        }
        if (msg->batchRequestIds.empty()) {
            delete msg;
            continue;
        }
        msg->batchCount = static_cast<uint16_t>(msg->batchRequestIds.size());
        msg->requestId = msg->batchRequestIds.front();
        msg->bytes = msg->batchBytes.front();
        msg->targetNode = static_cast<uint16_t>((msg->requestId >> 32) & 0xffffULL);
        reqOut_->send(msg);
    }
}

void RequestSchedulerEndpoint::flushWorkerDirectDones() {
    if (role_ != RequestSchedulerRole::WORKER || reqOut_ == nullptr || gm_ == nullptr) {
        return;
    }
    std::vector<uint64_t> doneIds;
    std::vector<uint64_t> donePendingClearCycles;
    doneIds.reserve(doneBatchSize_ > 0 ? doneBatchSize_ : 1);
    donePendingClearCycles.reserve(doneBatchSize_ > 0 ? doneBatchSize_ : 1);
    auto flushDone = [&]() {
        if (doneIds.empty()) {
            return;
        }
        auto* msg = new RequestSchedulerMsg(RequestSchedulerMsgType::DONE);
        msg->groupId = static_cast<uint8_t>(groupId_);
        msg->workerSlot = static_cast<uint8_t>(workerSlot_);
        msg->srcCoreId = static_cast<uint16_t>(coreId_);
        msg->requestId = doneIds.front();
        msg->targetNode = static_cast<uint16_t>((msg->requestId >> 32) & 0xffffULL);
        msg->batchCount = static_cast<uint16_t>(doneIds.size());
        msg->batchRequestIds = doneIds;
        msg->pendingClearCycle = donePendingClearCycles.front();
        msg->batchPendingClearCycles = donePendingClearCycles;
        reqOut_->send(msg);
        doneIds.clear();
        donePendingClearCycles.clear();
    };

    for (auto& entry : workerWindowTxns_) {
        for (auto& tile : entry.second.tiles) {
        if (tile.matRequestIds.size() == 1 && tile.vecRequestIds.size() == 1) {
            const uint64_t matReq = tile.matRequestIds[0];
            const uint64_t vecReq = tile.vecRequestIds[0];
            if (tile.matDoneSentKStep[0] == 0 && matReq != 0 && !gm_->ctrlIsReadRequestPending(matReq)) {
                doneIds.push_back(matReq);
                donePendingClearCycles.push_back(lastTickCycle_);
                tile.matDoneSentKStep[0] = 1;
                tile.matDoneSent = true;
                tile.matDoneCycle = lastTickCycle_;
            }
            if (tile.vecDoneSentKStep[0] == 0 && vecReq != 0 && !gm_->ctrlIsReadRequestPending(vecReq)) {
                doneIds.push_back(vecReq);
                donePendingClearCycles.push_back(lastTickCycle_);
                tile.vecDoneSentKStep[0] = 1;
                tile.vecDoneSent = true;
                tile.vecDoneCycle = lastTickCycle_;
            }
            if (doneIds.size() >= doneBatchSize_) {
                flushDone();
            }
        } else if (!tile.matRequestIds.empty() && !tile.vecRequestIds.empty()) {
                const size_t matSteps = tile.matRequestIds.size();
                const size_t vecSteps = tile.vecRequestIds.size();

                bool allMatDone = true;
                for (size_t k = 0; k < matSteps; ++k) {
                    if (k >= tile.matDoneSentKStep.size()) {
                        allMatDone = false;
                        continue;
                    }
                    if (tile.matDoneSentKStep[k] != 0) {
                        continue;
                    }
                    allMatDone = false;
                    const uint64_t reqId = tile.matRequestIds[k];
                    if (reqId != 0 && !gm_->ctrlIsReadRequestPending(reqId)) {
                        doneIds.push_back(reqId);
                        donePendingClearCycles.push_back(lastTickCycle_);
                        tile.matDoneSentKStep[k] = 1;
                    }
                    if (doneIds.size() >= doneBatchSize_) {
                        flushDone();
                    }
                }
                allMatDone = !tile.matDoneSentKStep.empty() &&
                             std::all_of(tile.matDoneSentKStep.begin(), tile.matDoneSentKStep.end(),
                                         [](uint8_t v) { return v != 0; });
                if (!tile.matDoneSent && allMatDone) {
                    tile.matDoneSent = true;
                    tile.matDoneCycle = lastTickCycle_;
                }

                bool allVecDone = true;
                for (size_t k = 0; k < vecSteps; ++k) {
                    if (k >= tile.vecDoneSentKStep.size()) {
                        allVecDone = false;
                        continue;
                    }
                    if (tile.vecDoneSentKStep[k] != 0) {
                        continue;
                    }
                    allVecDone = false;
                    const uint64_t reqId = tile.vecRequestIds[k];
                    if (reqId != 0 && !gm_->ctrlIsReadRequestPending(reqId)) {
                        doneIds.push_back(reqId);
                        donePendingClearCycles.push_back(lastTickCycle_);
                        tile.vecDoneSentKStep[k] = 1;
                    }
                    if (doneIds.size() >= doneBatchSize_) {
                        flushDone();
                    }
                }
                allVecDone = !tile.vecDoneSentKStep.empty() &&
                             std::all_of(tile.vecDoneSentKStep.begin(), tile.vecDoneSentKStep.end(),
                                         [](uint8_t v) { return v != 0; });
                if (!tile.vecDoneSent && allVecDone) {
                    tile.vecDoneSent = true;
                    tile.vecDoneCycle = lastTickCycle_;
                }
            } else {
                if (!tile.matDoneSent && tile.matRequestId != 0 && !gm_->ctrlIsReadRequestPending(tile.matRequestId)) {
                    doneIds.push_back(tile.matRequestId);
                    donePendingClearCycles.push_back(lastTickCycle_);
                    tile.matDoneSent = true;
                    tile.matDoneCycle = lastTickCycle_;
                }
                if (doneIds.size() >= doneBatchSize_) {
                    flushDone();
                }
                if (!tile.vecDoneSent && tile.vecRequestId != 0 && !gm_->ctrlIsReadRequestPending(tile.vecRequestId)) {
                    doneIds.push_back(tile.vecRequestId);
                    donePendingClearCycles.push_back(lastTickCycle_);
                    tile.vecDoneSent = true;
                    tile.vecDoneCycle = lastTickCycle_;
                }
                if (doneIds.size() >= doneBatchSize_) {
                    flushDone();
                }
            }

            if (tile.readyCycle == 0 && tile.matDoneSent && tile.vecDoneSent) {
                tile.readyCycle = lastTickCycle_;
            }
        }
    }
    flushDone();
}

void RequestSchedulerEndpoint::flushDoneForWindowTransaction(WorkerWindowTxnState& state) {
    if (role_ != RequestSchedulerRole::WORKER || reqOut_ == nullptr || gm_ == nullptr) {
        return;
    }

    std::vector<uint64_t> doneIds;
    std::vector<uint64_t> donePendingClearCycles;
    doneIds.reserve(doneBatchSize_ > 0 ? doneBatchSize_ : 1);
    donePendingClearCycles.reserve(doneBatchSize_ > 0 ? doneBatchSize_ : 1);

    auto flushDone = [&]() {
        if (doneIds.empty()) {
            return;
        }
        auto* msg = new RequestSchedulerMsg(RequestSchedulerMsgType::DONE);
        msg->groupId = static_cast<uint8_t>(groupId_);
        msg->workerSlot = static_cast<uint8_t>(workerSlot_);
        msg->srcCoreId = static_cast<uint16_t>(coreId_);
        msg->requestId = doneIds.front();
        msg->targetNode = static_cast<uint16_t>((msg->requestId >> 32) & 0xffffULL);
        msg->batchCount = static_cast<uint16_t>(doneIds.size());
        msg->batchRequestIds = doneIds;
        msg->pendingClearCycle = donePendingClearCycles.front();
        msg->batchPendingClearCycles = donePendingClearCycles;
        reqOut_->send(msg);
        doneIds.clear();
        donePendingClearCycles.clear();
    };

    auto queueDoneNoFlush = [&](uint64_t reqId, uint8_t& sentFlag, uint64_t clearCycle) {
        if (reqId == 0 || sentFlag != 0) {
            return;
        }
        doneIds.push_back(reqId);
        donePendingClearCycles.push_back(clearCycle);
        sentFlag = 1;
    };

    auto queueDone = [&](uint64_t reqId, uint8_t& sentFlag) {
        if (reqId == 0 || sentFlag != 0 || gm_->ctrlIsReadRequestPending(reqId)) {
            return;
        }
        queueDoneNoFlush(reqId, sentFlag, lastTickCycle_);
        if (doneIds.size() >= doneBatchSize_) {
            flushDone();
        }
    };

    for (auto& tile : state.tiles) {
        if (tile.matRequestIds.size() == 1 && tile.vecRequestIds.size() == 1) {
            const uint64_t matReq = tile.matRequestIds[0];
            const uint64_t vecReq = tile.vecRequestIds[0];
            if (tile.matDoneSentKStep[0] == 0 && matReq != 0 && !gm_->ctrlIsReadRequestPending(matReq)) {
                queueDoneNoFlush(matReq, tile.matDoneSentKStep[0], lastTickCycle_);
                tile.matDoneSent = true;
                tile.matDoneCycle = lastTickCycle_;
            }
            if (tile.vecDoneSentKStep[0] == 0 && vecReq != 0 && !gm_->ctrlIsReadRequestPending(vecReq)) {
                queueDoneNoFlush(vecReq, tile.vecDoneSentKStep[0], lastTickCycle_);
                tile.vecDoneSent = true;
                tile.vecDoneCycle = lastTickCycle_;
            }
            if (doneIds.size() >= doneBatchSize_) {
                flushDone();
            }
            if (tile.readyCycle == 0 && tile.matDoneSent && tile.vecDoneSent) {
                tile.readyCycle = lastTickCycle_;
            }
            continue;
        }

        for (size_t k = 0; k < tile.matRequestIds.size(); ++k) {
            if (k < tile.matDoneSentKStep.size()) {
                queueDone(tile.matRequestIds[k], tile.matDoneSentKStep[k]);
            }
        }
        for (size_t k = 0; k < tile.vecRequestIds.size(); ++k) {
            if (k < tile.vecDoneSentKStep.size()) {
                queueDone(tile.vecRequestIds[k], tile.vecDoneSentKStep[k]);
            }
        }
        if (!tile.matRequestIds.empty()) {
            tile.matDoneSent = std::all_of(tile.matDoneSentKStep.begin(), tile.matDoneSentKStep.end(),
                                           [](uint8_t v) { return v != 0; });
        } else if (tile.matRequestId != 0 && !tile.matDoneSent && !gm_->ctrlIsReadRequestPending(tile.matRequestId)) {
            doneIds.push_back(tile.matRequestId);
            donePendingClearCycles.push_back(lastTickCycle_);
            tile.matDoneSent = true;
            if (doneIds.size() >= doneBatchSize_) {
                flushDone();
            }
        }
        if (!tile.vecRequestIds.empty()) {
            tile.vecDoneSent = std::all_of(tile.vecDoneSentKStep.begin(), tile.vecDoneSentKStep.end(),
                                           [](uint8_t v) { return v != 0; });
        } else if (tile.vecRequestId != 0 && !tile.vecDoneSent && !gm_->ctrlIsReadRequestPending(tile.vecRequestId)) {
            doneIds.push_back(tile.vecRequestId);
            donePendingClearCycles.push_back(lastTickCycle_);
            tile.vecDoneSent = true;
            if (doneIds.size() >= doneBatchSize_) {
                flushDone();
            }
        }
    }
    flushDone();
}

void RequestSchedulerEndpoint::refillWorkerWindows() {
    if (role_ != RequestSchedulerRole::WORKER || gm_ == nullptr) {
        return;
    }
    for (auto& entry : workerWindowTxns_) {
        enqueueWindowTiles(entry.second);
    }
}

bool RequestSchedulerEndpoint::issueTransfer(const PendingTransfer& req) {
    if (!gm_) {
        gm_ = GlobalMemoryImplement::lookupByCoreId(static_cast<int>(coreId_));
    }
    if (!linkControl_ || networkId_ < 0 || !gm_) {
        output_.verbose(CALL_INFO, 1, 0, "manager core=%u cannot issue req=%" PRIu64 " link=%p nid=%d gm=%p\n",
                        coreId_, req.requestId, linkControl_, networkId_, gm_);
        return false;
    }
    int workerEp = gm_->ctrlResolveEndpointForAddress(req.dstAddr);
    if (workerEp < 0) {
        output_.verbose(CALL_INFO, 1, 0, "manager core=%u missing worker endpoint dst=0x%" PRIx64 " core=%u req=%" PRIu64 "\n",
                        coreId_, req.dstAddr, req.workerCoreId, req.requestId);
        return false;
    }
    int memnicEp = gm_->ctrlLookupMemNicEndpointId(req.srcAddr);
    auto* snReq = new SST::Interfaces::SimpleNetwork::Request();
    snReq->src = networkId_;
    snReq->dest = memnicEp != -1 ? static_cast<uint64_t>(memnicEp) : static_cast<uint64_t>(gm_->ctrlLookupDmaTargetRouter(req.srcAddr));
    snReq->vn = gm_->getRequestVn();
    auto* payload = new NetworkDataEvent(NetworkDataEvent::READ, req.srcAddr, req.bytes, std::vector<uint8_t>(),
                                         req.dstAddr, workerEp, req.completionFlagAddr, req.completionValue,
                                         req.requestId);
    snReq->size_in_bits = (sizeof(req.srcAddr) + sizeof(req.bytes) + sizeof(req.dstAddr)) * 8;
    snReq->givePayload(payload);
    if (!linkControl_->send(snReq, snReq->vn)) {
        output_.verbose(CALL_INFO, 1, 0, "manager core=%u send blocked req=%" PRIu64 " node=%u\n",
                        coreId_, req.requestId, req.targetNode);
        delete snReq;
        return false;
    }
    output_.verbose(CALL_INFO, 1, 0, "manager core=%u issued req=%" PRIu64 " worker=%u node=%u src=0x%" PRIx64 " dst=0x%" PRIx64 " ep=%d\n",
                    coreId_, req.requestId, req.workerCoreId, req.targetNode, req.srcAddr, req.dstAddr, workerEp);
    traceIssueAtManager(req.requestId);
    return true;
}

void RequestSchedulerEndpoint::tryIssue() {
    if (role_ != RequestSchedulerRole::MANAGER) return;
    if (pendingQ_.empty()) return;
    refillSmoothTokens();
    uint32_t issuedThisTick = 0;
    auto hasIssueBudget = [&](uint32_t need) {
        return issueBudgetPerTick_ == 0 || issuedThisTick + need <= issueBudgetPerTick_;
    };
    auto hasSmoothTokens = [&](const PendingTransfer& req, uint32_t bytesNeed) {
        if (!smoothStreamEnable_ || smoothReadBwBytesPerCycle_ <= 0.0) {
            return true;
        }
        if (req.targetNode >= smoothReadTokens_.size()) {
            return false;
        }
        return smoothReadTokens_[req.targetNode] + 0.5 >= static_cast<double>(bytesNeed);
    };
    auto consumeSmoothTokens = [&](const PendingTransfer& req, uint32_t bytesNeed) {
        if (!smoothStreamEnable_ || smoothReadBwBytesPerCycle_ <= 0.0 || req.targetNode >= smoothReadTokens_.size()) {
            return;
        }
        smoothReadTokens_[req.targetNode] = std::max<double>(0.0, smoothReadTokens_[req.targetNode] - static_cast<double>(bytesNeed));
    };
    auto canIssue = [&](const PendingTransfer& req, uint32_t workerNeed, uint32_t nodeNeed, uint32_t bytesNeed) {
        if (!hasIssueBudget(workerNeed)) {
            pressureIssueBudgetBlocked_++;
            return false;
        }
        if (!hasSmoothTokens(req, bytesNeed)) {
            pressureSmoothBlocked_++;
            return false;
        }
        const bool worker_throttled =
            req.workerSlot >= workerCredits_.size() ||
            workerCredits_[req.workerSlot] < workerNeed;
        if (worker_throttled) {
            pressureWorkerCreditBlocked_++;
            return false;
        }
        if (req.targetNode >= nodeCredits_.size() || nodeCredits_[req.targetNode] < nodeNeed) {
            pressureNodeCreditBlocked_++;
            return false;
        }
        return true;
    };
    auto siblingRequestId = [](uint64_t requestId) {
        return requestId ^ (1ULL << 48);
    };

    bool progress = true;
    while (progress && !pendingQ_.empty()) {
        progress = false;

        for (uint32_t node = 0; node < numMemoryNodes_ && !progress; ++node) {
            const size_t workerCount = std::max<size_t>(workerCredits_.size(), 1);
            const size_t cursor = node < smoothNodeWorkerCursor_.size() ? smoothNodeWorkerCursor_[node] : 0;
            for (size_t workerOff = 0; workerOff < workerCount && !progress; ++workerOff) {
                const uint8_t worker = static_cast<uint8_t>((cursor + workerOff) % workerCount);
                for (size_t idx = 0; idx < pendingQ_.size(); ++idx) {
                    const PendingTransfer& first = pendingQ_[idx];
                    if (first.targetNode != node || first.workerSlot != worker) {
                        continue;
                    }
                    const uint64_t siblingId = siblingRequestId(first.requestId);
                    size_t siblingIdx = pendingQ_.size();
                    for (size_t j = idx + 1; j < pendingQ_.size(); ++j) {
                        if (pendingQ_[j].requestId == siblingId) {
                            siblingIdx = j;
                            break;
                        }
                    }
                    if (siblingIdx == pendingQ_.size()) {
                        pressureNoSiblingBlocked_++;
                        continue;
                    }

                    const PendingTransfer& second = pendingQ_[siblingIdx];
                    if (first.workerSlot != second.workerSlot || first.targetNode != second.targetNode) {
                        continue;
                    }
                    const uint32_t bytesNeed = first.bytes + second.bytes;
                    const uint32_t nodeNeed = nodeCreditUnitsForBytes(first.bytes) + nodeCreditUnitsForBytes(second.bytes);
                    if (!canIssue(first, 2, nodeNeed, bytesNeed)) {
                        continue;
                    }

                    PendingTransfer reqA = first;
                    PendingTransfer reqB = second;
                    if (!issueTransfer(reqA) || !issueTransfer(reqB)) {
                        continue;
                    }

                    const uint32_t reqAUnits = nodeCreditUnitsForBytes(reqA.bytes);
                    const uint32_t reqBUnits = nodeCreditUnitsForBytes(reqB.bytes);
                    nodeCredits_[reqA.targetNode] -= nodeNeed;
                    workerCredits_[reqA.workerSlot] -= 2;
                    issuedNodeCreditUnits_[reqA.requestId] = reqAUnits;
                    issuedNodeCreditUnits_[reqB.requestId] = reqBUnits;
                    consumeSmoothTokens(reqA, bytesNeed);
                    issuedThisTick += 2;
                    pressureIssuedRequests_ += 2;
                    pressureIssuedPairs_++;
                    pendingQ_.erase(pendingQ_.begin() + static_cast<std::ptrdiff_t>(siblingIdx));
                    pendingQ_.erase(pendingQ_.begin() + static_cast<std::ptrdiff_t>(idx));
                    if (node < smoothNodeWorkerCursor_.size()) {
                        smoothNodeWorkerCursor_[node] = (static_cast<size_t>(worker) + 1) % workerCount;
                    }
                    progress = true;
                    break;
                }
            }
        }

        if (progress || pendingQ_.empty()) {
            continue;
        }

        for (uint32_t node = 0; node < numMemoryNodes_ && !progress; ++node) {
            const size_t workerCount = std::max<size_t>(workerCredits_.size(), 1);
            const size_t cursor = node < smoothNodeWorkerCursor_.size() ? smoothNodeWorkerCursor_[node] : 0;
            for (size_t workerOff = 0; workerOff < workerCount && !progress; ++workerOff) {
                const uint8_t worker = static_cast<uint8_t>((cursor + workerOff) % workerCount);
                for (size_t idx = 0; idx < pendingQ_.size(); ++idx) {
                    PendingTransfer req = pendingQ_[idx];
                    if (req.targetNode != node || req.workerSlot != worker) {
                        continue;
                    }
                    const uint32_t nodeNeed = nodeCreditUnitsForBytes(req.bytes);
                    if (!canIssue(req, 1, nodeNeed, req.bytes) || !issueTransfer(req)) {
                        continue;
                    }
                    nodeCredits_[req.targetNode] -= nodeNeed;
                    workerCredits_[req.workerSlot]--;
                    issuedNodeCreditUnits_[req.requestId] = nodeNeed;
                    consumeSmoothTokens(req, req.bytes);
                    issuedThisTick++;
                    pressureIssuedRequests_++;
                    pendingQ_.erase(pendingQ_.begin() + static_cast<std::ptrdiff_t>(idx));
                    if (node < smoothNodeWorkerCursor_.size()) {
                        smoothNodeWorkerCursor_[node] = (static_cast<size_t>(worker) + 1) % workerCount;
                    }
                    progress = true;
                    break;
                }
            }
        }
    }

    if (issuedThisTick == 0) {
        pressureNoIssueTicks_++;
    }
}

void RequestSchedulerEndpoint::refillSmoothTokens() {
    if (!smoothStreamEnable_ || smoothReadBwBytesPerCycle_ <= 0.0 || smoothReadTokens_.empty()) {
        return;
    }
    const double cap = static_cast<double>(std::max<uint32_t>(smoothTokenBurstBytes_, 1));
    for (double& tokens : smoothReadTokens_) {
        tokens = std::min<double>(cap, tokens + smoothReadBwBytesPerCycle_);
    }
}

bool RequestSchedulerEndpoint::tick(SST::Cycle_t cycle) {
    lastTickCycle_ = cycle;
    if (role_ == RequestSchedulerRole::WORKER) {
        pollWorkerMailbox();
        flushWorkerPartialCreditReturns();
        flushWorkerDirectSubmits();
        flushWorkerPartialCreditReturns();
        flushWorkerDirectDones();
        refillWorkerWindows();
        flushWorkerDirectSubmits();
    } else {
        if (!gm_) {
            gm_ = GlobalMemoryImplement::lookupByCoreId(static_cast<int>(coreId_));
        }
        sampleManagerPressure();
        tryIssue();
    }
    return false;
}

} // namespace Golem
} // namespace SST
