#ifndef _H_GOLEM_WORKER_COMMAND_PROCESSOR
#define _H_GOLEM_WORKER_COMMAND_PROCESSOR

#include <cinttypes>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <utility>
#include <vector>

#include <sst/core/component.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/subcomponent.h>

#include <sst/elements/golem/array/computeArray.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/requestscheduler/requestscheduler.h>

namespace SST {
namespace Golem {

struct WorkerTaskListHeader {
    uint32_t worker_slot = 0;
    uint32_t task_count = 0;
    uint32_t active_worker_cores = 0;
    uint32_t total_groups = 0;
    uint32_t data_memory_node_count = 0;
    uint64_t mem_node_size = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t hw_input_size = 0;
    uint32_t hw_output_size = 0;
    uint32_t block_m = 0;
    uint32_t block_n = 0;
    uint32_t block_k = 0;
    uint32_t elem_bytes = 0;
    uint64_t mat_stride_bytes = 0;
    uint64_t vec_stride_bytes = 0;
    uint64_t off_gemm_mat_base = 0;
    uint64_t off_gemm_vec_base = 0;
    uint64_t off_gemm_out_base = 0;
    uint64_t local_mat_ping_gm_addr = 0;
    uint64_t local_mat_pong_gm_addr = 0;
    uint64_t local_mat_slot2_gm_addr = 0;
    uint64_t local_mat_slot3_gm_addr = 0;
    uint64_t local_vec_ping_gm_addr = 0;
    uint64_t local_vec_pong_gm_addr = 0;
    uint64_t local_vec_slot2_gm_addr = 0;
    uint64_t local_vec_slot3_gm_addr = 0;
    uint64_t local_mat_slot_stride_bytes = 0;
    uint64_t local_vec_slot_stride_bytes = 0;
    uint32_t local_slot_count = 2;
    uint64_t local_accum_gm_addr = 0;
    uint64_t local_out_gm_addr = 0;
    uint64_t finished_mailbox_addr = 0;
    uint32_t a_reuse_n_tiles = 1;
    uint32_t n_group_count = 0;
    uint32_t b_reuse_m_tiles = 1;
    uint32_t m_group_count = 0;
    uint32_t data_node_map_mode = 0;
};

struct WorkerWindowDescriptor {
    uint64_t task_id = 0;
    uint64_t task_flags = 0;
    uint64_t mat_base_addr = 0;
    uint64_t vec_base_addr = 0;
    uint64_t accum_base_addr = 0;
    uint64_t completion_flag_addr = 0;
    uint64_t completion_value = 0;
    uint32_t k_begin = 0;
    uint32_t k_count = 0;
    uint32_t block_n = 0;
    uint32_t hw_input_size = 0;
    uint32_t hw_output_size = 0;
    uint32_t array_input_size = 0;
    uint32_t array_output_size = 0;
    uint32_t elem_bytes = 0;
    uint64_t mat_stride_bytes = 0;
    uint64_t vec_stride_bytes = 0;
    uint64_t local_mat_gm_addr = 0;
    uint64_t local_vec_gm_addr = 0;
    uint64_t local_accum_gm_addr = 0;
    uint64_t local_out_gm_addr = 0;
    uint64_t c_base_addr = 0;
};

class WorkerCommandProcessorAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::WorkerCommandProcessorAPI)

    WorkerCommandProcessorAPI(ComponentId_t id, SST::Params& params) : SST::SubComponent(id) {}
    ~WorkerCommandProcessorAPI() override = default;

    virtual void bindResources(
        uint32_t coreId,
        SST::Output* output,
        SST::Golem::GlobalMemoryAPI* globalMem,
        SST::Golem::ComputeArray* array,
        SST::Golem::RequestSchedulerAPI* requestScheduler) = 0;

    virtual bool startWindow(const WorkerTaskListHeader& header) = 0;
    virtual bool isBusy() const = 0;
    virtual bool tick(uint64_t cycle) = 0;
    virtual bool handleArrayDone(uint32_t arrayId, uint64_t cycle) = 0;
};

class WorkerCommandProcessorLocal : public WorkerCommandProcessorAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        WorkerCommandProcessorLocal,
        "golem",
        "WorkerCommandProcessorLocal",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Minimal local worker command processor prototype",
        SST::Golem::WorkerCommandProcessorAPI)

    SST_ELI_DOCUMENT_PARAMS(
        {"verbose", "Verbosity", "0"},
        {"dtype_is_float", "Output vector stores float elements", "0"},
        {"stage3_trace", "Enable Stage3 2D window trace", "0"},
        {"prefetch_windows", "Number of 2D K-windows to prefetch ahead of the active window", "1"},
        {"window_k_tiles", "WCP K-tiles per scheduler transaction", "4"})

    WorkerCommandProcessorLocal(ComponentId_t id, SST::Params& params)
        : WorkerCommandProcessorAPI(id, params),
          verbose_(params.find<int>("verbose", 0)),
          outputIsFloat_(params.find<int>("dtype_is_float", 0) != 0),
          stage3Trace_(params.find<int>("stage3_trace", 0) != 0),
          prefetchWindowDepth_(static_cast<uint32_t>(std::max(1, params.find<int>("prefetch_windows", 1)))),
          windowKtiles_(std::max(1, params.find<int>("window_k_tiles", 4))),
          output_("WorkerCommandProcessor[@p:@l]: ", verbose_, 0, SST::Output::STDOUT) {}

    void bindResources(
        uint32_t coreId,
        SST::Output* output,
        SST::Golem::GlobalMemoryAPI* globalMem,
        SST::Golem::ComputeArray* array,
        SST::Golem::RequestSchedulerAPI* requestScheduler) override {
        coreId_ = coreId;
        extOutput_ = output;
        globalMem_ = globalMem;
        array_ = array;
        requestScheduler_ = requestScheduler;
    }

    bool startWindow(const WorkerTaskListHeader& header) override {
        if (busy_) {
            return false;
        }
        if (header.hw_input_size == 0 || header.hw_output_size == 0 ||
            header.block_k == 0 || header.block_m == 0 ||
            (header.block_k % header.hw_input_size) != 0 ||
            header.block_m != header.hw_output_size) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: minimal micro-tiling supports block_m==hw_out and block_k multiple of hw_in, got block_m=%u block_k=%u hw_out=%u hw_in=%u\n",
                    coreId_, header.block_m, header.block_k, header.hw_output_size, header.hw_input_size);
            }
            return false;
        }
        const uint32_t reuseN = std::max<uint32_t>(header.a_reuse_n_tiles, 1u);
        const uint32_t reuseM = std::max<uint32_t>(header.b_reuse_m_tiles, 1u);
        const uint32_t kTiles = header.block_k > 0 ? header.k / header.block_k : 0;
        if (reuseN > 1 && reuseM > 1 && reuseN != reuseM) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: first 2D full-K implementation requires square reuse, got reuse_n=%u reuse_m=%u\n",
                    coreId_, reuseN, reuseM);
            }
            return false;
        }
        const uint32_t slotCount = std::max<uint32_t>(header.local_slot_count, 1u);
        const uint32_t residentK = residentKTileCount(header, true);
        if (reuseM > 1 && reuseN > 1 && residentK == 0) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: 2D K-window reuse needs enough slots for active+prefetch window buffers, got k_tiles=%u local_slot_count=%u reuse_m=%u reuse_n=%u prefetch_windows=%u\n",
                    coreId_, kTiles, header.local_slot_count, reuseM, reuseN, prefetchWindowDepth_);
            }
            return false;
        }
        if (!(reuseM > 1 && reuseN > 1) && reuseN > 1 && kTiles > slotCount) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: A-reuse requires k_tiles<=local_slot_count, got k_tiles=%u local_slot_count=%u reuse_n=%u\n",
                    coreId_, kTiles, header.local_slot_count, reuseN);
            }
            return false;
        }
        if (!(reuseM > 1 && reuseN > 1) && reuseM > 1 && kTiles > slotCount) {
            if (extOutput_ != nullptr) {
                extOutput_->output(
                    "[Core %u] [wcp] ERROR: B-reuse requires k_tiles<=local_slot_count, got k_tiles=%u local_slot_count=%u reuse_m=%u\n",
                    coreId_, kTiles, header.local_slot_count, reuseM);
            }
            return false;
        }
        header_ = header;
        busy_ = true;
        taskIndex_ = 0;
        reuseNIndex_ = 0;
        reuseMIndex_ = 0;
        deriveTask(taskIndex_);
        if (extOutput_ != nullptr) {
            extOutput_->verbose(CALL_INFO, 1, 0,
                "[WCP] core=%u start worker_slot=%u tasks=%u block_n=%u\n",
                coreId_, header_.worker_slot, header_.task_count, header_.block_n);
        }
        lastAccountCycle_ = 0;
        workerStartCycle_ = 0;
        workerEndCycle_ = 0;
        totalWindowCycles_ = 0;
        computeCycles_ = 0;
        tileReadyWaitCycles_ = 0;
        txnWaitCycles_ = 0;
        writebackWaitCycles_ = 0;
        wait2DActivateCycles_ = 0;
        wait2DActiveNotReadyCycles_ = 0;
        waitNon2DTxnCycles_ = 0;
        waitNoActiveTxnCycles_ = 0;
        windowSubmitActiveCount_ = 0;
        windowSubmitPrefetchCount_ = 0;
        windowActivateCount_ = 0;
        windowAdvanceWaitPrefetchCount_ = 0;
        pendingWritebackTokens_.clear();
        lastStage3TraceCycle_ = 0;
        tileComputeStartCycles_.clear();
        tileComputeDoneCycles_.clear();
        tileRetireCycles_.clear();
        tileComputeStartSchedCycles_.clear();
        tileComputeDoneSchedCycles_.clear();
        tileRetireSchedCycles_.clear();
        resetPipelineState();
        phase_ = Phase::RUN;
        return true;
    }

    bool isBusy() const override { return busy_; }

    bool tick(uint64_t cycle) override {
        if (!busy_) {
            return false;
        }
        if (workerStartCycle_ == 0) {
            workerStartCycle_ = cycle;
        }

        accountCycles(cycle);

        switch (phase_) {
        case Phase::RUN:
            tryIssuePrefetches();
            if (!computeInFlight_) {
                int tile = activeComputeTileIndex_;
                if (tile < 0) {
                    tile = selectNextTile();
                }
                if (tile >= 0) {
                    if (activeComputeTileIndex_ < 0) {
                        activeComputeTileIndex_ = tile;
                        activeComputeSlotIndex_ = static_cast<int>(
                            use2DWindowEngine()
                                ? groupMatSlotFor(static_cast<uint32_t>(tile))
                                : (static_cast<uint32_t>(tile) % std::max<uint32_t>(header_.local_slot_count, 1u)));
                        activeMicroKStep_ = 0;
                        if (activeComputeSlotIndex_ >= 0 && activeComputeSlotIndex_ < static_cast<int>(buffers_.size())) {
                            buffers_[activeComputeSlotIndex_].in_use = true;
                        }
                        if (!buildActiveTileMicroOps(static_cast<uint32_t>(activeComputeTileIndex_), static_cast<uint32_t>(activeComputeSlotIndex_))) {
                            phase_ = Phase::DONE;
                            break;
                        }
                    }
                    updateActiveTileInputReadiness();
                    if (!activeTilePayloadLoaded_ && !activeComputeReadyQueue_.empty()) {
                        if (!loadTilePayload(static_cast<uint32_t>(tile))) {
                            phase_ = Phase::DONE;
                            break;
                        }
                        activeTilePayloadLoaded_ = true;
                    }
                    if (activeTilePayloadLoaded_ && !activeComputeReadyQueue_.empty()) {
                        if (!issueActiveMicroTile()) {
                            phase_ = Phase::DONE;
                            break;
                        }
                        pendingArrays_ = current_.block_n;
                        computeInFlight_ = true;
                        if (static_cast<size_t>(tile) < tileComputeStartCycles_.size()) {
                            if (tileComputeStartCycles_[static_cast<size_t>(tile)] == 0) {
                                tileComputeStartCycles_[static_cast<size_t>(tile)] = cycle;
                                if (static_cast<size_t>(tile) < tileComputeStartSchedCycles_.size()) {
                                    tileComputeStartSchedCycles_[static_cast<size_t>(tile)] = schedulerTimelineCycle();
                                }
                            }
                        }
                    }
                } else if (allTilesScheduled_ && !computeInFlight_ && activeComputeTileIndex_ < 0 &&
                           activeTxnRetiredTileCount_ >= activeTxnTileCount_) {
                    traceStage3("RUN_TO_WRITEBACK", cycle);
                    phase_ = Phase::WRITEBACK;
                } else if (activeTxnId_ != 0 && activeTxnRetiredTileCount_ >= activeTxnTileCount_ &&
                           (!use2DWindowEngine() || allReuseDoneForWindow()) &&
                           requestScheduler_ != nullptr && activeTransactionsDone()) {
                    traceStage3("RETIRE_ACTIVE_TXN", cycle);
                    retireActiveTransactions();
                    activeTxnId_ = 0;
                    active2DTxnIds_.clear();
                    active2DSchedulerTileRetired_.clear();
                    activeTxnTileCount_ = 0;
                    nextTxnComputeTile_ = 0;
                } else {
                    traceStage3Periodic("RUN_WAIT", cycle);
                }
            }
            break;
        case Phase::WRITEBACK:
            {
                if (use2DWindowEngine() && !isFinal2DWindow()) {
                    traceStage3("SAVE_PARTIAL", cycle);
                    if (!savePartialCFromArray()) {
                        return false;
                    }
                    markCurrentReuseDone();
                    advanceAfterWriteback(cycle);
                    break;
                }
                traceStage3("ISSUE_WRITEBACK", cycle);
                std::vector<uint8_t> tile;
                if (!captureArrayOutput(tile)) {
                    return false;
                }
                writebackDone_ = false;
                writebackToken_ = issueDmaWrite(
                    current_.c_base_addr,
                    tile.size(),
                    tile);
                // Final C writeback is intentionally fire-and-forget. The SST
                // network/memory events still drain before simulation exit;
                // blocking the worker here serializes completion on write acks.
            }
            writebackDone_ = true;
            writebackToken_ = 0;
            if (use2DWindowEngine()) {
                markCurrentReuseDone();
            }
            advanceAfterWriteback(cycle);
            break;
        case Phase::WRITEBACK_WAIT:
            if (!drainPendingWritebacks()) {
                traceStage3Periodic("WRITEBACK_WAIT", cycle);
                break;
            }
            phase_ = Phase::DONE;
            break;
        case Phase::DONE:
            workerEndCycle_ = cycle;
            if (extOutput_ != nullptr) {
                const uint64_t dma_time = tileReadyWaitCycles_ + txnWaitCycles_ + writebackWaitCycles_;
                extOutput_->output(
                    "[Core %u] [wcp] LATENCY(cycles): dma_issue=0 dma_wait=%" PRIu64
                    " dma_total=%" PRIu64 " compute=%" PRIu64
                    " compute_submit=0 compute_wait=%" PRIu64
                    " sched_protocol=0 c_store=%" PRIu64
                    " tile_ready_wait=%" PRIu64 " txn_wait=%" PRIu64
                    " writeback_wait=%" PRIu64
                    " wait_2d_activate=%" PRIu64 " wait_2d_active_not_ready=%" PRIu64
                    " wait_non2d_txn=%" PRIu64 " wait_no_active_txn=%" PRIu64
                    " window_submit_active=%" PRIu64 " window_submit_prefetch=%" PRIu64
                    " window_activate=%" PRIu64 " window_advance_wait_prefetch=%" PRIu64
                    " group_wait=0 poll_iters=0 overlap_issue=0 overlap_wait=0"
                    " issue_block_q=0 issue_write=0 ov_issue_block_q=0 ov_issue_write=0"
                    " task_desc=0 nloop=0 submit_pack=0 finish_publish=0 total=%" PRIu64
                    " start_cycle=%" PRIu64 " end_cycle=%" PRIu64 "\n",
                    coreId_, dma_time, dma_time, computeCycles_, computeCycles_,
                    writebackWaitCycles_, tileReadyWaitCycles_, txnWaitCycles_,
                    writebackWaitCycles_, wait2DActivateCycles_, wait2DActiveNotReadyCycles_,
                    waitNon2DTxnCycles_, waitNoActiveTxnCycles_, windowSubmitActiveCount_,
                    windowSubmitPrefetchCount_, windowActivateCount_, windowAdvanceWaitPrefetchCount_,
                    totalWindowCycles_, workerStartCycle_, workerEndCycle_);
            }
            if (header_.finished_mailbox_addr != 0) {
                std::vector<uint8_t> one(sizeof(uint64_t), 0);
                const uint64_t val = 1;
                std::memcpy(one.data(), &val, sizeof(uint64_t));
                globalMem_->wr_to_globalmem(header_.finished_mailbox_addr, one.size(), one);
            }
            busy_ = false;
            phase_ = Phase::IDLE;
            break;
        case Phase::IDLE:
        default:
            break;
        }
        return busy_;
    }

    bool handleArrayDone(uint32_t arrayId, uint64_t) override {
        if (!busy_ || !computeInFlight_ || phase_ != Phase::RUN) {
            return false;
        }
        if (arrayId >= current_.block_n) {
            return false;
        }
        if (pendingArrays_ > 0) {
            pendingArrays_ -= 1;
        }
        if (pendingArrays_ == 0 && activeComputeTileIndex_ >= 0) {
            if (!completeActiveMicroTile()) {
                return false;
            }
            computeInFlight_ = false;
        }
        return true;
    }

private:
    struct BufferSlot {
        uint64_t mat_addr = 0;
        uint64_t vec_addr = 0;
        uint32_t k_tile = 0;
        bool mat_done = false;
        bool vec_done = false;
        bool ready = false;
        bool inflight = false;
        bool in_use = false;
        uint64_t mat_token = 0;
        uint64_t vec_token = 0;
    };

    struct MicroOp {
        uint64_t taskId = 0;
        uint32_t logicalTileIdx = 0;
        uint32_t mStep = 0;
        uint32_t nGroup = 0;
        uint32_t kStep = 0;
        uint32_t slotIdx = 0;
    };

    struct KStepScoreboard {
        std::vector<uint8_t> matReady;
        std::vector<uint8_t> vecReady;
        int32_t lastCompletedKStep = -1;
        bool valid = false;
    };

    struct Prefetch2DWindow {
        uint64_t txnId = 0;
        std::vector<uint64_t> txnIds;
        uint32_t kBegin = 0;
        uint32_t kCount = 0;
        uint32_t buffer = 0;
    };

    void resetPipelineState() {
        const uint32_t slotCount = std::max<uint32_t>(header_.local_slot_count, 1u);
        buffers_.assign(slotCount, BufferSlot{});
        for (uint32_t slot = 0; slot < slotCount; ++slot) {
            buffers_[slot] = BufferSlot{
                header_.local_mat_ping_gm_addr + static_cast<uint64_t>(slot) * header_.local_mat_slot_stride_bytes,
                header_.local_vec_ping_gm_addr + static_cast<uint64_t>(slot) * header_.local_vec_slot_stride_bytes,
            };
        }
        nextPrefetchK_ = current_.k_begin;
        allTilesScheduled_ = false;
        computeInFlight_ = false;
        activeComputeTileIndex_ = -1;
        activeComputeSlotIndex_ = -1;
        pendingArrays_ = 0;
        activeTxnId_ = 0;
        activeTxnTileCount_ = 0;
        nextTxnComputeTile_ = 0;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.clear();
        active2DSchedulerTileRetired_.clear();
        windowReuseDone_.clear();
        nextReadyScanCursor_ = 0;
        activeTxnKBegin_ = 0;
        activeMicroKStep_ = 0;
        taskAccumInitialized_ = false;
        activeMatPayload_.clear();
        activeVecPayload_.clear();
        activeTileMicroOps_.clear();
        activeTileMicroOpCursor_ = 0;
        activeTileScoreboard_ = KStepScoreboard{};
        activeComputeReadyQueue_.clear();
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTilePayloadLoaded_ = false;
        if (use2DWindowEngine()) {
            totalKTileCount_ = header_.block_k > 0 ? header_.k / header_.block_k : current_.k_count;
            residentKTileCount_ = std::min<uint32_t>(activeResidentKTileCount(), std::max<uint32_t>(totalKTileCount_, 1u));
            activeWindowKBegin_ = 0;
            activeWindowKCount_ = std::min<uint32_t>(residentKTileCount_, totalKTileCount_);
            activeWindowBuffer_ = 0;
            activeWindowValid_ = false;
            next2DPrefetchK_ = activeWindowKCount_;
            active2DTxnIds_.clear();
            prefetch2DWindows_.clear();
            current_.k_begin = activeWindowKBegin_;
            current_.k_count = activeWindowKCount_;
            const size_t partialCount = static_cast<size_t>(std::max<uint32_t>(header_.b_reuse_m_tiles, 1u)) *
                                        static_cast<size_t>(std::max<uint32_t>(header_.a_reuse_n_tiles, 1u));
            const size_t partialBytes = static_cast<size_t>(header_.block_m) * static_cast<size_t>(header_.block_n) *
                                        static_cast<size_t>(header_.elem_bytes);
            partialCTiles_.assign(partialCount, std::vector<uint8_t>(partialBytes, 0));
            partialValid_.assign(partialCount, 0);
            windowReuseDone_.assign(partialCount, 0);
        } else {
            totalKTileCount_ = current_.k_count;
            residentKTileCount_ = current_.k_count;
            activeWindowKBegin_ = 0;
            activeWindowKCount_ = current_.k_count;
            activeWindowBuffer_ = 0;
            activeWindowValid_ = false;
            next2DPrefetchK_ = 0;
            active2DTxnIds_.clear();
            prefetch2DWindows_.clear();
            partialCTiles_.clear();
            partialValid_.clear();
            windowReuseDone_.clear();
        }
    }

    bool noBuffersBusy() const {
        return !computeInFlight_ && activeTxnId_ == 0;
    }

    bool is2DReuse() const {
        return std::max<uint32_t>(header_.a_reuse_n_tiles, 1u) > 1 &&
               std::max<uint32_t>(header_.b_reuse_m_tiles, 1u) > 1;
    }

    uint32_t residentKTileCount(const WorkerTaskListHeader& header, bool pingPong) const {
        const uint32_t reuseN = std::max<uint32_t>(header.a_reuse_n_tiles, 1u);
        const uint32_t reuseM = std::max<uint32_t>(header.b_reuse_m_tiles, 1u);
        const uint32_t slotCount = std::max<uint32_t>(header.local_slot_count, 1u);
        const uint32_t buffers = pingPong ? twoDWindowBufferCount() : 1u;
        const uint32_t matLimit = slotCount / (buffers * reuseM);
        const uint32_t vecLimit = slotCount / (buffers * reuseN);
        const uint32_t requestedWindowK = std::max<uint32_t>(windowKtiles_, 1u);
        return std::min<uint32_t>(requestedWindowK, std::min<uint32_t>(matLimit, vecLimit));
    }

    uint32_t twoDWindowBufferCount() const {
        return std::max<uint32_t>(prefetchWindowDepth_ + 1u, 2u);
    }

    bool allocatePrefetchWindowBuffer(uint32_t& buffer) const {
        const uint32_t bufferCount = twoDWindowBufferCount();
        for (uint32_t off = 1; off < bufferCount; ++off) {
            const uint32_t candidate = (activeWindowBuffer_ + off) % bufferCount;
            bool inUse = (candidate == activeWindowBuffer_);
            for (const auto& window : prefetch2DWindows_) {
                if (window.buffer == candidate) {
                    inUse = true;
                    break;
                }
            }
            if (!inUse) {
                buffer = candidate;
                return true;
            }
        }
        return false;
    }

    uint32_t activeResidentKTileCount() const {
        if (!is2DReuse()) {
            return current_.k_count;
        }
        uint32_t resident = residentKTileCount(header_, true);
        if (resident == 0) {
            resident = residentKTileCount(header_, false);
        }
        return std::max<uint32_t>(resident, 1u);
    }

    bool use2DWindowEngine() const {
        return is2DReuse();
    }

    void traceStage3(const char* event, uint64_t cycle) {
        if (!stage3Trace_ || extOutput_ == nullptr || !use2DWindowEngine()) {
            return;
        }
        const bool activeReady = !active2DTxnIds_.empty() &&
                                 are2DTransactionsReady(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), false);
        const Prefetch2DWindow* frontPrefetch = prefetch2DWindows_.empty() ? nullptr : &prefetch2DWindows_.front();
        const bool prefetchReady = frontPrefetch != nullptr &&
                                  are2DTransactionsReady(frontPrefetch->txnIds, twoDWindowTransactionTileCount(frontPrefetch->kCount), false);
        extOutput_->output(
            "[Core %u] [wcp] STAGE3_TRACE event=%s cycle=%" PRIu64
            " task_idx=%u task=%" PRIu64 " macro=%u reuseM=%u/%u reuseN=%u/%u"
            " win_valid=%u win_k=%u+%u win_buf=%u totalK=%u residentK=%u"
            " active_txn=%" PRIu64 " active_ids=%zu active_ready=%u retired=%u/%u"
            " prefetch_txn=%" PRIu64 " prefetch_ids=%zu prefetch_ready=%u prefetch_k=%u+%u prefetch_buf=%u prefetch_queue=%zu/%u next_prefetch=%u"
            " all_sched=%u compute=%u active_tile=%d slot=%d buffers_busy=%u partial_idx=%zu final_win=%u\n",
            coreId_, event, cycle,
            taskIndex_, current_.task_id, currentMacroTaskId_,
            reuseMIndex_, currentReuseMCount_, reuseNIndex_, currentReuseNCount_,
            activeWindowValid_ ? 1u : 0u, activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_,
            totalKTileCount_, residentKTileCount_,
            activeTxnId_, active2DTxnIds_.size(), activeReady ? 1u : 0u,
            activeTxnRetiredTileCount_, activeTxnTileCount_,
            frontPrefetch != nullptr ? frontPrefetch->txnId : 0,
            frontPrefetch != nullptr ? frontPrefetch->txnIds.size() : 0,
            prefetchReady ? 1u : 0u,
            frontPrefetch != nullptr ? frontPrefetch->kBegin : 0,
            frontPrefetch != nullptr ? frontPrefetch->kCount : 0,
            frontPrefetch != nullptr ? frontPrefetch->buffer : 0,
            prefetch2DWindows_.size(), prefetchWindowDepth_, next2DPrefetchK_,
            allTilesScheduled_ ? 1u : 0u, computeInFlight_ ? 1u : 0u,
            activeComputeTileIndex_, activeComputeSlotIndex_, noBuffersBusy() ? 0u : 1u,
            currentPartialIndex(), isFinal2DWindow() ? 1u : 0u);
    }

    void traceStage3Periodic(const char* event, uint64_t cycle) {
        if (!use2DWindowEngine()) {
            return;
        }
        if (cycle < lastStage3TraceCycle_ + 100000) {
            return;
        }
        lastStage3TraceCycle_ = cycle;
        traceStage3(event, cycle);
    }

    uint32_t groupMatSlotFor(uint32_t kTile) const {
        const uint32_t kTiles = std::max<uint32_t>(current_.k_count, 1u);
        if (use2DWindowEngine()) {
            const uint32_t windowKCapacity = std::max<uint32_t>(residentKTileCount_, 1u);
            const uint32_t perBufferSlots = windowKCapacity * std::max<uint32_t>(header_.b_reuse_m_tiles, 1u);
            return activeWindowBuffer_ * perBufferSlots + reuseMIndex_ * std::max<uint32_t>(activeWindowKCount_, 1u) + kTile;
        }
        return is2DReuse() ? (reuseMIndex_ * kTiles + kTile) : kTile;
    }

    uint32_t groupVecSlotFor(uint32_t kTile) const {
        const uint32_t kTiles = std::max<uint32_t>(current_.k_count, 1u);
        if (use2DWindowEngine()) {
            const uint32_t windowKCapacity = std::max<uint32_t>(residentKTileCount_, 1u);
            const uint32_t perBufferSlots = windowKCapacity * std::max<uint32_t>(header_.a_reuse_n_tiles, 1u);
            return activeWindowBuffer_ * perBufferSlots + reuseNIndex_ * std::max<uint32_t>(activeWindowKCount_, 1u) + kTile;
        }
        return is2DReuse() ? (reuseNIndex_ * kTiles + kTile) : kTile;
    }

    size_t reuseIndex(uint32_t reuseM, uint32_t reuseN) const {
        return static_cast<size_t>(reuseM) * static_cast<size_t>(currentReuseNCount_) + static_cast<size_t>(reuseN);
    }

    void markCurrentReuseDone() {
        if (!use2DWindowEngine()) {
            return;
        }
        const size_t idx = reuseIndex(reuseMIndex_, reuseNIndex_);
        const size_t count = static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_);
        if (windowReuseDone_.size() != count) {
            windowReuseDone_.assign(count, 0);
        }
        if (idx < windowReuseDone_.size()) {
            windowReuseDone_[idx] = 1;
        }
    }

    bool allReuseDoneForWindow() const {
        if (!use2DWindowEngine()) {
            return true;
        }
        const size_t count = static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_);
        if (count == 0 || windowReuseDone_.size() != count) {
            return false;
        }
        for (uint8_t done : windowReuseDone_) {
            if (done == 0) {
                return false;
            }
        }
        return true;
    }

    bool buildActiveTileMicroOps(uint32_t localTileIdx, uint32_t slotIdx) {
        const uint32_t kSteps = microKStepCount();
        if (kSteps == 0) {
            return false;
        }

        activeTileMicroOps_.clear();
        activeTileMicroOps_.reserve(kSteps);
        for (uint32_t kStep = 0; kStep < kSteps; ++kStep) {
            MicroOp op{};
            op.taskId = current_.task_id;
            op.logicalTileIdx = localTileIdx;
            op.mStep = 0;
            op.nGroup = 0;
            op.kStep = kStep;
            op.slotIdx = slotIdx;
            activeTileMicroOps_.push_back(op);
        }

        activeTileMicroOpCursor_ = 0;
        activeTileScoreboard_.matReady.assign(kSteps, 0);
        activeTileScoreboard_.vecReady.assign(kSteps, 0);
        activeTileScoreboard_.lastCompletedKStep = -1;
        activeTileScoreboard_.valid = true;
        activeComputeReadyQueue_.clear();
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTilePayloadLoaded_ = false;
        refreshActiveComputeReadyQueue();
        return true;
    }

    bool isMicroOpReady(const MicroOp& op) const {
        if (!activeTileScoreboard_.valid) {
            return false;
        }
        if (op.kStep >= activeTileScoreboard_.matReady.size() ||
            op.kStep >= activeTileScoreboard_.vecReady.size()) {
            return false;
        }
        if (activeTileScoreboard_.matReady[op.kStep] == 0 ||
            activeTileScoreboard_.vecReady[op.kStep] == 0) {
            return false;
        }

        const int32_t expectedNextK = activeTileScoreboard_.lastCompletedKStep + 1;
        if (static_cast<int32_t>(op.kStep) != expectedNextK) {
            return false;
        }
        return true;
    }

    void refreshActiveComputeReadyQueue() {
        activeComputeReadyQueue_.clear();
        if (!activeTileScoreboard_.valid || activeMicroOpIssued_) {
            return;
        }
        for (size_t idx = activeTileMicroOpCursor_; idx < activeTileMicroOps_.size(); ++idx) {
            const MicroOp& op = activeTileMicroOps_[idx];
            if (isMicroOpReady(op)) {
                activeComputeReadyQueue_.push_back(op);
                break;
            }
        }
    }

    void updateActiveTileInputReadiness() {
        if (!activeTileScoreboard_.valid || activeComputeTileIndex_ < 0) {
            return;
        }
        bool changed = false;
        for (size_t i = 0; i < activeTileScoreboard_.matReady.size(); ++i) {
            bool matReady = true;
            bool vecReady = true;
            if (use2DWindowEngine() && activeWindowValid_) {
                const bool ready = is2DComputeTileReady(static_cast<uint32_t>(activeComputeTileIndex_));
                matReady = ready;
                vecReady = ready;
            } else if (is2DReuse() && groupResidentValid_ && residentMacroTaskId_ == currentMacroTaskId_) {
                matReady = true;
                vecReady = true;
            } else if (requestScheduler_ != nullptr && activeTxnId_ != 0) {
                matReady = false;
                vecReady = false;
                const uint32_t tileIdx = static_cast<uint32_t>(activeComputeTileIndex_);
                if (!requestScheduler_->getTileKStepReadiness(
                        activeTxnId_, tileIdx, static_cast<uint32_t>(i), matReady, vecReady)) {
                    if (requestScheduler_->isTileReady(activeTxnId_, tileIdx)) {
                        matReady = true;
                        vecReady = true;
                    } else {
                        WcpTileTimelineDebug dbg{};
                        if (requestScheduler_->getTileTimeline(activeTxnId_, tileIdx, dbg)) {
                            matReady = (dbg.matDoneCycle != 0);
                            vecReady = (dbg.vecDoneCycle != 0);
                        }
                    }
                }
            }
            const uint8_t matVal = matReady ? 1 : 0;
            const uint8_t vecVal = vecReady ? 1 : 0;
            if (activeTileScoreboard_.matReady[i] != matVal) {
                activeTileScoreboard_.matReady[i] = matVal;
                changed = true;
            }
            if (activeTileScoreboard_.vecReady[i] != vecVal) {
                activeTileScoreboard_.vecReady[i] = vecVal;
                changed = true;
            }
        }
        if (changed) {
            refreshActiveComputeReadyQueue();
        }
    }

    void accountCycles(uint64_t cycle) {
        if (!busy_) {
            lastAccountCycle_ = cycle;
            return;
        }
        if (lastAccountCycle_ == 0) {
            lastAccountCycle_ = cycle;
        }
        if (cycle < lastAccountCycle_) {
            return;
        }
        const uint64_t delta = cycle - lastAccountCycle_ + 1;
        totalWindowCycles_ += delta;
        if (phase_ == Phase::RUN) {
            if (computeInFlight_) {
                computeCycles_ += delta;
            } else if (activeTxnId_ != 0) {
                if (activeTxnRetiredTileCount_ < activeTxnTileCount_) {
                    tileReadyWaitCycles_ += delta;
                    if (use2DWindowEngine()) {
                        if (!activeWindowValid_) {
                            wait2DActivateCycles_ += delta;
                        } else {
                            wait2DActiveNotReadyCycles_ += delta;
                        }
                    } else {
                        waitNon2DTxnCycles_ += delta;
                    }
                } else {
                    txnWaitCycles_ += delta;
                }
            } else {
                waitNoActiveTxnCycles_ += delta;
            }
        } else if (phase_ == Phase::WRITEBACK || phase_ == Phase::WRITEBACK_WAIT) {
            writebackWaitCycles_ += delta;
        }
        lastAccountCycle_ = cycle + 1;
    }

    uint64_t issueDmaRead(uint64_t src_pa, size_t length, uint64_t gm_dst_addr) {
        if (auto* gm = dynamic_cast<GlobalMemoryImplement*>(globalMem_)) {
            return gm->dma_read_from_host_to_globalmem_async(src_pa, length, gm_dst_addr);
        }
        if (auto* gm = dynamic_cast<GlobalMemoryLocal*>(globalMem_)) {
            return gm->dma_read_from_host_to_globalmem_async(src_pa, length, gm_dst_addr);
        }
        return 0;
    }

    uint64_t issueDmaWrite(uint64_t dst_pa, size_t length, const std::vector<uint8_t>& data) {
        if (auto* gm = dynamic_cast<GlobalMemoryImplement*>(globalMem_)) {
            return gm->dma_write_to_host_async(dst_pa, length, data);
        }
        if (auto* gm = dynamic_cast<GlobalMemoryLocal*>(globalMem_)) {
            return gm->dma_write_to_host_async(dst_pa, length, data);
        }
        return 0;
    }

    bool dmaDone(uint64_t token) const {
        if (token == 0) {
            return true;
        }
        if (auto* gm = dynamic_cast<GlobalMemoryImplement*>(globalMem_)) {
            return gm->dma_completion_done(token);
        }
        if (auto* gm = dynamic_cast<GlobalMemoryLocal*>(globalMem_)) {
            return gm->dma_completion_done(token);
        }
        return false;
    }

    void retireDma(uint64_t token) {
        if (token == 0) {
            return;
        }
        if (auto* gm = dynamic_cast<GlobalMemoryImplement*>(globalMem_)) {
            gm->dma_completion_retire(token);
            return;
        }
        if (auto* gm = dynamic_cast<GlobalMemoryLocal*>(globalMem_)) {
            gm->dma_completion_retire(token);
        }
    }

    bool drainPendingWritebacks() {
        bool allDone = true;
        std::deque<uint64_t> remaining;
        while (!pendingWritebackTokens_.empty()) {
            const uint64_t token = pendingWritebackTokens_.front();
            pendingWritebackTokens_.pop_front();
            if (dmaDone(token)) {
                retireDma(token);
            } else {
                remaining.push_back(token);
                allDone = false;
            }
        }
        pendingWritebackTokens_.swap(remaining);
        return allDone;
    }

    void finishOrDrainWritebacks() {
        if (drainPendingWritebacks()) {
            phase_ = Phase::DONE;
        } else {
            phase_ = Phase::WRITEBACK_WAIT;
        }
    }

    void advanceAfterWriteback(uint64_t cycle) {
        if (use2DWindowEngine()) {
            traceStage3("ADVANCE_CALL", cycle);
            if (advance2DWindowEngine()) {
                traceStage3("ADVANCE_CONTINUE", cycle);
                phase_ = Phase::RUN;
                return;
            }
            traceStage3("ADVANCE_TASK_DONE", cycle);
            if ((taskIndex_ + 1) < header_.task_count) {
                taskIndex_ += 1;
                reuseNIndex_ = 0;
                reuseMIndex_ = 0;
                deriveTask(taskIndex_);
                phase_ = Phase::RUN;
                return;
            }
            finishOrDrainWritebacks();
            return;
        }
        if ((reuseNIndex_ + 1) < currentReuseNCount_) {
            reuseNIndex_ += 1;
            deriveTask(taskIndex_);
            phase_ = Phase::RUN;
            return;
        }
        if ((reuseMIndex_ + 1) < currentReuseMCount_) {
            reuseMIndex_ += 1;
            reuseNIndex_ = 0;
            deriveTask(taskIndex_);
            phase_ = Phase::RUN;
            return;
        }
        if ((taskIndex_ + 1) < header_.task_count) {
            taskIndex_ += 1;
            reuseNIndex_ = 0;
            reuseMIndex_ = 0;
            deriveTask(taskIndex_);
            phase_ = Phase::RUN;
            return;
        }
        finishOrDrainWritebacks();
    }

    void tryIssuePrefetches() {
        if (requestScheduler_ == nullptr) {
            return;
        }
        if (use2DWindowEngine()) {
            tryIssue2DWindowPrefetches();
            return;
        }
        if (activeTxnId_ != 0) {
            return;
        }
        const uint32_t k_end = current_.k_begin + current_.k_count;
        if (is2DReuse() && groupResidentValid_ && residentMacroTaskId_ == currentMacroTaskId_) {
            allTilesScheduled_ = true;
            if (activeTxnTileCount_ == 0) {
                activeTxnTileCount_ = current_.k_count;
                activeTxnTileRetired_.assign(current_.k_count, 0);
                tileComputeStartCycles_.assign(current_.k_count, 0);
                tileComputeDoneCycles_.assign(current_.k_count, 0);
                tileRetireCycles_.assign(current_.k_count, 0);
                tileComputeStartSchedCycles_.assign(current_.k_count, 0);
                tileComputeDoneSchedCycles_.assign(current_.k_count, 0);
                tileRetireSchedCycles_.assign(current_.k_count, 0);
            }
            return;
        }
        if (nextPrefetchK_ >= k_end) {
            allTilesScheduled_ = true;
            return;
        }
        const uint32_t tiles = is2DReuse()
                                   ? (std::max<uint32_t>(header_.b_reuse_m_tiles, 1u) * current_.k_count)
                                   : std::min<uint32_t>(windowKtiles_, k_end - nextPrefetchK_);
        WcpWindowTransaction txn{};
        txn.workerSlot = header_.worker_slot;
        txn.taskId = static_cast<uint32_t>(current_.task_id);
        txn.windowId = currentWindowId_++;
        txn.kBegin = nextPrefetchK_;
        txn.kTiles = tiles;
        txn.blockN = current_.block_n;
        txn.blockK = header_.block_k;
        txn.elemBytes = current_.elem_bytes;
        txn.hwInputSize = current_.array_input_size;
        txn.memNodeSize = header_.mem_node_size;
        txn.matBaseAddr = current_.mat_base_addr + static_cast<uint64_t>(nextPrefetchK_) * current_.mat_stride_bytes;
        txn.vecBaseAddr = current_.vec_base_addr + static_cast<uint64_t>(nextPrefetchK_) * current_.vec_stride_bytes;
        txn.localMatBaseAddr = header_.local_mat_ping_gm_addr;
        txn.localVecBaseAddr = header_.local_vec_ping_gm_addr;
        txn.localMatSlotStrideBytes = header_.local_mat_slot_stride_bytes;
        txn.localVecSlotStrideBytes = header_.local_vec_slot_stride_bytes;
        txn.slotCount = std::max<uint32_t>(header_.local_slot_count, 1u);
        txn.matStrideBytes = current_.mat_stride_bytes;
        txn.vecStrideBytes = current_.vec_stride_bytes;
        txn.skipMatRead = !is2DReuse() && (std::max<uint32_t>(header_.a_reuse_n_tiles, 1u) > 1 && reuseNIndex_ > 0);
        txn.skipVecRead = !is2DReuse() && (std::max<uint32_t>(header_.b_reuse_m_tiles, 1u) > 1 && reuseMIndex_ > 0);
        activeTxnId_ = requestScheduler_->submitWindowTransaction(txn);
        activeTxnTileCount_ = tiles;
        nextTxnComputeTile_ = 0;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.assign(tiles, 0);
        nextReadyScanCursor_ = 0;
        activeTxnKBegin_ = nextPrefetchK_;
        tileComputeStartCycles_.assign(tiles, 0);
        tileComputeDoneCycles_.assign(tiles, 0);
        tileRetireCycles_.assign(tiles, 0);
        tileComputeStartSchedCycles_.assign(tiles, 0);
        tileComputeDoneSchedCycles_.assign(tiles, 0);
        tileRetireSchedCycles_.assign(tiles, 0);
        nextPrefetchK_ += is2DReuse() ? current_.k_count : tiles;
        allTilesScheduled_ = (nextPrefetchK_ >= k_end);
    }

    std::vector<uint64_t> submit2DWindowTransactions(uint32_t kBegin, uint32_t kCount, uint32_t buffer) {
        std::vector<uint64_t> txnIds;
        if (requestScheduler_ == nullptr || kCount == 0) {
            return txnIds;
        }
        const uint32_t reuseM = std::max<uint32_t>(header_.b_reuse_m_tiles, 1u);
        const uint32_t reuseN = std::max<uint32_t>(header_.a_reuse_n_tiles, 1u);
        const uint32_t windowKCapacity = std::max<uint32_t>(residentKTileCount_, kCount);
        const uint32_t perBufferMatSlots = reuseM * windowKCapacity;
        const uint32_t perBufferVecSlots = reuseN * windowKCapacity;
        const uint64_t matGroupBase = current_.mat_base_addr - static_cast<uint64_t>(reuseMIndex_) * static_cast<uint64_t>(totalKTileCount_) * current_.mat_stride_bytes;
        const uint64_t vecGroupBase = current_.vec_base_addr - static_cast<uint64_t>(reuseNIndex_) * static_cast<uint64_t>(totalKTileCount_) * current_.vec_stride_bytes;
        const uint64_t localMatBufferBase = header_.local_mat_ping_gm_addr + static_cast<uint64_t>(buffer) * perBufferMatSlots * header_.local_mat_slot_stride_bytes;
        const uint64_t localVecBufferBase = header_.local_vec_ping_gm_addr + static_cast<uint64_t>(buffer) * perBufferVecSlots * header_.local_vec_slot_stride_bytes;

        WcpWindowTransaction txn{};
        txn.workerSlot = header_.worker_slot;
        txn.taskId = static_cast<uint32_t>(current_.task_id);
        txn.windowId = currentWindowId_++;
        txn.kBegin = kBegin;
        txn.kTiles = twoDWindowTransactionTileCount(kCount);
        txn.blockN = current_.block_n;
        txn.blockK = header_.block_k;
        txn.elemBytes = current_.elem_bytes;
        txn.hwInputSize = current_.array_input_size;
        txn.memNodeSize = header_.mem_node_size;
        txn.matBaseAddr = matGroupBase + static_cast<uint64_t>(kBegin) * current_.mat_stride_bytes;
        txn.vecBaseAddr = vecGroupBase + static_cast<uint64_t>(kBegin) * current_.vec_stride_bytes;
        txn.localMatBaseAddr = localMatBufferBase;
        txn.localVecBaseAddr = localVecBufferBase;
        txn.localMatSlotStrideBytes = header_.local_mat_slot_stride_bytes;
        txn.localVecSlotStrideBytes = header_.local_vec_slot_stride_bytes;
        txn.slotCount = std::max<uint32_t>(txn.kTiles, 1u);
        txn.matStrideBytes = current_.mat_stride_bytes;
        txn.vecStrideBytes = current_.vec_stride_bytes;
        txn.useIndependentMatVecTiles = true;
        txn.matTileCount = currentReuseMCount_ * kCount;
        txn.vecTileCount = currentReuseNCount_ * kCount;
        txn.kWindowTiles = kCount;
        txn.totalKTileCount = totalKTileCount_;
        txnIds.push_back(requestScheduler_->submitWindowTransaction(txn));
        return txnIds;
    }

    uint32_t twoDWindowTransactionTileCount(uint32_t kCount) const {
        return std::max<uint32_t>(currentReuseMCount_, currentReuseNCount_) * kCount;
    }

    bool hasSplitKTransactions(const std::vector<uint64_t>& txnIds, uint32_t kCount) const {
        return kCount > 1 && txnIds.size() == static_cast<size_t>(kCount);
    }

    uint32_t twoDPerTransactionTileCount(const std::vector<uint64_t>& txnIds, uint32_t kCount) const {
        return hasSplitKTransactions(txnIds, kCount) ? twoDWindowTransactionTileCount(1) : twoDWindowTransactionTileCount(kCount);
    }

    bool are2DTransactionsReady(const std::vector<uint64_t>& txnIds, uint32_t tileCount, bool retireReady = true) {
        if (requestScheduler_ == nullptr || txnIds.empty()) {
            return false;
        }
        const uint32_t perTxnTileCount = txnIds.size() > 1 ? twoDWindowTransactionTileCount(1) : tileCount;
        bool allReady = true;
        for (uint64_t txnId : txnIds) {
            for (uint32_t idx = 0; idx < perTxnTileCount; ++idx) {
                if (!requestScheduler_->isTileReady(txnId, idx)) {
                    allReady = false;
                    continue;
                }
                if (retireReady) {
                    requestScheduler_->retireTileReady(txnId, idx);
                }
            }
        }
        return allReady;
    }

    void retire2DTransactions(const std::vector<uint64_t>& txnIds, uint32_t tileCount) {
        if (requestScheduler_ == nullptr) {
            return;
        }
        const uint32_t perTxnTileCount = txnIds.size() > 1 ? twoDWindowTransactionTileCount(1) : tileCount;
        for (uint64_t txnId : txnIds) {
            for (uint32_t idx = 0; idx < perTxnTileCount; ++idx) {
                if (requestScheduler_->isTileReady(txnId, idx)) {
                    requestScheduler_->retireTileReady(txnId, idx);
                }
            }
            if (requestScheduler_->isTransactionDone(txnId)) {
                requestScheduler_->retireTransaction(txnId);
            }
        }
    }

    bool activeTransactionsDone() const {
        if (requestScheduler_ == nullptr || activeTxnId_ == 0) {
            return false;
        }
        if (use2DWindowEngine() && !active2DTxnIds_.empty()) {
            for (uint64_t txnId : active2DTxnIds_) {
                if (!requestScheduler_->isTransactionDone(txnId)) {
                    return false;
                }
            }
            return true;
        }
        return requestScheduler_->isTransactionDone(activeTxnId_);
    }

    void retireActiveTransactions() {
        if (requestScheduler_ == nullptr || activeTxnId_ == 0) {
            return;
        }
        if (use2DWindowEngine() && !active2DTxnIds_.empty()) {
            for (uint64_t txnId : active2DTxnIds_) {
                if (requestScheduler_->isTransactionDone(txnId)) {
                    requestScheduler_->retireTransaction(txnId);
                }
            }
            return;
        }
        if (requestScheduler_->isTransactionDone(activeTxnId_)) {
            requestScheduler_->retireTransaction(activeTxnId_);
        }
    }

    void retireReady2DTransactions(
        const std::vector<uint64_t>& txnIds,
        uint32_t tileCount,
        std::vector<uint8_t>& retired) {
        if (requestScheduler_ == nullptr || txnIds.empty()) {
            return;
        }
        const uint32_t perTxnTileCount = txnIds.size() > 1 ? twoDWindowTransactionTileCount(1) : tileCount;
        const size_t retiredCount = txnIds.size() * static_cast<size_t>(perTxnTileCount);
        if (retired.size() != retiredCount) {
            retired.assign(retiredCount, 0);
        }
        for (size_t txnIdx = 0; txnIdx < txnIds.size(); ++txnIdx) {
            const uint64_t txnId = txnIds[txnIdx];
            for (uint32_t idx = 0; idx < perTxnTileCount; ++idx) {
                const size_t retiredIdx = txnIdx * static_cast<size_t>(perTxnTileCount) + idx;
                if (retired[retiredIdx] != 0) {
                    continue;
                }
                if (requestScheduler_->isTileReady(txnId, idx)) {
                    requestScheduler_->retireTileReady(txnId, idx);
                    retired[retiredIdx] = 1;
                }
            }
        }
    }

    bool is2DComputeTileReady(uint32_t localTileIdx) const {
        return is2DComputeTileReadyFor(localTileIdx, reuseMIndex_, reuseNIndex_);
    }

    bool is2DComputeTileReadyFor(uint32_t localTileIdx, uint32_t reuseM, uint32_t reuseN) const {
        if (!use2DWindowEngine()) {
            return false;
        }
        if (activeTxnId_ == 0 || active2DTxnIds_.empty()) {
            return true;
        }
        if (requestScheduler_ == nullptr || activeWindowKCount_ == 0) {
            return false;
        }
        const bool splitK = hasSplitKTransactions(active2DTxnIds_, activeWindowKCount_);
        const uint32_t matIdx = splitK ? reuseM : (reuseM * activeWindowKCount_ + localTileIdx);
        const uint32_t vecIdx = splitK ? reuseN : (reuseN * activeWindowKCount_ + localTileIdx);
        const size_t txnBegin = splitK ? static_cast<size_t>(localTileIdx) : 0;
        const size_t txnEnd = splitK ? std::min<size_t>(txnBegin + 1, active2DTxnIds_.size()) : active2DTxnIds_.size();
        for (size_t txnIdx = txnBegin; txnIdx < txnEnd; ++txnIdx) {
            const uint64_t txnId = active2DTxnIds_[txnIdx];
            bool matReady = false;
            bool ignoredVecReady = false;
            bool ignoredMatReady = false;
            bool vecReady = false;
            const bool haveMatState = requestScheduler_->getTileKStepReadiness(
                txnId, matIdx, 0, matReady, ignoredVecReady);
            const bool haveVecState = requestScheduler_->getTileKStepReadiness(
                txnId, vecIdx, 0, ignoredMatReady, vecReady);
            if (!haveMatState || !haveVecState) {
                matReady = requestScheduler_->isTileReady(txnId, matIdx);
                vecReady = requestScheduler_->isTileReady(txnId, vecIdx);
            }
            if (matReady && vecReady) {
                return true;
            }
        }
        return false;
    }

    bool selectNextReuseForActiveWindow(uint32_t& nextM, uint32_t& nextN, bool requireReady = false) const {
        if (!use2DWindowEngine()) {
            return false;
        }
        const size_t count = static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_);
        if (count == 0 || windowReuseDone_.size() != count) {
            return false;
        }

        bool haveFallback = false;
        uint32_t fallbackM = 0;
        uint32_t fallbackN = 0;
        const uint32_t start = static_cast<uint32_t>(reuseIndex(reuseMIndex_, reuseNIndex_) + 1u);
        for (uint32_t off = 0; off < static_cast<uint32_t>(count); ++off) {
            const uint32_t flat = (start + off) % static_cast<uint32_t>(count);
            const uint32_t m = flat / currentReuseNCount_;
            const uint32_t n = flat % currentReuseNCount_;
            if (windowReuseDone_[flat] != 0) {
                continue;
            }
            if (!haveFallback) {
                fallbackM = m;
                fallbackN = n;
                haveFallback = true;
            }
            for (uint32_t k = 0; k < activeWindowKCount_; ++k) {
                if (is2DComputeTileReadyFor(k, m, n)) {
                    nextM = m;
                    nextN = n;
                    return true;
                }
            }
        }
        if (!requireReady && haveFallback) {
            nextM = fallbackM;
            nextN = fallbackN;
            return true;
        }
        return false;
    }

    void activate2DWindow(uint32_t kBegin, uint32_t kCount, uint32_t buffer) {
        windowActivateCount_++;
        activeWindowKBegin_ = kBegin;
        activeWindowKCount_ = kCount;
        activeWindowBuffer_ = buffer;
        activeWindowValid_ = true;
        current_.k_begin = kBegin;
        current_.k_count = kCount;
        activeTxnTileCount_ = kCount;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.assign(kCount, 0);
        windowReuseDone_.assign(static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_), 0);
        nextReadyScanCursor_ = 0;
        activeTxnKBegin_ = kBegin;
        tileComputeStartCycles_.assign(kCount, 0);
        tileComputeDoneCycles_.assign(kCount, 0);
        tileRetireCycles_.assign(kCount, 0);
        tileComputeStartSchedCycles_.assign(kCount, 0);
        tileComputeDoneSchedCycles_.assign(kCount, 0);
        tileRetireSchedCycles_.assign(kCount, 0);
        allTilesScheduled_ = true;
        traceStage3("ACTIVATE_WINDOW", lastAccountCycle_);
    }

    void tryIssue2DWindowPrefetches() {
        const uint32_t totalK = std::max<uint32_t>(totalKTileCount_, 1u);
        if (!activeWindowValid_) {
            if (active2DTxnIds_.empty()) {
                active2DTxnIds_ = submit2DWindowTransactions(activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_);
                windowSubmitActiveCount_++;
                activeTxnId_ = active2DTxnIds_.empty() ? 0 : active2DTxnIds_.front();
                activeTxnTileCount_ = activeWindowKCount_;
                active2DSchedulerTileRetired_.assign(twoDWindowTransactionTileCount(activeWindowKCount_), 0);
                traceStage3("SUBMIT_ACTIVE_WINDOW", lastAccountCycle_);
            }
            retireReady2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), active2DSchedulerTileRetired_);
            uint32_t readyM = reuseMIndex_;
            uint32_t readyN = reuseNIndex_;
            if (selectNextReuseForActiveWindow(readyM, readyN, true)) {
                reuseMIndex_ = readyM;
                reuseNIndex_ = readyN;
                deriveTask(taskIndex_, false);
                traceStage3("ACTIVE_WINDOW_READY_REUSE", lastAccountCycle_);
                activate2DWindow(activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_);
            } else if (is2DComputeTileReady(0)) {
                traceStage3("ACTIVE_WINDOW_PROGRESSIVE_READY", lastAccountCycle_);
                activate2DWindow(activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_);
            } else if (are2DTransactionsReady(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_))) {
                retire2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_));
                active2DTxnIds_.clear();
                activeTxnId_ = 0;
                traceStage3("ACTIVE_WINDOW_READY", lastAccountCycle_);
                activate2DWindow(activeWindowKBegin_, activeWindowKCount_, activeWindowBuffer_);
            }
            if (!active2DTxnIds_.empty() || activeWindowValid_) {
                fill2DPrefetchQueue(totalK);
            }
            return;
        }
        retireReady2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), active2DSchedulerTileRetired_);
        fill2DPrefetchQueue(totalK);
    }

    void fill2DPrefetchQueue(uint32_t totalK) {
        while (prefetch2DWindows_.size() < prefetchWindowDepth_ && next2DPrefetchK_ < totalK) {
            uint32_t buffer = 0;
            if (!allocatePrefetchWindowBuffer(buffer)) {
                break;
            }
            Prefetch2DWindow window{};
            window.kBegin = next2DPrefetchK_;
            window.kCount = std::min<uint32_t>(residentKTileCount_, totalK - next2DPrefetchK_);
            window.buffer = buffer;
            window.txnIds = submit2DWindowTransactions(window.kBegin, window.kCount, window.buffer);
            if (window.txnIds.empty()) {
                break;
            }
            windowSubmitPrefetchCount_++;
            window.txnId = window.txnIds.front();
            next2DPrefetchK_ += window.kCount;
            prefetch2DWindows_.push_back(std::move(window));
            traceStage3("SUBMIT_PREFETCH_WINDOW", lastAccountCycle_);
        }
    }

    int selectNextTile() {
        if (activeComputeTileIndex_ >= 0) {
            return activeComputeTileIndex_;
        }
        if (use2DWindowEngine() && activeWindowValid_) {
            for (uint32_t idx = 0; idx < activeWindowKCount_; ++idx) {
                if (idx < activeTxnTileRetired_.size() && activeTxnTileRetired_[idx] == 0) {
                    if (is2DComputeTileReady(idx)) {
                        return static_cast<int>(idx);
                    }
                }
            }
            if (activeTxnRetiredTileCount_ == 0 &&
                !computeInFlight_ && activeComputeTileIndex_ < 0 &&
                !taskAccumInitialized_ && !activeTilePayloadLoaded_) {
                uint32_t readyM = reuseMIndex_;
                uint32_t readyN = reuseNIndex_;
                if (selectNextReuseForActiveWindow(readyM, readyN, true) &&
                    (readyM != reuseMIndex_ || readyN != reuseNIndex_)) {
                    reuseMIndex_ = readyM;
                    reuseNIndex_ = readyN;
                    deriveTask(taskIndex_, false);
                    resetComputeOnlyStateForNextTile();
                    allTilesScheduled_ = true;
                    traceStage3("SELECT_READY_REUSE", lastAccountCycle_);
                    for (uint32_t idx = 0; idx < activeWindowKCount_; ++idx) {
                        if (idx < activeTxnTileRetired_.size() && activeTxnTileRetired_[idx] == 0) {
                            if (is2DComputeTileReady(idx)) {
                                return static_cast<int>(idx);
                            }
                        }
                    }
                }
            }
            return -1;
        }
        if (use2DWindowEngine() && !activeWindowValid_) {
            return -1;
        }
        if (!use2DWindowEngine() && is2DReuse() && groupResidentValid_ && residentMacroTaskId_ == currentMacroTaskId_) {
            for (uint32_t idx = 0; idx < current_.k_count; ++idx) {
                if (idx < activeTxnTileRetired_.size() && activeTxnTileRetired_[idx] == 0) {
                    return static_cast<int>(idx);
                }
            }
            return -1;
        }
        if (requestScheduler_ == nullptr || activeTxnId_ == 0 || activeTxnTileCount_ == 0) {
            return -1;
        }

        if (!use2DWindowEngine() && is2DReuse() && activeTxnId_ != 0) {
            bool allReady = true;
            for (uint32_t idx = 0; idx < activeTxnTileCount_; ++idx) {
                if (requestScheduler_->isTileReady(activeTxnId_, idx)) {
                    requestScheduler_->retireTileReady(activeTxnId_, idx);
                } else {
                    allReady = false;
                }
            }
            if (!allReady) {
                return -1;
            }
            groupResidentValid_ = true;
            residentMacroTaskId_ = currentMacroTaskId_;
            requestScheduler_->retireTransaction(activeTxnId_);
            activeTxnId_ = 0;
            activeTxnTileCount_ = current_.k_count;
            activeTxnRetiredTileCount_ = 0;
            activeTxnTileRetired_.assign(current_.k_count, 0);
            nextReadyScanCursor_ = 0;
            for (uint32_t idx = 0; idx < current_.k_count; ++idx) {
                return static_cast<int>(idx);
            }
        }

        for (uint32_t off = 0; off < activeTxnTileCount_; ++off) {
            const uint32_t idx = (nextReadyScanCursor_ + off) % activeTxnTileCount_;
            if (idx < activeTxnTileRetired_.size() && activeTxnTileRetired_[idx] != 0) {
                continue;
            }
            if (requestScheduler_->isTileReady(activeTxnId_, idx)) {
                nextReadyScanCursor_ = (idx + 1) % activeTxnTileCount_;
                return static_cast<int>(idx);
            }
            bool matReady = false;
            bool vecReady = false;
            if (requestScheduler_->getTileKStepReadiness(activeTxnId_, idx, 0, matReady, vecReady) &&
                matReady && vecReady) {
                nextReadyScanCursor_ = (idx + 1) % activeTxnTileCount_;
                return static_cast<int>(idx);
            }
        }
        return -1;
    }

    bool loadTilePayload(uint32_t local_tile_idx) {
        const uint64_t vec_bytes = current_.vec_stride_bytes;
        const uint32_t slotCount = std::max<uint32_t>(header_.local_slot_count, 1u);
        const uint32_t matSlotIdx = is2DReuse() ? groupMatSlotFor(local_tile_idx) : (local_tile_idx % slotCount);
        const uint32_t vecSlotIdx = is2DReuse() ? groupVecSlotFor(local_tile_idx) : (local_tile_idx % slotCount);
        const uint64_t mat_addr = header_.local_mat_ping_gm_addr + static_cast<uint64_t>(matSlotIdx) * header_.local_mat_slot_stride_bytes;
        const uint64_t vec_addr = header_.local_vec_ping_gm_addr + static_cast<uint64_t>(vecSlotIdx) * header_.local_vec_slot_stride_bytes;
        globalMem_->rd_from_globalmem(mat_addr, static_cast<size_t>(current_.mat_stride_bytes), activeMatPayload_);
        globalMem_->rd_from_globalmem(vec_addr, static_cast<size_t>(vec_bytes), activeVecPayload_);
        if (activeMatPayload_.size() < current_.mat_stride_bytes || activeVecPayload_.size() < vec_bytes) {
            return false;
        }
        return true;
    }

    bool loadActiveMicroTileToArrays() {
        const uint32_t kBase = activeMicroKStep_ * current_.array_input_size;
        for (uint32_t array_id = 0; array_id < current_.block_n; ++array_id) {
            for (uint32_t idx = 0; idx < current_.array_input_size * current_.array_output_size; ++idx) {
                const uint32_t row = idx / current_.array_input_size;
                const uint32_t col = idx % current_.array_input_size;
                const size_t matIdx =
                    (static_cast<size_t>(row) * header_.block_k + static_cast<size_t>(kBase + col)) *
                    current_.elem_bytes;
                double value = decodeElement(&activeMatPayload_[matIdx]);
                array_->setMatrixItem(static_cast<int32_t>(array_id), static_cast<int32_t>(idx), value);
            }
            for (uint32_t idx = 0; idx < current_.array_input_size; ++idx) {
                const size_t off =
                    (static_cast<size_t>(array_id) * header_.block_k + static_cast<size_t>(kBase + idx)) *
                    current_.elem_bytes;
                double value = decodeElement(&activeVecPayload_[off]);
                array_->setVectorItem(static_cast<int32_t>(array_id), static_cast<int32_t>(idx), value);
            }
        }
        return true;
    }

    bool issueActiveMicroTile() {
        if (activeTileMicroOpCursor_ >= activeTileMicroOps_.size() || activeMicroOpIssued_) {
            return false;
        }
        if (activeComputeReadyQueue_.empty()) {
            refreshActiveComputeReadyQueue();
        }
        if (activeComputeReadyQueue_.empty()) {
            return false;
        }

        const MicroOp op = activeComputeReadyQueue_.front();
        activeComputeReadyQueue_.pop_front();
        if (activeComputeTileIndex_ < 0 || activeComputeSlotIndex_ < 0) {
            return false;
        }
        if (op.logicalTileIdx != static_cast<uint32_t>(activeComputeTileIndex_) ||
            op.slotIdx != static_cast<uint32_t>(activeComputeSlotIndex_)) {
            return false;
        }
        if (!isMicroOpReady(op)) {
            return false;
        }

        activeIssuedMicroOp_ = op;
        activeMicroOpIssued_ = true;
        activeMicroKStep_ = op.kStep;
        if (use2DWindowEngine() && !taskAccumInitialized_ && activeWindowKBegin_ > 0) {
            if (!loadPartialCToArray()) {
                return false;
            }
            taskAccumInitialized_ = true;
        }
        if (!loadActiveMicroTileToArrays()) {
            return false;
        }
        const uint64_t outputMode = taskAccumInitialized_ ? 1 : 0;
        taskAccumInitialized_ = true;
        for (uint32_t array_id = 0; array_id < current_.block_n; ++array_id) {
            array_->configureOutputMode(array_id, outputMode);
            array_->beginComputation(array_id);
        }
        return true;
    }

    size_t currentPartialIndex() const {
        const uint32_t reuseN = std::max<uint32_t>(header_.a_reuse_n_tiles, 1u);
        return static_cast<size_t>(reuseMIndex_) * reuseN + static_cast<size_t>(reuseNIndex_);
    }

    bool captureArrayOutput(std::vector<uint8_t>& tile) const {
        tile.assign(static_cast<size_t>(header_.block_m) * current_.block_n * current_.elem_bytes, 0);
        for (uint32_t n = 0; n < current_.block_n; ++n) {
            const size_t dst_off = static_cast<size_t>(n) * header_.block_m * current_.elem_bytes;
            if (outputIsFloat_) {
                auto* outVec = static_cast<std::vector<float>*>(array_->getOutputVector(n));
                if (outVec == nullptr || outVec->size() < header_.block_m) {
                    return false;
                }
                std::memcpy(&tile[dst_off], outVec->data(), static_cast<size_t>(header_.block_m) * current_.elem_bytes);
            } else {
                auto* outVec = static_cast<std::vector<int32_t>*>(array_->getOutputVector(n));
                if (outVec == nullptr || outVec->size() < header_.block_m) {
                    return false;
                }
                std::memcpy(&tile[dst_off], outVec->data(), static_cast<size_t>(header_.block_m) * current_.elem_bytes);
            }
        }
        return true;
    }

    bool savePartialCFromArray() {
        const size_t idx = currentPartialIndex();
        if (idx >= partialCTiles_.size() || idx >= partialValid_.size()) {
            return false;
        }
        if (!captureArrayOutput(partialCTiles_[idx])) {
            return false;
        }
        partialValid_[idx] = 1;
        return true;
    }

    bool loadPartialCToArray() {
        const size_t idx = currentPartialIndex();
        if (idx >= partialCTiles_.size() || idx >= partialValid_.size() || partialValid_[idx] == 0) {
            return false;
        }
        const auto& tile = partialCTiles_[idx];
        for (uint32_t n = 0; n < current_.block_n; ++n) {
            const size_t src_off = static_cast<size_t>(n) * header_.block_m * current_.elem_bytes;
            if (outputIsFloat_) {
                auto* outVec = static_cast<std::vector<float>*>(array_->getOutputVector(n));
                if (outVec == nullptr || outVec->size() < header_.block_m) {
                    return false;
                }
                std::memcpy(outVec->data(), &tile[src_off], static_cast<size_t>(header_.block_m) * current_.elem_bytes);
            } else {
                auto* outVec = static_cast<std::vector<int32_t>*>(array_->getOutputVector(n));
                if (outVec == nullptr || outVec->size() < header_.block_m) {
                    return false;
                }
                std::memcpy(outVec->data(), &tile[src_off], static_cast<size_t>(header_.block_m) * current_.elem_bytes);
            }
        }
        return true;
    }

    bool isFinal2DWindow() const {
        return !use2DWindowEngine() || (activeWindowKBegin_ + activeWindowKCount_ >= totalKTileCount_);
    }

    void resetComputeOnlyStateForNextTile() {
        allTilesScheduled_ = false;
        computeInFlight_ = false;
        activeComputeTileIndex_ = -1;
        activeComputeSlotIndex_ = -1;
        pendingArrays_ = 0;
        activeTxnRetiredTileCount_ = 0;
        activeTxnTileRetired_.assign(activeWindowKCount_, 0);
        nextReadyScanCursor_ = 0;
        activeMicroKStep_ = 0;
        taskAccumInitialized_ = false;
        activeMatPayload_.clear();
        activeVecPayload_.clear();
        activeTileMicroOps_.clear();
        activeTileMicroOpCursor_ = 0;
        activeTileScoreboard_ = KStepScoreboard{};
        activeComputeReadyQueue_.clear();
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTilePayloadLoaded_ = false;
        tileComputeStartCycles_.assign(activeWindowKCount_, 0);
        tileComputeDoneCycles_.assign(activeWindowKCount_, 0);
        tileRetireCycles_.assign(activeWindowKCount_, 0);
        tileComputeStartSchedCycles_.assign(activeWindowKCount_, 0);
        tileComputeDoneSchedCycles_.assign(activeWindowKCount_, 0);
        tileRetireSchedCycles_.assign(activeWindowKCount_, 0);
    }

    bool advance2DWindowEngine() {
        traceStage3("ADVANCE_ENTER", lastAccountCycle_);
        if (activeWindowValid_ && !allReuseDoneForWindow()) {
            uint32_t nextM = reuseMIndex_;
            uint32_t nextN = reuseNIndex_;
            if (selectNextReuseForActiveWindow(nextM, nextN)) {
                reuseMIndex_ = nextM;
                reuseNIndex_ = nextN;
                deriveTask(taskIndex_, false);
                resetComputeOnlyStateForNextTile();
                allTilesScheduled_ = activeWindowValid_;
                traceStage3("ADVANCE_READY_REUSE", lastAccountCycle_);
                return true;
            }
        }
        if (activeWindowValid_ && allReuseDoneForWindow()) {
            reuseNIndex_ = 0;
            reuseMIndex_ = 0;
        } else {
        if ((reuseNIndex_ + 1) < currentReuseNCount_) {
            reuseNIndex_ += 1;
            deriveTask(taskIndex_, false);
            resetComputeOnlyStateForNextTile();
            allTilesScheduled_ = activeWindowValid_;
            traceStage3("ADVANCE_REUSE_N", lastAccountCycle_);
            return true;
        }
        if ((reuseMIndex_ + 1) < currentReuseMCount_) {
            reuseMIndex_ += 1;
            reuseNIndex_ = 0;
            deriveTask(taskIndex_, false);
            resetComputeOnlyStateForNextTile();
            allTilesScheduled_ = activeWindowValid_;
            traceStage3("ADVANCE_REUSE_M", lastAccountCycle_);
            return true;
        }
        }
        reuseNIndex_ = 0;
        reuseMIndex_ = 0;
        if (isFinal2DWindow()) {
            if (activeTxnId_ != 0) {
                retireReady2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), active2DSchedulerTileRetired_);
                if (activeTransactionsDone()) {
                    retireActiveTransactions();
                    activeTxnId_ = 0;
                    active2DTxnIds_.clear();
                    active2DSchedulerTileRetired_.clear();
                }
            }
            traceStage3("ADVANCE_FINAL_WINDOW_DONE", lastAccountCycle_);
            return false;
        }
        if (!prefetch2DWindows_.empty()) {
            Prefetch2DWindow nextWindow = std::move(prefetch2DWindows_.front());
            prefetch2DWindows_.pop_front();
            if (!are2DTransactionsReady(nextWindow.txnIds, twoDWindowTransactionTileCount(nextWindow.kCount), false)) {
                windowAdvanceWaitPrefetchCount_++;
                traceStage3("ADVANCE_WAIT_PREFETCH", lastAccountCycle_);
                activeWindowValid_ = false;
                activeTxnId_ = nextWindow.txnId;
                active2DTxnIds_ = std::move(nextWindow.txnIds);
                activeTxnTileCount_ = nextWindow.kCount;
                activeWindowKBegin_ = nextWindow.kBegin;
                activeWindowKCount_ = nextWindow.kCount;
                activeWindowBuffer_ = nextWindow.buffer;
                active2DSchedulerTileRetired_.assign(twoDWindowTransactionTileCount(activeWindowKCount_), 0);
                windowReuseDone_.assign(static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_), 0);
                deriveTask(taskIndex_, false);
                resetComputeOnlyStateForNextTile();
                return true;
            }
            retire2DTransactions(nextWindow.txnIds, twoDWindowTransactionTileCount(nextWindow.kCount));
            activeTxnId_ = 0;
            active2DTxnIds_.clear();
            active2DSchedulerTileRetired_.clear();
            traceStage3("ADVANCE_PREFETCH_READY", lastAccountCycle_);
            activate2DWindow(nextWindow.kBegin, nextWindow.kCount, nextWindow.buffer);
        } else {
            traceStage3("ADVANCE_NO_PREFETCH", lastAccountCycle_);
            const uint32_t nextK = activeWindowKBegin_ + activeWindowKCount_;
            const uint32_t nextCount = std::min<uint32_t>(residentKTileCount_, totalKTileCount_ - nextK);
            activeWindowValid_ = false;
            activeWindowKBegin_ = nextK;
            activeWindowKCount_ = nextCount;
            uint32_t buffer = 0;
            if (allocatePrefetchWindowBuffer(buffer)) {
                activeWindowBuffer_ = buffer;
            } else {
                activeWindowBuffer_ = (activeWindowBuffer_ + 1u) % twoDWindowBufferCount();
            }
            windowReuseDone_.assign(static_cast<size_t>(currentReuseMCount_) * static_cast<size_t>(currentReuseNCount_), 0);
            next2DPrefetchK_ = nextK + nextCount;
        }
        deriveTask(taskIndex_, false);
        resetComputeOnlyStateForNextTile();
        allTilesScheduled_ = activeWindowValid_;
        return true;
    }

    bool completeActiveMicroTile() {
        if (!activeMicroOpIssued_ || activeTileMicroOpCursor_ >= activeTileMicroOps_.size()) {
            return false;
        }

        const MicroOp& expectedOp = activeTileMicroOps_[activeTileMicroOpCursor_];
        if (expectedOp.kStep != activeMicroKStep_ ||
            expectedOp.kStep != activeIssuedMicroOp_.kStep ||
            expectedOp.logicalTileIdx != activeIssuedMicroOp_.logicalTileIdx) {
            return false;
        }

        activeTileScoreboard_.lastCompletedKStep = static_cast<int32_t>(activeIssuedMicroOp_.kStep);
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTileMicroOpCursor_ += 1;
        refreshActiveComputeReadyQueue();
        if (activeTileMicroOpCursor_ < activeTileMicroOps_.size()) {
            activeTilePayloadLoaded_ = false;
            return true;
        }
        activeMicroKStep_ = 0;

        const int doneTile = activeComputeTileIndex_;
        const int doneSlot = activeComputeSlotIndex_;
        if (doneTile >= 0 && static_cast<size_t>(doneTile) < tileComputeDoneCycles_.size()) {
            tileComputeDoneCycles_[static_cast<size_t>(doneTile)] = lastAccountCycle_;
            if (static_cast<size_t>(doneTile) < tileComputeDoneSchedCycles_.size()) {
                tileComputeDoneSchedCycles_[static_cast<size_t>(doneTile)] = schedulerTimelineCycle();
            }
        }
        if (requestScheduler_ != nullptr && activeTxnId_ != 0) {
            if (use2DWindowEngine()) {
                retireReady2DTransactions(active2DTxnIds_, twoDWindowTransactionTileCount(activeWindowKCount_), active2DSchedulerTileRetired_);
            } else {
                WcpTileTimelineDebug dbg{};
                const bool haveDbg = requestScheduler_->getTileTimeline(activeTxnId_, static_cast<uint32_t>(doneTile), dbg);
                requestScheduler_->retireTileReady(activeTxnId_, static_cast<uint32_t>(doneTile));
                if (doneTile >= 0 && static_cast<size_t>(doneTile) < activeTxnTileRetired_.size() &&
                    activeTxnTileRetired_[static_cast<size_t>(doneTile)] == 0) {
                    activeTxnTileRetired_[static_cast<size_t>(doneTile)] = 1;
                    activeTxnRetiredTileCount_ += 1;
                }
                if (doneTile >= 0 && static_cast<size_t>(doneTile) < tileRetireCycles_.size()) {
                    tileRetireCycles_[static_cast<size_t>(doneTile)] = lastAccountCycle_;
                    if (static_cast<size_t>(doneTile) < tileRetireSchedCycles_.size()) {
                        tileRetireSchedCycles_[static_cast<size_t>(doneTile)] = schedulerTimelineCycle();
                    }
                }
                if (haveDbg && extOutput_ != nullptr && doneTile >= 0 && static_cast<size_t>(doneTile) < tileRetireCycles_.size()) {
                    extOutput_->output(
                        "[Core %u] [wcp] TILE TRACE: task=%" PRIu64 " tile=%d slot=%d submit=%" PRIu64
                        " mat_done=%" PRIu64 " vec_done=%" PRIu64 " ready=%" PRIu64
                        " compute_start=%" PRIu64 " compute_done=%" PRIu64 " retire=%" PRIu64
                        " compute_start_sched=%" PRIu64 " compute_done_sched=%" PRIu64 " retire_sched=%" PRIu64 "\n",
                        coreId_, current_.task_id, doneTile, doneSlot,
                        dbg.submitCycle, dbg.matDoneCycle, dbg.vecDoneCycle, dbg.readyCycle,
                        tileComputeStartCycles_[static_cast<size_t>(doneTile)],
                        tileComputeDoneCycles_[static_cast<size_t>(doneTile)],
                        tileRetireCycles_[static_cast<size_t>(doneTile)],
                        tileComputeStartSchedCycles_[static_cast<size_t>(doneTile)],
                        tileComputeDoneSchedCycles_[static_cast<size_t>(doneTile)],
                        tileRetireSchedCycles_[static_cast<size_t>(doneTile)]);
                }
            }
        }
        if (use2DWindowEngine() && activeWindowValid_) {
            if (doneTile >= 0 && static_cast<size_t>(doneTile) < activeTxnTileRetired_.size() &&
                activeTxnTileRetired_[static_cast<size_t>(doneTile)] == 0) {
                activeTxnTileRetired_[static_cast<size_t>(doneTile)] = 1;
                activeTxnRetiredTileCount_ += 1;
            }
        } else if (!use2DWindowEngine() && is2DReuse() && groupResidentValid_ && residentMacroTaskId_ == currentMacroTaskId_) {
            if (doneTile >= 0 && static_cast<size_t>(doneTile) < activeTxnTileRetired_.size() &&
                activeTxnTileRetired_[static_cast<size_t>(doneTile)] == 0) {
                activeTxnTileRetired_[static_cast<size_t>(doneTile)] = 1;
                activeTxnRetiredTileCount_ += 1;
            }
        }
        if (activeComputeSlotIndex_ >= 0 && activeComputeSlotIndex_ < static_cast<int>(std::max<uint32_t>(header_.local_slot_count, 1u))) {
            if (activeComputeSlotIndex_ < static_cast<int>(buffers_.size())) {
                buffers_[activeComputeSlotIndex_].in_use = false;
            }
        }
        activeComputeTileIndex_ = -1;
        activeComputeSlotIndex_ = -1;
        activeMicroKStep_ = 0;
        activeMatPayload_.clear();
        activeVecPayload_.clear();
        activeTileMicroOps_.clear();
        activeTileMicroOpCursor_ = 0;
        activeTileScoreboard_ = KStepScoreboard{};
        activeComputeReadyQueue_.clear();
        activeMicroOpIssued_ = false;
        activeIssuedMicroOp_ = MicroOp{};
        activeTilePayloadLoaded_ = false;
        return true;
    }

    uint32_t microKStepCount() const {
        return std::max<uint32_t>(header_.block_k / std::max<uint32_t>(current_.array_input_size, 1u), 1u);
    }

    uint64_t schedulerTimelineCycle() const {
        if (requestScheduler_ == nullptr) {
            return 0;
        }
        return requestScheduler_->getTimelineCycle();
    }

    double decodeElement(const uint8_t* raw) const {
        if (outputIsFloat_) {
            float value = 0.0f;
            std::memcpy(&value, raw, sizeof(value));
            return static_cast<double>(value);
        }

        int32_t value = 0;
        std::memcpy(&value, raw, sizeof(value));
        return static_cast<double>(value);
    }

    void deriveTask(uint32_t taskIndex, bool resetState = true) {
        const uint32_t macro_task_id = header_.worker_slot + taskIndex * header_.active_worker_cores;
        const uint32_t m_tiles = header_.m / header_.block_m;
        const uint32_t n_tiles = header_.n / header_.block_n;
        const uint32_t k_tiles = header_.k / header_.block_k;
        const uint32_t reuseN = std::max<uint32_t>(header_.a_reuse_n_tiles, 1u);
        const uint32_t reuseM = std::max<uint32_t>(header_.b_reuse_m_tiles, 1u);
        uint32_t m_tile = 0;
        uint32_t n_tile = 0;
        const uint32_t m_groups = header_.m_group_count != 0
                                      ? header_.m_group_count
                                      : ((m_tiles + reuseM - 1) / reuseM);
        const uint32_t n_groups = header_.n_group_count != 0
                                      ? header_.n_group_count
                                      : ((n_tiles + reuseN - 1) / reuseN);
        const uint32_t m_group = macro_task_id % m_groups;
        const uint32_t n_group = ((macro_task_id / m_groups) + m_group) % n_groups;
        const uint32_t m_begin = m_group * reuseM;
        const uint32_t n_begin = n_group * reuseN;
        currentReuseMCount_ = std::min<uint32_t>(reuseM, m_tiles - m_begin);
        currentReuseNCount_ = std::min<uint32_t>(reuseN, n_tiles - n_begin);
        if (reuseM == 1) {
            reuseMIndex_ = 0;
        }
        if (reuseN == 1) {
            reuseNIndex_ = 0;
        }
        if (reuseMIndex_ >= currentReuseMCount_) {
            reuseMIndex_ = 0;
        }
        if (reuseNIndex_ >= currentReuseNCount_) {
            reuseNIndex_ = 0;
        }
        m_tile = m_begin + reuseMIndex_;
        n_tile = n_begin + reuseNIndex_;
        const uint64_t output_task_id = static_cast<uint64_t>(m_tile) * n_tiles + n_tile;
        const uint32_t activeWorkers = std::max<uint32_t>(header_.active_worker_cores, 1u);
        auto nodeForMacroTask = [&](uint32_t macroTaskId) -> uint32_t {
            const uint32_t owner_slot = macroTaskId % activeWorkers;
            const uint32_t group_id = header_.total_groups > 0 ? (owner_slot % header_.total_groups) : 0;
            const uint32_t dataMapKey = header_.data_node_map_mode != 0 ? owner_slot : group_id;
            return 1 + (header_.data_memory_node_count > 0 ? (dataMapKey % header_.data_memory_node_count) : 0);
        };
        auto macroSlotInNode = [&](uint32_t macroTaskId, uint32_t nodeIdx) -> uint32_t {
            uint32_t slot = 0;
            for (uint32_t i = 0; i < macroTaskId; ++i) {
                if (nodeForMacroTask(i) == nodeIdx) {
                    slot++;
                }
            }
            return slot;
        };
        const uint32_t c_node_idx = nodeForMacroTask(macro_task_id);
        const uint32_t macro_slot = macroSlotInNode(macro_task_id, c_node_idx);
        const uint32_t a_owner_macro = m_group;
        const uint32_t b_owner_macro = n_group;
        const uint32_t a_node_idx = nodeForMacroTask(a_owner_macro);
        const uint32_t b_node_idx = nodeForMacroTask(b_owner_macro);
        uint32_t a_slot = 0;
        for (uint32_t mt = 0; mt < m_tile; ++mt) {
            const uint32_t mt_group = mt / reuseM;
            if (nodeForMacroTask(mt_group) == a_node_idx) {
                a_slot++;
            }
        }
        uint32_t b_slot = 0;
        for (uint32_t nt = 0; nt < n_tile; ++nt) {
            const uint32_t nt_group = nt / reuseN;
            if (nodeForMacroTask(nt_group) == b_node_idx) {
                b_slot++;
            }
        }

        const uint64_t a_node_base = static_cast<uint64_t>(a_node_idx) * header_.mem_node_size;
        const uint64_t b_node_base = static_cast<uint64_t>(b_node_idx) * header_.mem_node_size;
        const uint64_t c_node_base = static_cast<uint64_t>(c_node_idx) * header_.mem_node_size;
        const uint64_t mat_group_slot = static_cast<uint64_t>(a_slot);
        const uint64_t vec_group_slot = static_cast<uint64_t>(b_slot);
        const uint64_t out_group_slot =
            static_cast<uint64_t>(macro_slot) * (reuseM > 1 ? reuseM : 1) * (reuseN > 1 ? reuseN : 1) +
            static_cast<uint64_t>(reuseMIndex_) * (reuseN > 1 ? reuseN : 1) + static_cast<uint64_t>(reuseNIndex_);
        current_.task_id = output_task_id;
        current_.task_flags = 0;
        current_.mat_base_addr = a_node_base + header_.off_gemm_mat_base + mat_group_slot * static_cast<uint64_t>(k_tiles) * header_.mat_stride_bytes;
        current_.vec_base_addr = b_node_base + header_.off_gemm_vec_base + vec_group_slot * static_cast<uint64_t>(k_tiles * header_.block_n) * header_.vec_stride_bytes;
        const uint64_t out_bytes = static_cast<uint64_t>(header_.block_m) * static_cast<uint64_t>(header_.block_n) * static_cast<uint64_t>(header_.elem_bytes);
        const uint64_t out_stride = ((out_bytes + 0xffULL) / 0x100ULL) * 0x100ULL;
        current_.c_base_addr = c_node_base + header_.off_gemm_out_base + out_group_slot * out_stride;
        current_.accum_base_addr = header_.local_accum_gm_addr;
        current_.completion_flag_addr = 0;
        current_.completion_value = 0;
        current_.k_begin = (!resetState && use2DWindowEngine()) ? activeWindowKBegin_ : 0;
        current_.k_count = (!resetState && use2DWindowEngine() && activeWindowKCount_ != 0) ? activeWindowKCount_ : k_tiles;
        current_.block_n = header_.block_n;
        current_.hw_input_size = header_.hw_input_size;
        current_.hw_output_size = header_.hw_output_size;
        current_.array_input_size = header_.hw_input_size;
        current_.array_output_size = header_.hw_output_size;
        current_.elem_bytes = header_.elem_bytes;
        current_.mat_stride_bytes = header_.mat_stride_bytes;
        current_.vec_stride_bytes = static_cast<uint64_t>(header_.block_n) * header_.vec_stride_bytes;
        current_.local_accum_gm_addr = header_.local_accum_gm_addr;
        current_.local_out_gm_addr = header_.local_out_gm_addr;
        currentK_ = current_.k_begin;
        currentMacroTaskId_ = macro_task_id;
        if (resetState) {
            resetPipelineState();
        }
    }

    enum class Phase : uint8_t {
        IDLE = 0,
        RUN,
        WRITEBACK,
        WRITEBACK_WAIT,
        DONE,
    };

    int verbose_ = 0;
    bool outputIsFloat_ = false;
    bool stage3Trace_ = false;
    uint32_t prefetchWindowDepth_ = 1;
    uint32_t windowKtiles_ = 4;
    SST::Output output_;
    uint32_t coreId_ = 0;
    SST::Output* extOutput_ = nullptr;
    SST::Golem::GlobalMemoryAPI* globalMem_ = nullptr;
    SST::Golem::ComputeArray* array_ = nullptr;
    SST::Golem::RequestSchedulerAPI* requestScheduler_ = nullptr;
    WorkerTaskListHeader header_{};
    WorkerWindowDescriptor current_{};
    bool busy_ = false;
    Phase phase_ = Phase::IDLE;
    uint32_t currentK_ = 0;
    uint32_t nextPrefetchK_ = 0;
    uint32_t taskIndex_ = 0;
    uint32_t pendingArrays_ = 0;
    uint32_t currentWindowId_ = 0;
    uint32_t activeTxnTileCount_ = 0;
    uint32_t nextTxnComputeTile_ = 0;
    uint32_t activeTxnRetiredTileCount_ = 0;
    uint32_t nextReadyScanCursor_ = 0;
    uint32_t activeTxnKBegin_ = 0;
    uint32_t reuseNIndex_ = 0;
    uint32_t reuseMIndex_ = 0;
    uint32_t currentReuseNCount_ = 1;
    uint32_t currentReuseMCount_ = 1;
    uint32_t currentMacroTaskId_ = 0;
    uint32_t residentMacroTaskId_ = UINT32_MAX;
    bool groupResidentValid_ = false;
    uint32_t totalKTileCount_ = 0;
    uint32_t residentKTileCount_ = 0;
    uint32_t activeWindowKBegin_ = 0;
    uint32_t activeWindowKCount_ = 0;
    uint32_t activeWindowBuffer_ = 0;
    bool activeWindowValid_ = false;
    uint32_t next2DPrefetchK_ = 0;
    std::vector<uint64_t> active2DTxnIds_;
    std::deque<Prefetch2DWindow> prefetch2DWindows_;
    std::vector<uint8_t> active2DSchedulerTileRetired_;
    uint64_t lastAccountCycle_ = 0;
    uint64_t workerStartCycle_ = 0;
    uint64_t workerEndCycle_ = 0;
    uint64_t totalWindowCycles_ = 0;
    uint64_t computeCycles_ = 0;
    uint64_t tileReadyWaitCycles_ = 0;
    uint64_t txnWaitCycles_ = 0;
    uint64_t writebackWaitCycles_ = 0;
    uint64_t wait2DActivateCycles_ = 0;
    uint64_t wait2DActiveNotReadyCycles_ = 0;
    uint64_t waitNon2DTxnCycles_ = 0;
    uint64_t waitNoActiveTxnCycles_ = 0;
    uint64_t windowSubmitActiveCount_ = 0;
    uint64_t windowSubmitPrefetchCount_ = 0;
    uint64_t windowActivateCount_ = 0;
    uint64_t windowAdvanceWaitPrefetchCount_ = 0;
    uint64_t lastStage3TraceCycle_ = 0;
    bool allTilesScheduled_ = false;
    bool computeInFlight_ = false;
    bool writebackDone_ = false;
    uint64_t activeTxnId_ = 0;
    uint64_t writebackToken_ = 0;
    std::deque<uint64_t> pendingWritebackTokens_;
    int activeComputeTileIndex_ = -1;
    int activeComputeSlotIndex_ = -1;
    uint32_t activeMicroKStep_ = 0;
    std::vector<BufferSlot> buffers_;
    std::vector<uint64_t> tileComputeStartCycles_;
    std::vector<uint64_t> tileComputeDoneCycles_;
    std::vector<uint64_t> tileRetireCycles_;
    std::vector<uint64_t> tileComputeStartSchedCycles_;
    std::vector<uint64_t> tileComputeDoneSchedCycles_;
    std::vector<uint64_t> tileRetireSchedCycles_;
    std::vector<uint8_t> activeTxnTileRetired_;
    std::vector<uint8_t> windowReuseDone_;
    std::vector<uint8_t> activeMatPayload_;
    std::vector<uint8_t> activeVecPayload_;
    std::vector<MicroOp> activeTileMicroOps_;
    size_t activeTileMicroOpCursor_ = 0;
    KStepScoreboard activeTileScoreboard_;
    std::deque<MicroOp> activeComputeReadyQueue_;
    bool activeMicroOpIssued_ = false;
    MicroOp activeIssuedMicroOp_{};
    bool activeTilePayloadLoaded_ = false;
    bool taskAccumInitialized_ = false;
    std::vector<std::vector<uint8_t>> partialCTiles_;
    std::vector<uint8_t> partialValid_;
};

} // namespace Golem
} // namespace SST

#endif
