#include <cinttypes>

#include <sst/elements/golem/sfu/sfu.h>

namespace SST {
namespace Golem {

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
    (void)descAddr;
    (void)tag;

    if (inflight_ >= maxInflight_) {
        statCreditStalls_->addData(1);
        return false;
    }

    ++inflight_;
    statOpsIssued_->addData(1);
    statSoftmaxTiles_->addData(1);
    statPartialSubmits_->addData(1);
    statPartialDone_->addData(1);
    return true;
}

bool SFU::wait(uint64_t tag, uint64_t* status)
{
    (void)tag;
    if (inflight_ > 0) {
        --inflight_;
    }
    if (status != nullptr) {
        *status = 0;
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
