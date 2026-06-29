#ifndef _H_GOLEM_SFU
#define _H_GOLEM_SFU

#include <cstdint>

#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/subcomponent.h>

#include <sst/elements/golem/globalmemory/globalmemory.h>

namespace SST {
namespace Golem {

class SFUAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::SFUAPI)

    SFUAPI(ComponentId_t id, SST::Params& params) : SST::SubComponent(id) {}
    ~SFUAPI() override = default;

    virtual bool issueSoftmaxTile(uint64_t descAddr, uint64_t tag) = 0;
    virtual bool wait(uint64_t tag, uint64_t* status) = 0;
    virtual void bindGlobalMemory(GlobalMemoryAPI* globalMem) = 0;
    virtual void setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores) = 0;
};

class SFU : public SFUAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        SFU,
        "golem",
        "SFU",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Special Function Unit for Golem",
        SST::Golem::SFUAPI)

    SST_ELI_DOCUMENT_PARAMS(
        {"core_id", "Owning core id", "0"},
        {"active_worker_cores", "Number of active worker cores", "1"},
        {"max_inflight", "Maximum in-flight SFU operations", "8"},
        {"stats_latency", "Softmax tile statistics latency", "1"},
        {"merge_latency", "Softmax online merge latency", "1"},
        {"normalize_latency", "Softmax normalize latency", "1"},
        {"verbose", "Verbosity", "0"})

    SST_ELI_DOCUMENT_STATISTICS(
        {"sfu_ops_issued", "Issued SFU operations", "ops", 1},
        {"sfu_softmax_rows", "Softmax rows processed by SFU", "rows", 1},
        {"sfu_softmax_tiles", "Softmax tiles processed by SFU", "tiles", 1},
        {"sfu_partial_submits", "Softmax partial stats submitted", "partials", 1},
        {"sfu_partial_done", "Softmax partial stats completed", "partials", 1},
        {"sfu_credit_stalls", "SFU credit stalls", "stalls", 1},
        {"sfu_cross_tile_wait_cycles", "SFU cross-tile wait cycles", "cycles", 1},
        {"sfu_retry_events", "SFU retry events", "events", 1})

    SFU(ComponentId_t id, SST::Params& params);
    ~SFU() override = default;

    void finish() override;

    bool issueSoftmaxTile(uint64_t descAddr, uint64_t tag) override;
    bool wait(uint64_t tag, uint64_t* status) override;
    void bindGlobalMemory(GlobalMemoryAPI* globalMem) override;
    void setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores) override;

private:
    uint32_t coreId_;
    uint32_t activeWorkerCores_;
    uint32_t maxInflight_;
    uint32_t statsLatency_;
    uint32_t mergeLatency_;
    uint32_t normalizeLatency_;
    uint32_t inflight_;
    int verbose_;

    GlobalMemoryAPI* globalMem_;
    SST::Output output_;

    Statistic<uint64_t>* statOpsIssued_;
    Statistic<uint64_t>* statSoftmaxRows_;
    Statistic<uint64_t>* statSoftmaxTiles_;
    Statistic<uint64_t>* statPartialSubmits_;
    Statistic<uint64_t>* statPartialDone_;
    Statistic<uint64_t>* statCreditStalls_;
    Statistic<uint64_t>* statCrossTileWaitCycles_;
    Statistic<uint64_t>* statRetryEvents_;
};

} // namespace Golem
} // namespace SST

#endif
