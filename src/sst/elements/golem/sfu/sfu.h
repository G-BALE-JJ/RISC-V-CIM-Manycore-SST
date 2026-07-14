#ifndef _H_GOLEM_SFU
#define _H_GOLEM_SFU

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/subcomponent.h>

#include <sst/elements/golem/globalmemory/globalmemory.h>

namespace SST {
namespace Golem {

struct SFUSoftmaxTileDesc {
    uint64_t job_id;
    uint64_t local_input_gm_addr;
    uint64_t local_output_gm_addr;
    uint32_t global_m;
    uint32_t global_n;
    uint32_t block_m;
    uint32_t block_n;
    uint32_t m_tile;
    uint32_t n_tile;
    uint32_t valid_m;
    uint32_t valid_n;
    uint32_t n_tiles_per_row;
    uint32_t elem_bytes;
    uint32_t flags;
};

static_assert(sizeof(SFUSoftmaxTileDesc) == 72,
              "SFUSoftmaxTileDesc ABI must stay fixed for RISC-V workload descriptors");

enum class SFUPrimitiveOp : uint32_t {
    EXP = 0x01,
    LOG = 0x02,
    RECIPROCAL = 0x03,
    RSQRT = 0x04,
    SQRT = 0x05,
    TANH = 0x06,
    SIGMOID = 0x07,
    REDUCE_MAX = 0x20,
    REDUCE_SUM = 0x21,
    GELU = 0x40,
    LAYERNORM = 0x41,
    FUSED_SOFTMAX = 0x80,
};

struct SFUPrimitiveDesc {
    uint64_t job_id;
    uint64_t input0_gm_addr;
    uint64_t input1_gm_addr;
    uint64_t output_gm_addr;
    uint32_t op;
    uint32_t dtype;
    uint32_t elem_count;
    uint32_t input0_stride_bytes;
    uint32_t input1_stride_bytes;
    uint32_t output_stride_bytes;
    uint32_t flags;
    uint32_t approx_mode;
};

static_assert(sizeof(SFUPrimitiveDesc) == 64,
              "SFUPrimitiveDesc ABI must stay fixed for RISC-V workload descriptors");

struct SFUPrimitiveBatchDesc {
    uint64_t job_id;
    uint64_t desc_array_gm_addr;
    uint32_t desc_count;
    uint32_t flags;
    uint64_t reserved0;
};

static_assert(sizeof(SFUPrimitiveBatchDesc) == 32,
              "SFUPrimitiveBatchDesc ABI must stay fixed for RISC-V workload descriptors");

enum class SFUJobOp : uint32_t {
    ELEMENTWISE = 0x01,
    REDUCE = 0x02,
    SOFTMAX_ROW = 0x10,
    LAYERNORM = 0x11,
    GELU = 0x12,
};

enum class SFUJobSubOp : uint32_t {
    NONE = 0x00,
    EXP = 0x01,
    LOG = 0x02,
    RECIPROCAL = 0x03,
    RSQRT = 0x04,
    TANH = 0x05,
    SIGMOID = 0x06,
    REDUCE_MAX = 0x20,
    REDUCE_SUM = 0x21,
};

constexpr uint32_t SFU_JOB_FLAG_DISTRIBUTED_COLUMNS = 0x1u;
constexpr uint32_t SFU_JOB_FLAG_DISTRIBUTED_ABORT = 0x2u;

struct SFUJobDesc {
    uint64_t job_id;
    uint64_t input0_addr;
    uint64_t input1_addr;
    uint64_t output_addr;
    uint64_t params_addr;
    uint64_t scratch_addr;
    uint32_t op_type;
    uint32_t sub_op;
    uint32_t dtype;
    uint32_t layout;
    uint32_t rows;
    uint32_t cols;
    uint32_t elem_count;
    uint32_t chunk_elems;
    uint32_t worker_cores;
    uint32_t owner_core;
    uint32_t flags;
    uint32_t reserved0;  // With DISTRIBUTED_COLUMNS, reserved0 stores the worker slot.
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t reserved3;
    uint64_t reserved4;
};

static_assert(sizeof(SFUJobDesc) == 128,
              "SFUJobDesc ABI must stay fixed for RISC-V workload descriptors");

enum class SFUStatus : uint64_t {
    Success = 0,
    Pending = 1,
    InvalidDescriptor = 2,
    UnsupportedElemBytes = 3,
    GlobalMemoryUnavailable = 4,
    InvalidShape = 5,
};

struct SFUTileRowStats {
    uint32_t global_row;
    double tile_m;
    double tile_l;
};

class SFUAPI : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::Golem::SFUAPI)

    SFUAPI(ComponentId_t id, SST::Params& params) : SST::SubComponent(id) {}
    ~SFUAPI() override = default;

    virtual bool issueSoftmaxTile(uint64_t descAddr, uint64_t tag) = 0;
    virtual bool issuePrimitive(uint64_t descAddr, uint64_t tag) = 0;
    virtual bool issuePrimitiveBatch(uint64_t descAddr, uint64_t tag) = 0;
    virtual bool issueJob(uint64_t descAddr, uint64_t tag) = 0;
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
        {"distributed_reduction_transport", "Distributed softmax reduction transport: shared, modeled_noc, or explicit_noc", "shared"},
        {"verbose", "Verbosity", "0"})

    SST_ELI_DOCUMENT_STATISTICS(
        {"sfu_ops_issued", "Issued SFU operations", "ops", 1},
        {"sfu_softmax_rows", "Softmax rows processed by SFU", "rows", 1},
        {"sfu_softmax_tiles", "Softmax tiles processed by SFU", "tiles", 1},
        {"sfu_job_softmax_max_chunks", "Unified softmax job max-pass chunks", "chunks", 1},
        {"sfu_job_softmax_sum_chunks", "Unified softmax job exp/sum-pass chunks", "chunks", 1},
        {"sfu_job_softmax_norm_chunks", "Unified softmax job normalize-pass chunks", "chunks", 1},
        {"sfu_primitive_elems", "Logical primitive elements processed by SFU", "elements", 1},
        {"sfu_partial_submits", "Softmax partial stats submitted", "partials", 1},
        {"sfu_partial_done", "Softmax partial stats completed", "partials", 1},
        {"sfu_reduction_max_requests", "Distributed softmax max reduction requests", "messages", 1},
        {"sfu_reduction_max_responses", "Distributed softmax max reduction responses", "messages", 1},
        {"sfu_reduction_sum_requests", "Distributed softmax sum reduction requests", "messages", 1},
        {"sfu_reduction_sum_responses", "Distributed softmax sum reduction responses", "messages", 1},
        {"sfu_reduction_transport_received", "Explicit-NoC reduction messages delivered to SFU", "messages", 1},
        {"sfu_reduction_transport_stale_dropped", "Explicit-NoC stale or duplicate reduction messages dropped", "messages", 1},
        {"sfu_reduction_transport_inbox_high_water", "Explicit-NoC reduction response inbox high-water mark", "messages", 1},
        {"sfu_reduction_transport_latency_cycles", "Explicit-NoC reduction message transport latency", "cycles", 1},
        {"sfu_credit_stalls", "SFU credit stalls", "stalls", 1},
        {"sfu_cross_tile_wait_cycles", "SFU cross-tile wait cycles", "cycles", 1},
        {"sfu_retry_events", "SFU retry events", "events", 1})

    SFU(ComponentId_t id, SST::Params& params);
    ~SFU() override = default;

    void finish() override;

    bool issueSoftmaxTile(uint64_t descAddr, uint64_t tag) override;
    bool issuePrimitive(uint64_t descAddr, uint64_t tag) override;
    bool issuePrimitiveBatch(uint64_t descAddr, uint64_t tag) override;
    bool issueJob(uint64_t descAddr, uint64_t tag) override;
    bool wait(uint64_t tag, uint64_t* status) override;
    void bindGlobalMemory(GlobalMemoryAPI* globalMem) override;
    void setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores) override;

private:
    enum class SoftmaxJobStage : uint8_t {
        None,
        MaxSubmitted,
        SumSubmitted,
        Complete,
    };

    enum class DistributedReductionTransport : uint8_t {
        Shared,
        ModeledNoC,
        ExplicitNoC,
    };

    using DistributedReductionResponseInboxKey =
        std::tuple<uint64_t,
                   uint64_t,
                   uint32_t,
                   uint32_t,
                   uint32_t,
                   ReductionTransportMessageKind>;

    struct SoftmaxOpState {
        SFUSoftmaxTileDesc desc;
        uint64_t descAddr;
        uint64_t tag;
        SFUStatus status;
        std::vector<SFUTileRowStats> rowStats;
        std::vector<float> inputTile;
        bool normalizeReady;
    };

    struct PrimitiveOpState {
        SFUPrimitiveDesc desc;
        uint64_t descAddr;
        uint64_t tag;
        SFUStatus status;
        std::vector<float> input0;
        std::vector<float> output;
    };

    struct PrimitiveBatchOpState {
        SFUPrimitiveBatchDesc desc;
        uint64_t descAddr;
        uint64_t tag;
        SFUStatus status;
        uint64_t processedElems;
    };

    struct JobOpState {
        SFUJobDesc desc;
        uint64_t descAddr;
        uint64_t tag;
        SFUStatus status;
        uint64_t processedElems;
        SoftmaxJobStage stage;
        uint32_t workerSlot;
        uint32_t colBegin;
        uint32_t colEnd;
        std::vector<double> localMax;
        std::vector<double> localSum;
        std::vector<uint8_t> maxResponseSeen;
        std::vector<uint8_t> sumResponseSeen;
        bool distributedAbortObserved;
    };

    bool readSoftmaxDescriptor(uint64_t descAddr, SFUSoftmaxTileDesc* desc);
    SFUStatus validateSoftmaxDescriptor(const SFUSoftmaxTileDesc& desc) const;
    bool readInputTile(const SFUSoftmaxTileDesc& desc, std::vector<float>* values);
    void computeTileStats(SoftmaxOpState* state);
    bool mergeTileStats(SoftmaxOpState* state);
    bool tileGlobalStatsReady(const SoftmaxOpState& state) const;
    bool normalizeTile(SoftmaxOpState* state);
    bool readPrimitiveDescriptor(uint64_t descAddr, SFUPrimitiveDesc* desc);
    bool readPrimitiveBatchDescriptor(uint64_t descAddr, SFUPrimitiveBatchDesc* desc);
    SFUStatus validatePrimitiveDescriptor(const SFUPrimitiveDesc& desc) const;
    SFUStatus validatePrimitiveBatchDescriptor(const SFUPrimitiveBatchDesc& desc) const;
    bool readPrimitiveInput(const SFUPrimitiveDesc& desc, std::vector<float>* values);
    bool executePrimitive(PrimitiveOpState* state);
    bool writePrimitiveOutput(const SFUPrimitiveDesc& desc, const std::vector<float>& values);
    bool executePrimitiveDesc(uint64_t descAddr, uint64_t tag, uint64_t* processedElems);
    uint64_t primitiveProcessedElems(const SFUPrimitiveDesc& desc) const;
    bool readJobDescriptor(uint64_t descAddr, SFUJobDesc* desc);
    SFUStatus validateJobDescriptor(const SFUJobDesc& desc) const;
    bool executeJob(JobOpState* state);
    bool executeSoftmaxRowJob(JobOpState* state);
    bool executeDistributedSoftmaxRowJob(JobOpState* state);
    bool advanceDistributedSoftmaxJob(JobOpState* state);
    void abortDistributedSoftmaxJob(JobOpState* state);
    void observeDistributedSoftmaxJobAbort(JobOpState* state);
    bool modeledDistributedReductionEnabled() const;
    bool explicitDistributedReductionEnabled() const;
    void recordDistributedReductionRequest(bool maxStage);
    void recordDistributedReductionResponse(bool maxStage);
    void handleReductionTransportMessage(const ReductionTransportMessage& message);
    void clearDistributedReductionResponseInbox(const JobOpState& state);

    uint32_t coreId_;
    uint32_t activeWorkerCores_;
    uint32_t maxInflight_;
    uint32_t statsLatency_;
    uint32_t mergeLatency_;
    uint32_t normalizeLatency_;
    uint32_t inflight_;
    int verbose_;
    DistributedReductionTransport distributedReductionTransport_;
    std::map<DistributedReductionResponseInboxKey, ReductionTransportMessage>
        distributedReductionResponseInbox_;
    uint64_t distributedReductionResponseInboxHighWater_ = 0;

    GlobalMemoryAPI* globalMem_;
    SST::Output output_;
    std::unordered_map<uint64_t, SoftmaxOpState> pendingSoftmaxOps_;
    std::unordered_map<uint64_t, PrimitiveOpState> pendingPrimitiveOps_;
    std::unordered_map<uint64_t, PrimitiveBatchOpState> pendingPrimitiveBatchOps_;
    std::unordered_map<uint64_t, JobOpState> pendingJobOps_;

    Statistic<uint64_t>* statOpsIssued_;
    Statistic<uint64_t>* statSoftmaxRows_;
    Statistic<uint64_t>* statSoftmaxTiles_;
    Statistic<uint64_t>* statJobSoftmaxMaxChunks_;
    Statistic<uint64_t>* statJobSoftmaxSumChunks_;
    Statistic<uint64_t>* statJobSoftmaxNormChunks_;
    Statistic<uint64_t>* statPrimitiveElems_;
    Statistic<uint64_t>* statPartialSubmits_;
    Statistic<uint64_t>* statPartialDone_;
    Statistic<uint64_t>* statReductionMaxRequests_;
    Statistic<uint64_t>* statReductionMaxResponses_;
    Statistic<uint64_t>* statReductionSumRequests_;
    Statistic<uint64_t>* statReductionSumResponses_;
    Statistic<uint64_t>* statReductionTransportReceived_;
    Statistic<uint64_t>* statReductionTransportStaleDropped_;
    Statistic<uint64_t>* statReductionTransportInboxHighWater_;
    Statistic<uint64_t>* statReductionTransportLatencyCycles_;
    Statistic<uint64_t>* statCreditStalls_;
    Statistic<uint64_t>* statCrossTileWaitCycles_;
    Statistic<uint64_t>* statRetryEvents_;
};

} // namespace Golem
} // namespace SST

#endif
