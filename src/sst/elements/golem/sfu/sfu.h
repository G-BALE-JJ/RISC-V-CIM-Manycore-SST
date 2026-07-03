#ifndef _H_GOLEM_SFU
#define _H_GOLEM_SFU

#include <cstdint>
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
        {"sfu_primitive_elems", "Logical primitive elements processed by SFU", "elements", 1},
        {"sfu_partial_submits", "Softmax partial stats submitted", "partials", 1},
        {"sfu_partial_done", "Softmax partial stats completed", "partials", 1},
        {"sfu_credit_stalls", "SFU credit stalls", "stalls", 1},
        {"sfu_cross_tile_wait_cycles", "SFU cross-tile wait cycles", "cycles", 1},
        {"sfu_retry_events", "SFU retry events", "events", 1})

    SFU(ComponentId_t id, SST::Params& params);
    ~SFU() override = default;

    void finish() override;

    bool issueSoftmaxTile(uint64_t descAddr, uint64_t tag) override;
    bool issuePrimitive(uint64_t descAddr, uint64_t tag) override;
    bool issuePrimitiveBatch(uint64_t descAddr, uint64_t tag) override;
    bool wait(uint64_t tag, uint64_t* status) override;
    void bindGlobalMemory(GlobalMemoryAPI* globalMem) override;
    void setCoreInfo(uint32_t coreId, uint32_t activeWorkerCores) override;

private:
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
    std::unordered_map<uint64_t, SoftmaxOpState> pendingSoftmaxOps_;
    std::unordered_map<uint64_t, PrimitiveOpState> pendingPrimitiveOps_;
    std::unordered_map<uint64_t, PrimitiveBatchOpState> pendingPrimitiveBatchOps_;

    Statistic<uint64_t>* statOpsIssued_;
    Statistic<uint64_t>* statSoftmaxRows_;
    Statistic<uint64_t>* statSoftmaxTiles_;
    Statistic<uint64_t>* statPrimitiveElems_;
    Statistic<uint64_t>* statPartialSubmits_;
    Statistic<uint64_t>* statPartialDone_;
    Statistic<uint64_t>* statCreditStalls_;
    Statistic<uint64_t>* statCrossTileWaitCycles_;
    Statistic<uint64_t>* statRetryEvents_;
};

} // namespace Golem
} // namespace SST

#endif
