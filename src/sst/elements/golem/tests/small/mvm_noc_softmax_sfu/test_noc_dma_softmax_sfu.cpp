#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <string>
#include <vector>

#include "core_bind.h"
#include "ex_instr.h"
#include "pipeline_config.h"
#include "../mvm_noc_int_array/golem_matmul_runtime.h"
#include "../mvm_noc_int_array/gemm_matmul_op.h"
#include "golem_softmax_sfu_runtime.h"

namespace {

constexpr uint64_t kPrimitiveElemStride = sizeof(float);
constexpr uint64_t kPrimitiveLocalInputOffset = LOCAL_DATA_BASE;
constexpr uint64_t kPrimitiveLocalGuardBytes = 0x40;
constexpr uint64_t kPrimitiveDefaultChunkElems = 4;
constexpr uint64_t kPrimitiveHbmDefaultChunkElems = 64;
constexpr uint64_t kPrimitiveChunkCapElems = 8192;
constexpr uint64_t kPrimitiveBatchMaxItems = 64;
constexpr uint64_t kPrimitiveTagBase = 0x19000000ULL;
constexpr uint32_t kPrimitiveFlagRepeatChunk = 0x1;

constexpr uint64_t primitive_max_local_chunk_elems() {
    return (GOLEM_GLOBAL_STRIDE_BYTES > (LOCAL_DATA_BASE + kPrimitiveLocalGuardBytes))
        ? ((GOLEM_GLOBAL_STRIDE_BYTES - LOCAL_DATA_BASE - kPrimitiveLocalGuardBytes) /
           (2 * kPrimitiveElemStride))
        : 0;
}

int64_t read_i64_env_or_default(const char* name, int64_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
        return default_value;
    }
    return static_cast<int64_t>(parsed);
}

const char* read_string_env_or_default(const char* name, const char* default_value) {
    const char* raw = std::getenv(name);
    return (raw == nullptr || raw[0] == '\0') ? default_value : raw;
}

golem_matmul_op_desc_t make_matmul_desc_from_env() {
    return {
        .m = read_i64_env_or_default("GOLEM_MATMUL_M", GEMM_M),
        .n = read_i64_env_or_default("GOLEM_MATMUL_N", GEMM_N),
        .k = read_i64_env_or_default("GOLEM_MATMUL_K", GEMM_K),
        .block_m = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_M", TILE_M),
        .block_n = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_N", TILE_N_MAX),
        .block_k = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_K", TILE_K),
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
        .transpose_a = 0,
        .transpose_b = 0,
    };
}

uint64_t fp32_to_reg(float value) {
    uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return static_cast<uint64_t>(raw);
}

float fp32_from_reg(uint64_t raw) {
    const uint32_t raw32 = static_cast<uint32_t>(raw & 0xffffffffu);
    float value = 0.0f;
    std::memcpy(&value, &raw32, sizeof(value));
    return value;
}

void write_sfu_primitive_desc_to_gm(uint64_t desc_gm_addr, const SFUPrimitiveDesc& desc) {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(&desc);
    constexpr size_t kWords = sizeof(SFUPrimitiveDesc) / sizeof(uint64_t);
    for (size_t i = 0; i < kWords; ++i) {
        reg2gm(words[i], desc_gm_addr + static_cast<uint64_t>(i) * sizeof(uint64_t));
    }
}

void write_sfu_primitive_batch_desc_to_gm(uint64_t desc_gm_addr, const SFUPrimitiveBatchDesc& desc) {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(&desc);
    constexpr size_t kWords = sizeof(SFUPrimitiveBatchDesc) / sizeof(uint64_t);
    for (size_t i = 0; i < kWords; ++i) {
        reg2gm(words[i], desc_gm_addr + static_cast<uint64_t>(i) * sizeof(uint64_t));
    }
}

float primitive_input_value(SFUPrimitiveOp op, uint64_t elem_index) {
    switch (op) {
    case SFUPrimitiveOp::EXP: {
        const int centered = static_cast<int>(elem_index % 9) - 4;
        return static_cast<float>(centered) * 0.25f;
    }
    case SFUPrimitiveOp::LOG:
        return 0.5f + static_cast<float>(elem_index % 32) * 0.125f;
    case SFUPrimitiveOp::RECIPROCAL: {
        const int centered = static_cast<int>(elem_index % 15) - 7;
        return centered == 0 ? 0.5f : static_cast<float>(centered) * 0.5f;
    }
    case SFUPrimitiveOp::RSQRT:
        return 0.25f + static_cast<float>(elem_index % 16) * 0.125f;
    case SFUPrimitiveOp::TANH: {
        const int centered = static_cast<int>(elem_index % 17) - 8;
        return static_cast<float>(centered) * 0.25f;
    }
    case SFUPrimitiveOp::SIGMOID: {
        const int centered = static_cast<int>(elem_index % 19) - 9;
        return static_cast<float>(centered) * 0.25f;
    }
    default:
        return 0.0f;
    }
}

float primitive_expected_value(SFUPrimitiveOp op, float input) {
    switch (op) {
    case SFUPrimitiveOp::EXP:
        return static_cast<float>(std::exp(static_cast<double>(input)));
    case SFUPrimitiveOp::LOG:
        return static_cast<float>(std::log(static_cast<double>(input)));
    case SFUPrimitiveOp::RECIPROCAL:
        return 1.0f / input;
    case SFUPrimitiveOp::RSQRT:
        return 1.0f / static_cast<float>(std::sqrt(static_cast<double>(input)));
    case SFUPrimitiveOp::TANH:
        return static_cast<float>(std::tanh(static_cast<double>(input)));
    case SFUPrimitiveOp::SIGMOID:
        return 1.0f / (1.0f + static_cast<float>(std::exp(-static_cast<double>(input))));
    default:
        return 0.0f;
    }
}

const char* primitive_op_name(SFUPrimitiveOp op) {
    switch (op) {
    case SFUPrimitiveOp::EXP:
        return "EXP";
    case SFUPrimitiveOp::LOG:
        return "LOG";
    case SFUPrimitiveOp::RECIPROCAL:
        return "RECIPROCAL";
    case SFUPrimitiveOp::RSQRT:
        return "RSQRT";
    case SFUPrimitiveOp::TANH:
        return "TANH";
    case SFUPrimitiveOp::SIGMOID:
        return "SIGMOID";
    default:
        return "UNKNOWN";
    }
}

std::string normalize_primitive_op_token(const std::string& raw) {
    size_t begin = 0;
    while (begin < raw.size() && std::isspace(static_cast<unsigned char>(raw[begin]))) {
        ++begin;
    }
    size_t end = raw.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(raw[end - 1]))) {
        --end;
    }
    std::string token;
    token.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        token.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(raw[i]))));
    }
    return token;
}

bool parse_sfu_primitive_hbm_op_token(const std::string& token, SFUPrimitiveOp* op) {
    if (token == "EXP") {
        *op = SFUPrimitiveOp::EXP;
        return true;
    }
    if (token == "LOG") {
        *op = SFUPrimitiveOp::LOG;
        return true;
    }
    if (token == "RECIPROCAL" || token == "RECIP") {
        *op = SFUPrimitiveOp::RECIPROCAL;
        return true;
    }
    if (token == "RSQRT") {
        *op = SFUPrimitiveOp::RSQRT;
        return true;
    }
    if (token == "TANH") {
        *op = SFUPrimitiveOp::TANH;
        return true;
    }
    if (token == "SIGMOID") {
        *op = SFUPrimitiveOp::SIGMOID;
        return true;
    }
    return false;
}

bool parse_sfu_primitive_hbm_ops(const char* raw_ops, std::vector<SFUPrimitiveOp>* ops) {
    ops->clear();
    const std::string raw = raw_ops == nullptr ? "EXP" : raw_ops;
    size_t begin = 0;
    while (begin <= raw.size()) {
        const size_t end = raw.find_first_of(",;", begin);
        const std::string token = normalize_primitive_op_token(
            raw.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (!token.empty()) {
            if (token == "ALL") {
                ops->push_back(SFUPrimitiveOp::EXP);
                ops->push_back(SFUPrimitiveOp::LOG);
                ops->push_back(SFUPrimitiveOp::RECIPROCAL);
                ops->push_back(SFUPrimitiveOp::RSQRT);
                ops->push_back(SFUPrimitiveOp::TANH);
                ops->push_back(SFUPrimitiveOp::SIGMOID);
            } else {
                SFUPrimitiveOp op = SFUPrimitiveOp::EXP;
                if (!parse_sfu_primitive_hbm_op_token(token, &op)) {
                    std::fprintf(stderr, "[SFU-HBM-PRIMITIVE] unsupported op token=%s\n", token.c_str());
                    return false;
                }
                ops->push_back(op);
            }
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (ops->empty()) {
        ops->push_back(SFUPrimitiveOp::EXP);
    }
    return true;
}

std::string join_primitive_ops(const std::vector<SFUPrimitiveOp>& ops) {
    std::string joined;
    for (size_t i = 0; i < ops.size(); ++i) {
        if (i != 0) {
            joined += ",";
        }
        joined += primitive_op_name(ops[i]);
    }
    return joined;
}

bool should_validate_primitive_elem(uint64_t local_index, uint64_t elem_count) {
    if (elem_count <= 1024) {
        return true;
    }
    return local_index == 0 ||
           local_index + 1 == elem_count ||
           local_index == elem_count / 2 ||
           (local_index % 257) == 0;
}

uint64_t primitive_input_gm(int executor_core_id) {
    return gm_addr(executor_core_id, kPrimitiveLocalInputOffset);
}

uint64_t primitive_output_gm(int executor_core_id, uint64_t chunk_elems) {
    return primitive_input_gm(executor_core_id) + chunk_elems * kPrimitiveElemStride;
}

uint64_t primitive_batch_slot_gm(int executor_core_id, uint64_t chunk_elems, uint64_t item_index) {
    const uint64_t slot_bytes = 2 * chunk_elems * kPrimitiveElemStride;
    return primitive_input_gm(executor_core_id) + item_index * slot_bytes;
}

uint64_t primitive_batch_output_gm(int executor_core_id, uint64_t chunk_elems, uint64_t item_index) {
    return primitive_batch_slot_gm(executor_core_id, chunk_elems, item_index) +
           chunk_elems * kPrimitiveElemStride;
}

uint64_t primitive_batch_max_items(uint64_t chunk_elems) {
    if (chunk_elems == 0) {
        return 0;
    }
    const uint64_t usable_bytes =
        GOLEM_GLOBAL_STRIDE_BYTES > (LOCAL_DATA_BASE + kPrimitiveLocalGuardBytes)
            ? (GOLEM_GLOBAL_STRIDE_BYTES - LOCAL_DATA_BASE - kPrimitiveLocalGuardBytes)
            : 0;
    const uint64_t slot_bytes = 2 * chunk_elems * kPrimitiveElemStride;
    if (slot_bytes == 0) {
        return 0;
    }
    uint64_t items = usable_bytes / slot_bytes;
    if (items > kPrimitiveBatchMaxItems) {
        items = kPrimitiveBatchMaxItems;
    }
    return items;
}

void prepare_sfu_primitive_input_once(int executor_core_id, SFUPrimitiveOp op, uint64_t chunk_elems) {
    const uint64_t input_gm = primitive_input_gm(executor_core_id);
    const uint64_t chunk_bytes = chunk_elems * sizeof(float);

    std::vector<float> input_values(static_cast<size_t>(chunk_elems));
    for (uint64_t i = 0; i < chunk_elems; ++i) {
        input_values[static_cast<size_t>(i)] = primitive_input_value(op, i);
    }
    set_len(chunk_bytes);
    mm2gm(input_values.data(), input_gm);
}

bool issue_sfu_primitive_chunk(int executor_core_id,
                               SFUPrimitiveOp op,
                               uint64_t chunk_capacity_elems,
                               uint64_t elem_count,
                               uint64_t processed_elem_count,
                               uint64_t tag) {
    const uint64_t desc_gm = gm_addr(executor_core_id, LOCAL_LAYOUT.tmp);
    const uint64_t input_gm = primitive_input_gm(executor_core_id);
    const uint64_t output_gm = primitive_output_gm(executor_core_id, chunk_capacity_elems);
    const uint32_t primitive_flags_for_processed_elems =
        processed_elem_count > elem_count ? kPrimitiveFlagRepeatChunk : 0;

    const SFUPrimitiveDesc desc = {
        .job_id = tag,
        .input0_gm_addr = input_gm,
        .input1_gm_addr = processed_elem_count,
        .output_gm_addr = output_gm,
        .op = static_cast<uint32_t>(op),
        .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
        .elem_count = static_cast<uint32_t>(elem_count),
        .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
        .input1_stride_bytes = 0,
        .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
        .flags = primitive_flags_for_processed_elems,
        .approx_mode = 0,
    };
    write_sfu_primitive_desc_to_gm(desc_gm, desc);

    sfu_primitive(desc_gm, tag);
    const uint64_t status = sfu_primitive_wait(tag);
    if (status != 0) {
        std::fprintf(stderr, "[SFU-PRIMITIVE] wait failed tag=%llu status=%llu\n",
                     static_cast<unsigned long long>(tag),
                     static_cast<unsigned long long>(status));
        return false;
    }
    return true;
}

bool validate_sfu_primitive_output(int executor_core_id,
                                   SFUPrimitiveOp op,
                                   uint64_t chunk_capacity_elems,
                                   uint64_t elem_count) {
    const uint64_t output_gm = primitive_output_gm(executor_core_id, chunk_capacity_elems);
    const uint64_t chunk_bytes = elem_count * sizeof(float);

    std::vector<float> output_values(static_cast<size_t>(elem_count), 0.0f);
    set_len(chunk_bytes);
    gm2mm(output_values.data(), output_gm);

    for (uint64_t i = 0; i < elem_count; ++i) {
        if (!should_validate_primitive_elem(i, elem_count)) {
            continue;
        }
        const float input = primitive_input_value(op, i);
        const float expected = primitive_expected_value(op, input);
        const float got = output_values[static_cast<size_t>(i)];
        const float diff = std::fabs(got - expected);
        const float tol = 1.0e-5f + 1.0e-5f * std::fabs(expected);
        if (!std::isfinite(got) || diff > tol) {
            std::fprintf(stderr,
                         "[SFU-PRIMITIVE] mismatch op=%u idx=%llu got=%g expected=%g diff=%g tol=%g\n",
                         static_cast<uint32_t>(op),
                         static_cast<unsigned long long>(i),
                         got,
                         expected,
                         diff,
                         tol);
            return false;
        }
    }
    return true;
}

bool validate_sfu_primitive_output_from_values_at_gm(uint64_t output_gm,
                                                     SFUPrimitiveOp op,
                                                     const std::vector<float>& input_values,
                                                     uint64_t elem_count,
                                                     uint64_t global_elem_base) {
    const uint64_t chunk_bytes = elem_count * sizeof(float);

    std::vector<float> output_values(static_cast<size_t>(elem_count), 0.0f);
    set_len(chunk_bytes);
    gm2mm(output_values.data(), output_gm);

    for (uint64_t i = 0; i < elem_count; ++i) {
        if (!should_validate_primitive_elem(i, elem_count)) {
            continue;
        }
        const float input = input_values[static_cast<size_t>(i)];
        const float expected = primitive_expected_value(op, input);
        const float got = output_values[static_cast<size_t>(i)];
        const float diff = std::fabs(got - expected);
        const float tol = 1.0e-5f + 1.0e-5f * std::fabs(expected);
        if (!std::isfinite(got) || diff > tol) {
            std::fprintf(stderr,
                         "[SFU-HBM-PRIMITIVE] mismatch op=%u global_idx=%llu local_idx=%llu input=%g got=%g expected=%g diff=%g tol=%g\n",
                         static_cast<uint32_t>(op),
                         static_cast<unsigned long long>(global_elem_base + i),
                         static_cast<unsigned long long>(i),
                         input,
                         got,
                         expected,
                         diff,
                         tol);
            return false;
        }
    }
    return true;
}

bool validate_sfu_primitive_output_from_values(int executor_core_id,
                                               SFUPrimitiveOp op,
                                               uint64_t chunk_capacity_elems,
                                               const std::vector<float>& input_values,
                                               uint64_t elem_count,
                                               uint64_t global_elem_base) {
    const uint64_t output_gm = primitive_output_gm(executor_core_id, chunk_capacity_elems);
    return validate_sfu_primitive_output_from_values_at_gm(
        output_gm, op, input_values, elem_count, global_elem_base);
}

uint64_t primitive_chunk_elems(uint64_t requested_total,
                               int64_t requested_chunk,
                               uint64_t default_chunk_elems) {
    const uint64_t max_local = primitive_max_local_chunk_elems();
    if (max_local == 0) {
        return 0;
    }
    uint64_t chunk = requested_chunk > 0
        ? static_cast<uint64_t>(requested_chunk)
        : default_chunk_elems;
    if (chunk > requested_total) {
        chunk = requested_total;
    }
    if (chunk > kPrimitiveChunkCapElems) {
        chunk = kPrimitiveChunkCapElems;
    }
    if (chunk > max_local) {
        chunk = max_local;
    }
    return chunk;
}

uint64_t primitive_smoke_chunk_elems(uint64_t requested_total, int64_t requested_chunk) {
    return primitive_chunk_elems(requested_total, requested_chunk, kPrimitiveDefaultChunkElems);
}

uint64_t primitive_hbm_chunk_elems(uint64_t requested_total, int64_t requested_chunk) {
    return primitive_chunk_elems(requested_total, requested_chunk, kPrimitiveHbmDefaultChunkElems);
}

bool run_scaled_sfu_primitive_case(int executor_core_id,
                                   SFUPrimitiveOp op,
                                   uint64_t total_elems,
                                   uint64_t chunk_elems,
                                   uint64_t tag_base,
                                   uint64_t* chunks_out,
                                   uint64_t* processed_out) {
    prepare_sfu_primitive_input_once(executor_core_id, op, chunk_elems);

    const uint64_t elems_this_chunk =
        (total_elems < chunk_elems) ? total_elems : chunk_elems;
    const uint64_t chunks = (total_elems + chunk_elems - 1) / chunk_elems;
    if (!issue_sfu_primitive_chunk(
            executor_core_id, op, chunk_elems, elems_this_chunk, total_elems, tag_base + 1)) {
        return false;
    }
    const uint64_t validation_elems =
        (total_elems < chunk_elems) ? total_elems : chunk_elems;
    if (!validate_sfu_primitive_output(executor_core_id, op, chunk_elems, validation_elems)) {
        return false;
    }
    if (chunks_out != nullptr) {
        *chunks_out = chunks;
    }
    if (processed_out != nullptr) {
        *processed_out = total_elems;
    }
    return true;
}

uint64_t primitive_hbm_available_bytes(const GemmTaskDescriptor& desc) {
    const uint64_t node_base = node_base_addr(desc.data_node_idx);
    if (desc.c_base_mm < node_base || desc.c_base_mm >= node_base + OFF_GEMM_BIAS_BASE) {
        return 0;
    }
    return node_base + OFF_GEMM_BIAS_BASE - desc.c_base_mm;
}

bool run_hbm_stream_sfu_primitive_case(int executor_core_id,
                                       uint64_t input_hbm_base,
                                       SFUPrimitiveOp op,
                                       uint64_t total_elems,
                                       uint64_t chunk_elems,
                                       uint64_t tag_base,
                                       uint64_t* chunks_out,
                                       uint64_t* hbm_read_bytes_out,
                                       uint64_t* hbm_write_bytes_out) {
    uint64_t chunks = 0;
    uint64_t hbm_read_bytes = 0;
    uint64_t hbm_write_bytes = 0;
    uint64_t processed_elems = 0;
    while (processed_elems < total_elems) {
        const uint64_t remaining = total_elems - processed_elems;
        const uint64_t elems_this_chunk =
            remaining < chunk_elems ? remaining : chunk_elems;
        const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
        const uint64_t hbm_addr = input_hbm_base + processed_elems * sizeof(float);
        const uint64_t input_gm = primitive_input_gm(executor_core_id);
        const uint64_t output_gm = primitive_output_gm(executor_core_id, chunk_elems);
        const uint64_t tag = tag_base + chunks + 1;

        dma_remote_load_to_gm(executor_core_id, hbm_addr, input_gm, chunk_bytes);
        hbm_read_bytes += chunk_bytes;

        std::vector<float> input_values(static_cast<size_t>(elems_this_chunk), 0.0f);
        set_len(chunk_bytes);
        gm2mm(input_values.data(), input_gm);

        if (!issue_sfu_primitive_chunk(
                executor_core_id, op, chunk_elems, elems_this_chunk, elems_this_chunk, tag)) {
            return false;
        }
        if (!validate_sfu_primitive_output_from_values(
                executor_core_id, op, chunk_elems, input_values, elems_this_chunk, processed_elems)) {
            return false;
        }

        set_len(chunk_bytes);
        remote_store(output_gm, hbm_addr);
        hbm_write_bytes += chunk_bytes;

        processed_elems += elems_this_chunk;
        ++chunks;
    }

    if (chunks_out != nullptr) {
        *chunks_out = chunks;
    }
    if (hbm_read_bytes_out != nullptr) {
        *hbm_read_bytes_out = hbm_read_bytes;
    }
    if (hbm_write_bytes_out != nullptr) {
        *hbm_write_bytes_out = hbm_write_bytes;
    }
    return true;
}

bool run_hbm_stream_sfu_primitive_batch_case(int executor_core_id,
                                             uint64_t input_hbm_base,
                                             SFUPrimitiveOp op,
                                             uint64_t total_elems,
                                             uint64_t chunk_elems,
                                             uint64_t tag_base,
                                             uint64_t* chunks_out,
                                             uint64_t* hbm_read_bytes_out,
                                             uint64_t* hbm_write_bytes_out) {
    const uint64_t max_batch_items = primitive_batch_max_items(chunk_elems);
    if (max_batch_items == 0) {
        std::fprintf(stderr, "[SFU-HBM-PRIMITIVE] invalid primitive batch capacity\n");
        return false;
    }

    uint64_t chunks = 0;
    uint64_t batches = 0;
    uint64_t hbm_read_bytes = 0;
    uint64_t hbm_write_bytes = 0;
    uint64_t processed_elems = 0;
    const uint64_t batch_desc_gm = gm_addr(executor_core_id, LOCAL_LAYOUT.tmp);
    const uint64_t child_desc_array_gm =
        batch_desc_gm + static_cast<uint64_t>(sizeof(SFUPrimitiveBatchDesc));

    while (processed_elems < total_elems) {
        const uint64_t remaining_chunks =
            (total_elems - processed_elems + chunk_elems - 1) / chunk_elems;
        const uint64_t batch_items =
            remaining_chunks < max_batch_items ? remaining_chunks : max_batch_items;
        const uint64_t batch_tag = tag_base + batches + 1;

        std::vector<uint64_t> elem_counts(static_cast<size_t>(batch_items), 0);
        std::vector<uint64_t> hbm_addrs(static_cast<size_t>(batch_items), 0);
        std::vector<uint64_t> output_gms(static_cast<size_t>(batch_items), 0);
        std::vector<std::vector<float>> input_values(static_cast<size_t>(batch_items));

        uint64_t batch_processed_base = processed_elems;
        for (uint64_t item = 0; item < batch_items; ++item) {
            const uint64_t remaining = total_elems - processed_elems;
            const uint64_t elems_this_chunk =
                remaining < chunk_elems ? remaining : chunk_elems;
            const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
            const uint64_t hbm_addr = input_hbm_base + processed_elems * sizeof(float);
            const uint64_t input_gm =
                primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
            const uint64_t output_gm =
                primitive_batch_output_gm(executor_core_id, chunk_elems, item);

            dma_remote_load_to_gm(executor_core_id, hbm_addr, input_gm, chunk_bytes);
            hbm_read_bytes += chunk_bytes;

            input_values[static_cast<size_t>(item)].assign(
                static_cast<size_t>(elems_this_chunk), 0.0f);
            set_len(chunk_bytes);
            gm2mm(input_values[static_cast<size_t>(item)].data(), input_gm);

            const SFUPrimitiveDesc child_desc = {
                .job_id = batch_tag + item + 1,
                .input0_gm_addr = input_gm,
                .input1_gm_addr = elems_this_chunk,
                .output_gm_addr = output_gm,
                .op = static_cast<uint32_t>(op),
                .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                .elem_count = static_cast<uint32_t>(elems_this_chunk),
                .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                .input1_stride_bytes = 0,
                .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                .flags = 0,
                .approx_mode = 0,
            };
            write_sfu_primitive_desc_to_gm(
                child_desc_array_gm + item * static_cast<uint64_t>(sizeof(SFUPrimitiveDesc)),
                child_desc);

            elem_counts[static_cast<size_t>(item)] = elems_this_chunk;
            hbm_addrs[static_cast<size_t>(item)] = hbm_addr;
            output_gms[static_cast<size_t>(item)] = output_gm;
            processed_elems += elems_this_chunk;
            ++chunks;
        }

        const SFUPrimitiveBatchDesc batch_desc = {
            .job_id = batch_tag,
            .desc_array_gm_addr = child_desc_array_gm,
            .desc_count = static_cast<uint32_t>(batch_items),
            .flags = 0,
            .reserved0 = 0,
        };
        write_sfu_primitive_batch_desc_to_gm(batch_desc_gm, batch_desc);
        sfu_primitive_batch(batch_desc_gm, batch_tag);
        const uint64_t status = sfu_primitive_batch_wait(batch_tag);
        if (status != 0) {
            std::fprintf(stderr, "[SFU-HBM-PRIMITIVE] batch wait failed tag=%llu status=%llu\n",
                         static_cast<unsigned long long>(batch_tag),
                         static_cast<unsigned long long>(status));
            return false;
        }

        for (uint64_t item = 0; item < batch_items; ++item) {
            const uint64_t elems_this_chunk = elem_counts[static_cast<size_t>(item)];
            const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
            const uint64_t global_elem_base = batch_processed_base + item * chunk_elems;
            if (!validate_sfu_primitive_output_from_values_at_gm(
                    output_gms[static_cast<size_t>(item)], op,
                    input_values[static_cast<size_t>(item)],
                    elems_this_chunk, global_elem_base)) {
                return false;
            }
            set_len(chunk_bytes);
            remote_store(output_gms[static_cast<size_t>(item)],
                         hbm_addrs[static_cast<size_t>(item)]);
            hbm_write_bytes += chunk_bytes;
        }
        ++batches;
    }

    if (chunks_out != nullptr) {
        *chunks_out = chunks;
    }
    if (hbm_read_bytes_out != nullptr) {
        *hbm_read_bytes_out = hbm_read_bytes;
    }
    if (hbm_write_bytes_out != nullptr) {
        *hbm_write_bytes_out = hbm_write_bytes;
    }
    return true;
}

int run_sfu_primitive_hbm_stream_for_core(int executor_core_id,
                                          int requested_core_id,
                                          const MatmulRuntimeConfig& cfg) {
    if (requested_core_id != 0) {
        return 0;
    }

    int64_t requested_total = read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_ELEMS", 64);
    if (requested_total <= 0) {
        requested_total = 64;
    }
    const uint64_t total_elems = static_cast<uint64_t>(requested_total);
    const int64_t requested_chunk =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS", 0);
    const uint64_t chunk_elems = primitive_hbm_chunk_elems(total_elems, requested_chunk);
    if (chunk_elems == 0) {
        std::fprintf(stderr, "[SFU-HBM-PRIMITIVE] invalid primitive chunk capacity\n");
        return 1;
    }
    const char* raw_ops = read_string_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_OPS", "EXP");
    std::vector<SFUPrimitiveOp> ops;
    if (!parse_sfu_primitive_hbm_ops(raw_ops, &ops)) {
        return 1;
    }
    const int64_t batch_mode = read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_BATCH", 0);
    const std::string ops_label = join_primitive_ops(ops);
    const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, 0, cfg);
    const uint64_t total_bytes = total_elems * sizeof(float);
    const uint64_t required_bytes = total_bytes * static_cast<uint64_t>(ops.size());
    const uint64_t available_bytes = primitive_hbm_available_bytes(desc);
    if (available_bytes < required_bytes) {
        std::fprintf(stderr,
                     "[SFU-HBM-PRIMITIVE] requested bytes=%llu exceed available HBM C-region bytes=%llu\n",
                     static_cast<unsigned long long>(required_bytes),
                     static_cast<unsigned long long>(available_bytes));
        return 1;
    }
    const uint64_t hbm_init_write_bytes = required_bytes;

    std::printf("[SFU-HBM-PRIMITIVE] begin executor_core=%d ops=%s total_elems=%llu chunk_elems=%llu hbm_init_write_bytes=%llu\n",
                executor_core_id,
                ops_label.c_str(),
                static_cast<unsigned long long>(total_elems),
                static_cast<unsigned long long>(chunk_elems),
                static_cast<unsigned long long>(hbm_init_write_bytes));
    std::fflush(stdout);

    uint64_t total_chunks = 0;
    uint64_t hbm_read_bytes = 0;
    uint64_t hbm_write_bytes = 0;
    uint64_t processed_elems = 0;
    for (size_t op_index = 0; op_index < ops.size(); ++op_index) {
        uint64_t chunks = 0;
        uint64_t op_hbm_read_bytes = 0;
        uint64_t op_hbm_write_bytes = 0;
        const uint64_t input_hbm_base = desc.c_base_mm + static_cast<uint64_t>(op_index) * total_bytes;
        const uint64_t tag_base = kPrimitiveTagBase + 0x700000ULL +
                                  static_cast<uint64_t>(op_index) * 0x100000ULL;
        const bool ok = batch_mode != 0
            ? run_hbm_stream_sfu_primitive_batch_case(executor_core_id, input_hbm_base, ops[op_index],
                                                      total_elems, chunk_elems, tag_base,
                                                      &chunks,
                                                      &op_hbm_read_bytes,
                                                      &op_hbm_write_bytes)
            : run_hbm_stream_sfu_primitive_case(executor_core_id, input_hbm_base, ops[op_index],
                                                total_elems, chunk_elems, tag_base,
                                                &chunks,
                                                &op_hbm_read_bytes,
                                                &op_hbm_write_bytes);
        if (!ok) {
            return 1;
        }
        total_chunks += chunks;
        hbm_read_bytes += op_hbm_read_bytes;
        hbm_write_bytes += op_hbm_write_bytes;
        processed_elems += total_elems;
    }

    std::printf("[SOFTMAX] mode=sfu-primitive-hbm-stream executor_core=%d ops=%s total_elems=%llu chunk_elems=%llu chunks=%llu processed_elems=%llu hbm_init_write_bytes=%llu hbm_read_bytes=%llu hbm_write_bytes=%llu PASS\n",
                executor_core_id,
                ops_label.c_str(),
                static_cast<unsigned long long>(total_elems),
                static_cast<unsigned long long>(chunk_elems),
                static_cast<unsigned long long>(total_chunks),
                static_cast<unsigned long long>(processed_elems),
                static_cast<unsigned long long>(hbm_init_write_bytes),
                static_cast<unsigned long long>(hbm_read_bytes),
                static_cast<unsigned long long>(hbm_write_bytes));
    std::fflush(stdout);
    return 0;
}

int run_sfu_primitive_smoke_for_core(int executor_core_id, int requested_core_id) {
    if (requested_core_id != 0) {
        return 0;
    }

    int64_t requested_total = read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS", 4);
    if (requested_total <= 0) {
        requested_total = 4;
    }
    const uint64_t total_elems = static_cast<uint64_t>(requested_total);
    const int64_t requested_chunk =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS", 0);
    const uint64_t chunk_elems = primitive_smoke_chunk_elems(total_elems, requested_chunk);
    if (chunk_elems == 0) {
        std::fprintf(stderr, "[SFU-PRIMITIVE] invalid primitive chunk capacity\n");
        return 1;
    }
    std::printf("[SFU-PRIMITIVE] begin executor_core=%d total_elems=%llu chunk_elems=%llu ops=EXP,LOG,RECIPROCAL,RSQRT,TANH,SIGMOID\n",
                executor_core_id,
                static_cast<unsigned long long>(total_elems),
                static_cast<unsigned long long>(chunk_elems));
    std::fflush(stdout);

    uint64_t exp_chunks = 0;
    uint64_t exp_processed_elems = 0;
    if (!run_scaled_sfu_primitive_case(executor_core_id, SFUPrimitiveOp::EXP,
                                       total_elems, chunk_elems,
                                       kPrimitiveTagBase + 0x100000ULL,
                                       &exp_chunks, &exp_processed_elems)) {
        return 1;
    }
    uint64_t log_chunks = 0;
    uint64_t log_processed_elems = 0;
    if (!run_scaled_sfu_primitive_case(executor_core_id, SFUPrimitiveOp::LOG,
                                       total_elems, chunk_elems,
                                       kPrimitiveTagBase + 0x200000ULL,
                                       &log_chunks, &log_processed_elems)) {
        return 1;
    }
    uint64_t reciprocal_chunks = 0;
    uint64_t reciprocal_processed_elems = 0;
    if (!run_scaled_sfu_primitive_case(executor_core_id, SFUPrimitiveOp::RECIPROCAL,
                                       total_elems, chunk_elems,
                                       kPrimitiveTagBase + 0x300000ULL,
                                       &reciprocal_chunks,
                                       &reciprocal_processed_elems)) {
        return 1;
    }
    uint64_t rsqrt_chunks = 0;
    uint64_t rsqrt_processed_elems = 0;
    if (!run_scaled_sfu_primitive_case(executor_core_id, SFUPrimitiveOp::RSQRT,
                                       total_elems, chunk_elems,
                                       kPrimitiveTagBase + 0x400000ULL,
                                       &rsqrt_chunks, &rsqrt_processed_elems)) {
        return 1;
    }
    uint64_t tanh_chunks = 0;
    uint64_t tanh_processed_elems = 0;
    if (!run_scaled_sfu_primitive_case(executor_core_id, SFUPrimitiveOp::TANH,
                                       total_elems, chunk_elems,
                                       kPrimitiveTagBase + 0x500000ULL,
                                       &tanh_chunks, &tanh_processed_elems)) {
        return 1;
    }
    uint64_t sigmoid_chunks = 0;
    uint64_t sigmoid_processed_elems = 0;
    if (!run_scaled_sfu_primitive_case(executor_core_id, SFUPrimitiveOp::SIGMOID,
                                       total_elems, chunk_elems,
                                       kPrimitiveTagBase + 0x600000ULL,
                                       &sigmoid_chunks, &sigmoid_processed_elems)) {
        return 1;
    }

    const uint64_t chunks = exp_chunks + log_chunks + reciprocal_chunks +
                            rsqrt_chunks + tanh_chunks + sigmoid_chunks;
    const uint64_t processed_elems =
        exp_processed_elems + log_processed_elems + reciprocal_processed_elems +
        rsqrt_processed_elems + tanh_processed_elems + sigmoid_processed_elems;
    std::printf("[SOFTMAX] mode=sfu-primitive-smoke executor_core=%d ops=EXP,LOG,RECIPROCAL,RSQRT,TANH,SIGMOID total_elems=%llu chunk_elems=%llu chunks=%llu processed_elems=%llu PASS\n",
                executor_core_id,
                static_cast<unsigned long long>(total_elems),
                static_cast<unsigned long long>(chunk_elems),
                static_cast<unsigned long long>(chunks),
                static_cast<unsigned long long>(processed_elems));
    std::fflush(stdout);
    return 0;
}

int resolve_executor_core_from_argv_or_exit(int argc, char* argv[], int requested_core_id) {
    if (requested_core_id < 0 || requested_core_id >= TOTAL_CORES) {
        std::fprintf(stderr, "[ERROR] invalid requested core id=%d, TOTAL_CORES=%d\n",
                     requested_core_id, TOTAL_CORES);
        return -1;
    }
    if (argc >= 2) {
        bind_process_to_core(requested_core_id);
    }
    const int executor_core_id = sched_getcpu();
    if (executor_core_id < 0 || executor_core_id >= TOTAL_CORES) {
        std::fprintf(stderr, "[ERROR] invalid executor core id=%d, TOTAL_CORES=%d\n",
                     executor_core_id, TOTAL_CORES);
        return -1;
    }
    return executor_core_id;
}

int run_gemm_for_core(int executor_core_id, int worker_core_id, const golem_matmul_op_desc_t& op_desc) {
    const MatmulRuntimeConfig cfg = {
        .m = static_cast<int>(op_desc.m),
        .n = static_cast<int>(op_desc.n),
        .k = static_cast<int>(op_desc.k),
        .block_m = static_cast<int>(op_desc.block_m),
        .block_n = static_cast<int>(op_desc.block_n),
        .block_k = static_cast<int>(op_desc.block_k),
    };
    if (op_desc.layout != GOLEM_LAYOUT_ROW_MAJOR ||
        op_desc.transpose_a != 0 ||
        op_desc.transpose_b != 0 ||
        !validate_matmul_call(cfg)) {
        std::fprintf(stderr, "[ERROR] invalid SFU GEMM op descriptor\n");
        return 1;
    }
    if ((cfg.block_m % TILE_M) != 0 || (cfg.block_k % TILE_K) != 0 || cfg.block_n > TILE_N_MAX) {
        std::fprintf(stderr, "[ERROR] unsupported SFU GEMM tile shape\n");
        return 1;
    }

    const GemmTileRuntimeContext rt = make_gemm_runtime_context(executor_core_id);
    const int total_tasks = gemm_total_tasks(cfg);
    const int worker_slot = gemm_worker_slot_for_core(worker_core_id);
    if (worker_slot < 0 || worker_slot >= total_tasks) {
        return 0;
    }

    GemmKernelStats stats = {};
    for (int task_id = worker_slot; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, task_id, cfg);
        gemm_tiled<float>(executor_core_id, desc, rt, &stats);
    }
    return 0;
}

int run_gemm_interleaved_sfu_for_core(int executor_core_id,
                                      int worker_core_id,
                                      const golem_matmul_op_desc_t& op_desc,
                                      const golem_softmax_op_desc_t& softmax_desc,
                                      const MatmulRuntimeConfig& cfg,
                                      uint64_t job_id) {
    if (op_desc.layout != GOLEM_LAYOUT_ROW_MAJOR ||
        op_desc.transpose_a != 0 ||
        op_desc.transpose_b != 0 ||
        !validate_matmul_call(cfg)) {
        std::fprintf(stderr, "[ERROR] invalid interleaved SFU GEMM op descriptor\n");
        return 1;
    }
    if ((cfg.block_m % TILE_M) != 0 || (cfg.block_k % TILE_K) != 0 || cfg.block_n > TILE_N_MAX) {
        std::fprintf(stderr, "[ERROR] unsupported interleaved SFU GEMM tile shape\n");
        return 1;
    }

    const GemmTileRuntimeContext rt = make_gemm_runtime_context(executor_core_id);
    const int total_tasks = gemm_total_tasks(cfg);
    const int worker_slot = gemm_worker_slot_for_core(worker_core_id);
    if (worker_slot < 0 || worker_slot >= total_tasks) {
        return 0;
    }

    struct PendingTile {
        uint64_t tag;
        uint64_t output_hbm;
        uint64_t local_output_gm;
        uint64_t bytes;
    };
    std::vector<PendingTile> pending;

    const uint64_t desc_gm = gm_addr(executor_core_id, LOCAL_LAYOUT.tmp);
    GemmKernelStats stats = {};
    for (int task_id = worker_slot; task_id < total_tasks; task_id += ACTIVE_GEMM_CORES) {
        const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, task_id, cfg);
        gemm_tiled<float>(executor_core_id, desc, rt, &stats);

        const uint64_t tile_bytes =
            static_cast<uint64_t>(desc.block_m) * static_cast<uint64_t>(desc.block_n) * sizeof(float);
        const uint64_t tag = static_cast<uint64_t>(task_id) + 1;
        const golem_status_t sfu_status = golemRunSoftmaxSfuTileFromLocalAccum(
            &softmax_desc,
            executor_core_id,
            &cfg,
            &desc,
            rt.local_accum,
            rt.local_out,
            desc_gm,
            job_id,
            tag);
        if (sfu_status != GOLEM_STATUS_OK) {
            std::fprintf(stderr, "[SOFTMAX-SFU] local-accum issue failed on core %d task %d\n",
                         worker_core_id, task_id);
            return 1;
        }
        pending.push_back(PendingTile{
            .tag = tag,
            .output_hbm = desc.c_base_mm,
            .local_output_gm = rt.local_out,
            .bytes = tile_bytes,
        });
    }

    for (const PendingTile& tile : pending) {
        const golem_status_t wait_status = golemWaitSoftmaxSfuTileAndStore(
            tile.tag, tile.local_output_gm, tile.output_hbm, tile.bytes);
        if (wait_status != GOLEM_STATUS_OK) {
            std::fprintf(stderr, "[SOFTMAX-SFU] local-accum wait failed on core %d\n",
                         worker_core_id);
            return 1;
        }
    }
    return 0;
}

int read_requested_core_from_argv(int argc, char* argv[]) {
    if (argc < 2) {
        return 0;
    }
    return std::atoi(argv[1]);
}

int run_riscv_gemm_softmax_sfu(int argc, char* argv[]) {
    const int requested_core_id = read_requested_core_from_argv(argc, argv);
    const int executor_core_id = resolve_executor_core_from_argv_or_exit(argc, argv, requested_core_id);
    if (executor_core_id < 0) {
        return 1;
    }

    const golem_matmul_op_desc_t op_desc = make_matmul_desc_from_env();
    if (op_desc.dtype != GOLEM_DTYPE_FP32) {
        std::fprintf(stderr, "[SOFTMAX-SFU] only GOLEM_DTYPE_FP32 is supported\n");
        return 1;
    }

    const MatmulRuntimeConfig cfg = {
        .m = static_cast<int>(op_desc.m),
        .n = static_cast<int>(op_desc.n),
        .k = static_cast<int>(op_desc.k),
        .block_m = static_cast<int>(op_desc.block_m),
        .block_n = static_cast<int>(op_desc.block_n),
        .block_k = static_cast<int>(op_desc.block_k),
    };
    const golem_softmax_op_desc_t softmax_desc = {
        .outer = cfg.m,
        .dim = cfg.n,
        .axis = -1,
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    const uint64_t job_id =
        (static_cast<uint64_t>(cfg.m) << 48) ^
        (static_cast<uint64_t>(cfg.n) << 32) ^
        (static_cast<uint64_t>(cfg.block_m) << 16) ^
        static_cast<uint64_t>(cfg.block_n);

    const int64_t primitive_hbm_stream =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_STREAM", 0);
    if (primitive_hbm_stream != 0) {
        return run_sfu_primitive_hbm_stream_for_core(executor_core_id, requested_core_id, cfg);
    }

    const int64_t primitive_smoke = read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SMOKE", 0);
    if (primitive_smoke != 0) {
        return run_sfu_primitive_smoke_for_core(executor_core_id, requested_core_id);
    }

    const int64_t standalone_softmax = read_i64_env_or_default("GOLEM_SFU_STANDALONE_SOFTMAX", 0);
    if (standalone_softmax != 0) {
        if (requested_core_id == 0) {
            std::printf("[SOFTMAX] mode=sfu-standalone-softmax m=%d n=%d block_m=%d block_n=%d executor_core=%d\n",
                        cfg.m, cfg.n, cfg.block_m, cfg.block_n, executor_core_id);
            std::fflush(stdout);
        }
        const golem_status_t sfu_status = golemRunStandaloneSoftmaxSfuForCore(
            &softmax_desc, executor_core_id, requested_core_id, &cfg, job_id);
        if (sfu_status != GOLEM_STATUS_OK) {
            std::fprintf(stderr, "[SOFTMAX-SFU] standalone failed on core %d\n", requested_core_id);
            return 1;
        }
        return 0;
    }

    const int64_t interleave_gemm = read_i64_env_or_default("GOLEM_SFU_INTERLEAVE_GEMM", 0);
    if (interleave_gemm != 0) {
        if (requested_core_id == 0) {
            std::printf("[SOFTMAX] mode=sfu-interleaved-local-accum m=%d n=%d block_m=%d block_n=%d executor_core=%d\n",
                        cfg.m, cfg.n, cfg.block_m, cfg.block_n, executor_core_id);
            std::fflush(stdout);
        }
        return run_gemm_interleaved_sfu_for_core(
            executor_core_id, requested_core_id, op_desc, softmax_desc, cfg, job_id);
    }

    int status = run_gemm_for_core(executor_core_id, requested_core_id, op_desc);
    if (status != 0) {
        return status;
    }
    const int64_t skip_softmax = read_i64_env_or_default("GOLEM_SFU_SKIP_SOFTMAX", 0);
    if (skip_softmax != 0) {
        if (requested_core_id == 0) {
            std::printf("[SOFTMAX] mode=sfu-skip-softmax executor_core=%d\n", executor_core_id);
            std::fflush(stdout);
        }
        return 0;
    }

    if (requested_core_id == 0) {
        std::printf("[SOFTMAX] mode=sfu m=%d n=%d block_m=%d block_n=%d executor_core=%d\n",
                    cfg.m, cfg.n, cfg.block_m, cfg.block_n, executor_core_id);
        std::fflush(stdout);
    }

    const golem_status_t sfu_status = golemRunSoftmaxSfuForCore(
        &softmax_desc, executor_core_id, requested_core_id, &cfg, job_id);
    if (sfu_status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[SOFTMAX-SFU] failed on core %d\n", requested_core_id);
        return 1;
    }
    return 0;
}

int run_host_smoke() {
    const golem_softmax_op_desc_t desc = {
        .outer = 1,
        .dim = 4,
        .axis = -1,
        .dtype = GOLEM_DTYPE_FP32,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };
    if (desc.dtype != GOLEM_DTYPE_FP32 || desc.dim != 4) {
        return 1;
    }
    std::printf("[SOFTMAX-SFU-SELFTEST] PASS descriptor smoke\n");
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc > 1) {
        return run_riscv_gemm_softmax_sfu(argc, argv);
    }
    return run_host_smoke();
}
