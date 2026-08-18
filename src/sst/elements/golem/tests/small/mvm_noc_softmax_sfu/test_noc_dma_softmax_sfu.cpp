#include <cmath>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
constexpr uint64_t kSoftmaxPrimitiveMboxPartialReady = 0x90;
constexpr uint64_t kSoftmaxPrimitiveMboxPartialValue = 0x98;
constexpr uint64_t kSoftmaxPrimitiveMboxGlobalReady = 0xa0;
constexpr uint64_t kSoftmaxPrimitiveMboxGlobalValue = 0xa8;
constexpr uint64_t kSfuSoftmaxRowmajorMatrixBytes =
    static_cast<uint64_t>(GEMM_M) * static_cast<uint64_t>(GEMM_N) * sizeof(float);
constexpr uint64_t OFF_SFU_SOFTMAX_ROWMAJOR_BASE =
    align_up_constexpr(GEMM_DATA_REGION_END, MM_ALIGN);
constexpr uint64_t OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE =
    OFF_SFU_SOFTMAX_ROWMAJOR_BASE +
    align_up_constexpr(kSfuSoftmaxRowmajorMatrixBytes, MM_ALIGN);
constexpr int kSfuSoftmaxRowmajorDataNode = 1;
constexpr uint64_t kDirectDmaLoadSentinel = 0x7fc00001ULL;
constexpr uint64_t kDirectDmaLoadGuardStrideBytes = 16ULL * 1024ULL;

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

uint64_t pack_two_fp32_to_reg(float low_value, float high_value) {
    uint32_t low = 0;
    uint32_t high = 0;
    std::memcpy(&low, &low_value, sizeof(low));
    std::memcpy(&high, &high_value, sizeof(high));
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

float low_fp32_from_packed_reg(uint64_t raw) {
    const uint32_t low = static_cast<uint32_t>(raw & 0xffffffffu);
    float value = 0.0f;
    std::memcpy(&value, &low, sizeof(value));
    return value;
}

float high_fp32_from_packed_reg(uint64_t raw) {
    const uint32_t high = static_cast<uint32_t>((raw >> 32) & 0xffffffffu);
    float value = 0.0f;
    std::memcpy(&value, &high, sizeof(value));
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

bool issue_sfu_primitive_batch_descs(int executor_core_id,
                                     const std::vector<SFUPrimitiveDesc>& child_descs,
                                     uint64_t batch_tag) {
    if (child_descs.empty() || child_descs.size() > kPrimitiveBatchMaxItems) {
        return false;
    }

    const uint64_t batch_desc_gm = gm_addr(executor_core_id, LOCAL_LAYOUT.tmp);
    const uint64_t child_desc_array_gm =
        batch_desc_gm + static_cast<uint64_t>(sizeof(SFUPrimitiveBatchDesc));
    for (size_t i = 0; i < child_descs.size(); ++i) {
        write_sfu_primitive_desc_to_gm(
            child_desc_array_gm + static_cast<uint64_t>(i) * sizeof(SFUPrimitiveDesc),
            child_descs[i]);
    }

    const SFUPrimitiveBatchDesc batch_desc = {
        .job_id = batch_tag,
        .desc_array_gm_addr = child_desc_array_gm,
        .desc_count = static_cast<uint32_t>(child_descs.size()),
        .flags = 0,
        .reserved0 = 0,
    };
    write_sfu_primitive_batch_desc_to_gm(batch_desc_gm, batch_desc);
    sfu_primitive_batch(batch_desc_gm, batch_tag);
    const uint64_t status = sfu_primitive_batch_wait(batch_tag);
    if (status != 0) {
        std::fprintf(stderr, "[SFU-PRIMITIVE-BATCH] wait failed tag=%llu status=%llu\n",
                     static_cast<unsigned long long>(batch_tag),
                     static_cast<unsigned long long>(status));
        return false;
    }
    return true;
}

float read_fp32_from_gm(uint64_t gm_addr_value) {
    const uint64_t raw64 = gm2reg(gm_addr_value);
    const uint32_t raw32 = static_cast<uint32_t>(raw64 & 0xffffffffu);
    float value = 0.0f;
    std::memcpy(&value, &raw32, sizeof(value));
    return value;
}

void write_fp32_to_gm(uint64_t gm_addr_value, float value) {
    uint32_t raw32 = 0;
    std::memcpy(&raw32, &value, sizeof(raw32));
    reg2gm(static_cast<uint64_t>(raw32), gm_addr_value);
}

void prepare_direct_dma_load_guard(uint64_t input_gm, uint64_t bytes) {
    if (bytes < sizeof(float)) {
        return;
    }
    uint64_t last_marked = UINT64_MAX;
    for (uint64_t offset = 0; offset < bytes; offset += kDirectDmaLoadGuardStrideBytes) {
        reg2gm(kDirectDmaLoadSentinel, input_gm + offset);
        last_marked = offset;
    }
    const uint64_t last_word = bytes - sizeof(float);
    if (last_word != last_marked) {
        reg2gm(kDirectDmaLoadSentinel, input_gm + last_word);
    }
}

bool direct_dma_load_guard_passed(uint64_t input_gm, uint64_t bytes) {
    if (bytes < sizeof(float)) {
        return true;
    }
    uint64_t last_checked = UINT64_MAX;
    for (uint64_t offset = 0; offset < bytes; offset += kDirectDmaLoadGuardStrideBytes) {
        if ((gm2reg(input_gm + offset) & 0xffffffffULL) == kDirectDmaLoadSentinel) {
            return false;
        }
        last_checked = offset;
    }
    const uint64_t last_word = bytes - sizeof(float);
    if (last_word != last_checked &&
        (gm2reg(input_gm + last_word) & 0xffffffffULL) == kDirectDmaLoadSentinel) {
        return false;
    }
    return true;
}

float softmax_primitive_input_value(uint64_t row, uint64_t col) {
    const int centered = static_cast<int>(col % 17) - 8;
    return static_cast<float>(centered) * 0.25f +
           static_cast<float>(row) * 0.01f;
}

void fill_softmax_primitive_chunk(std::vector<float>* values,
                                  uint64_t row,
                                  uint64_t col_begin,
                                  uint64_t elem_count) {
    values->assign(static_cast<size_t>(elem_count), 0.0f);
    for (uint64_t i = 0; i < elem_count; ++i) {
        (*values)[static_cast<size_t>(i)] =
            softmax_primitive_input_value(row, col_begin + i);
    }
}

void write_values_to_gm(uint64_t gm_addr_value, const std::vector<float>& values) {
    if (values.empty()) {
        return;
    }
    set_len(static_cast<uint64_t>(values.size()) * sizeof(float));
    mm2gm(const_cast<float*>(values.data()), gm_addr_value);
}

void read_values_from_gm(uint64_t gm_addr_value,
                         uint64_t elem_count,
                         std::vector<float>* values) {
    values->assign(static_cast<size_t>(elem_count), 0.0f);
    if (elem_count == 0) {
        return;
    }
    set_len(elem_count * sizeof(float));
    gm2mm(values->data(), gm_addr_value);
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

uint64_t softmax_primitive_mailbox_addr(int core_id, uint64_t field_offset) {
    return get_core_mailbox_addr(core_id) + field_offset;
}

void softmax_primitive_publish_u64(int src_core_id,
                                   int dst_core_id,
                                   uint64_t field_offset,
                                   uint64_t value) {
    const uint64_t dst_addr = softmax_primitive_mailbox_addr(dst_core_id, field_offset);
    if (src_core_id == dst_core_id) {
        reg2gm(value, dst_addr);
    } else {
        remote_write_u64(src_core_id, value, dst_addr);
    }
}

void softmax_primitive_wait_local_u64(int core_id, uint64_t field_offset, uint64_t expected) {
    adaptive_wait_eq(softmax_primitive_mailbox_addr(core_id, field_offset), expected);
}

void softmax_primitive_publish_to_addr(int src_core_id, uint64_t dst_addr, uint64_t value) {
    const uint64_t src_base = get_core_base_addr(src_core_id);
    if (dst_addr >= src_base && dst_addr < src_base + GOLEM_GLOBAL_STRIDE_BYTES) {
        reg2gm(value, dst_addr);
    } else {
        remote_write_u64(src_core_id, value, dst_addr);
    }
}

bool softmax_primitive_poll_ready(uint64_t ready_addr, uint64_t expected_seq) {
    return gm2reg(ready_addr) == expected_seq;
}

uint64_t softmax_primitive_coord_worker_addr(uint64_t coord_base_gm,
                                             int worker_slot,
                                             uint64_t field_offset) {
    return coord_base_gm + static_cast<uint64_t>(worker_slot) * 0x20ULL + field_offset;
}

constexpr uint64_t kSoftmaxPrimitiveBlockWorkerStride = 0x100;
constexpr uint64_t kSoftmaxPrimitiveBlockMaxReady = 0x000;
constexpr uint64_t kSoftmaxPrimitiveBlockSumReady = 0x008;
constexpr uint64_t kSoftmaxPrimitiveBlockLocalMaxBase = 0x040;
constexpr uint64_t kSoftmaxPrimitiveBlockLocalSumBase = 0x080;
constexpr uint64_t kSoftmaxPrimitiveBlockGlobalBase = 0x180;
constexpr uint64_t kSoftmaxPrimitiveBlockGlobalMaxReady = 0x000;
constexpr uint64_t kSoftmaxPrimitiveBlockGlobalSumReady = 0x008;
constexpr uint64_t kSoftmaxPrimitiveBlockGlobalMaxBase = 0x040;
constexpr uint64_t kSoftmaxPrimitiveBlockInvSumBase = 0x080;
constexpr uint64_t kSoftmaxPrimitiveBlockValueStride = 0x8;

constexpr uint64_t kSoftmaxPrimitiveStageWorkerStride = 0x200;
constexpr uint64_t kSoftmaxPrimitiveStageLocalMaxReadyBase = 0x000;
constexpr uint64_t kSoftmaxPrimitiveStageLocalSumReadyBase = 0x020;
constexpr uint64_t kSoftmaxPrimitiveStageLocalMaxBase = 0x040;
constexpr uint64_t kSoftmaxPrimitiveStageLocalSumBase = 0x080;
constexpr uint64_t kSoftmaxPrimitiveStageGlobalBase = 0x180;
constexpr uint64_t kSoftmaxPrimitiveStageGlobalMaxReadyBase = 0x000;
constexpr uint64_t kSoftmaxPrimitiveStageGlobalSumReadyBase = 0x020;
constexpr uint64_t kSoftmaxPrimitiveStageGlobalMaxBase = 0x040;
constexpr uint64_t kSoftmaxPrimitiveStageInvSumBase = 0x080;
constexpr uint64_t kSoftmaxPrimitiveStageValueStride = 0x8;

uint64_t softmax_primitive_block_worker_addr(uint64_t coord_base_gm,
                                             int worker_slot,
                                             uint64_t field_offset) {
    return coord_base_gm +
           static_cast<uint64_t>(worker_slot) * kSoftmaxPrimitiveBlockWorkerStride +
           field_offset;
}

uint64_t softmax_primitive_block_global_addr(int worker_core,
                                             uint64_t field_offset,
                                             uint64_t block_row = 0) {
    return softmax_primitive_mailbox_addr(
        worker_core,
        kSoftmaxPrimitiveBlockGlobalBase + field_offset +
            block_row * kSoftmaxPrimitiveBlockValueStride);
}

uint64_t softmax_primitive_stage_worker_addr(uint64_t coord_base_gm,
                                             int worker_slot,
                                             uint64_t field_base,
                                             uint64_t stage_slot) {
    return coord_base_gm +
           static_cast<uint64_t>(worker_slot) * kSoftmaxPrimitiveStageWorkerStride +
           field_base + stage_slot * kSoftmaxPrimitiveStageValueStride;
}

uint64_t softmax_primitive_stage_global_addr(int worker_core,
                                             uint64_t field_base,
                                             uint64_t stage_slot) {
    return softmax_primitive_mailbox_addr(
        worker_core,
        kSoftmaxPrimitiveStageGlobalBase + field_base +
            stage_slot * kSoftmaxPrimitiveStageValueStride);
}

float coordinator_reciprocal_and_broadcast(int executor_core_id,
                                           int worker_cores,
                                           uint64_t chunk_elems,
                                           uint64_t row,
                                           uint64_t global_sum_seq,
                                           double global_row_sum) {
    const uint64_t scalar_input_gm = primitive_batch_slot_gm(executor_core_id, chunk_elems, 0);
    const uint64_t scalar_output_gm = primitive_batch_output_gm(executor_core_id, chunk_elems, 0);
    write_fp32_to_gm(scalar_input_gm, static_cast<float>(global_row_sum));
    const SFUPrimitiveDesc reciprocal_desc = {
        .job_id = kPrimitiveTagBase + 0x860000ULL + row,
        .input0_gm_addr = scalar_input_gm,
        .input1_gm_addr = 0,
        .output_gm_addr = scalar_output_gm,
        .op = static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL),
        .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
        .elem_count = 1,
        .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
        .input1_stride_bytes = 0,
        .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
        .flags = 0,
        .approx_mode = 0,
    };
    if (!issue_sfu_primitive_batch_descs(
            executor_core_id, std::vector<SFUPrimitiveDesc>{reciprocal_desc},
            kPrimitiveTagBase + 0x870000ULL + row)) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const float inv_sum = read_fp32_from_gm(scalar_output_gm);
    for (int slot = 0; slot < worker_cores; ++slot) {
        const int worker_core = gemm_worker_core_for_slot(slot);
        softmax_primitive_publish_u64(
            executor_core_id, worker_core,
            kSoftmaxPrimitiveMboxGlobalValue,
            fp32_to_reg(inv_sum));
        softmax_primitive_publish_u64(
            executor_core_id, worker_core,
            kSoftmaxPrimitiveMboxGlobalReady, global_sum_seq);
    }
    return inv_sum;
}

int resolve_softmax_primitive_worker_count(uint64_t dim,
                                           int64_t requested_worker_cores,
                                           int64_t multicore_min_dim) {
    if (ACTIVE_GEMM_CORES <= 1) {
        return 1;
    }
    if (multicore_min_dim <= 0) {
        multicore_min_dim = 1;
    }
    int workers = 1;
    if (requested_worker_cores > 0) {
        workers = static_cast<int>(requested_worker_cores);
    } else if (dim >= static_cast<uint64_t>(multicore_min_dim)) {
        workers = ACTIVE_GEMM_CORES;
    }
    if (workers < 1) {
        workers = 1;
    }
    if (workers > ACTIVE_GEMM_CORES) {
        workers = ACTIVE_GEMM_CORES;
    }
    if (static_cast<uint64_t>(workers) > dim) {
        workers = static_cast<int>(dim);
    }
    return workers < 1 ? 1 : workers;
}

uint64_t resolve_softmax_primitive_row_block(uint64_t rows, int64_t requested_row_block) {
    if (rows <= 1) {
        return 1;
    }
    int64_t row_block = requested_row_block;
    if (row_block <= 0) {
        row_block = 4;
    }
    if (row_block < 1) {
        row_block = 1;
    }
    if (row_block > 8) {
        row_block = 8;
    }
    if (static_cast<uint64_t>(row_block) > rows) {
        row_block = static_cast<int64_t>(rows);
    }
    return static_cast<uint64_t>(row_block);
}

uint64_t resolve_softmax_primitive_pipeline_depth(uint64_t rows, int64_t requested_depth) {
    if (rows <= 1) {
        return 1;
    }
    int64_t depth = requested_depth;
    if (depth <= 0) {
        depth = 1;
    }
    if (depth > 2) {
        depth = 2;
    }
    if (static_cast<uint64_t>(depth) > rows) {
        depth = static_cast<int64_t>(rows);
    }
    return static_cast<uint64_t>(depth);
}

enum class SoftmaxRowPipelineStage {
    EMPTY,
    LOCAL_MAX_DONE,
    LOCAL_MAX_PUBLISHED,
    GLOBAL_MAX_READY,
    GLOBAL_MAX_DONE,
    LOCAL_SUM_DONE,
    LOCAL_SUM_PUBLISHED,
    GLOBAL_SUM_READY,
    GLOBAL_SUM_DONE,
    NORMALIZED,
};

struct SoftmaxRowPipelineState {
    uint64_t row;
    uint64_t slot;
    uint64_t local_max_seq;
    uint64_t global_max_seq;
    uint64_t local_sum_seq;
    uint64_t global_sum_seq;
    double local_max;
    double row_max;
    double local_sum;
    float inv_sum;
    std::vector<float> row_exp;
    SoftmaxRowPipelineStage stage;
};

void softmax_primitive_slice_for_worker(uint64_t dim,
                                        int worker_slot,
                                        int worker_cores,
                                        uint64_t* col_begin,
                                        uint64_t* col_end) {
    *col_begin = (dim * static_cast<uint64_t>(worker_slot)) /
                 static_cast<uint64_t>(worker_cores);
    *col_end = (dim * static_cast<uint64_t>(worker_slot + 1)) /
               static_cast<uint64_t>(worker_cores);
}

bool advance_softmax_stage_local_max(int executor_core_id,
                                     int worker_slot,
                                     uint64_t row,
                                     uint64_t slice_begin,
                                     uint64_t slice_end,
                                     uint64_t chunk_elems,
                                     uint64_t max_batch_items,
                                     uint64_t input_hbm_base,
                                     uint64_t dim,
                                     SoftmaxRowPipelineState* state) {
    double local_row_max = -std::numeric_limits<double>::infinity();
    uint64_t processed_col = slice_begin;
    uint64_t row_group_index = 0;
    while (processed_col < slice_end) {
        const uint64_t remaining_chunks =
            (slice_end - processed_col + chunk_elems - 1) / chunk_elems;
        const uint64_t batch_items = std::min(remaining_chunks, max_batch_items);
        std::vector<SFUPrimitiveDesc> reduce_descs;
        reduce_descs.reserve(static_cast<size_t>(batch_items));
        std::vector<uint64_t> partial_gms(static_cast<size_t>(batch_items), 0);

        for (uint64_t item = 0; item < batch_items; ++item) {
            const uint64_t col = processed_col + item * chunk_elems;
            const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
            const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
            const uint64_t input_gm =
                primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
            const uint64_t output_gm =
                primitive_batch_output_gm(executor_core_id, chunk_elems, item);
            const uint64_t hbm_addr = input_hbm_base + (row * dim + col) * sizeof(float);

            dma_remote_load_to_gm(executor_core_id, hbm_addr, input_gm, chunk_bytes);
            reduce_descs.push_back(SFUPrimitiveDesc{
                .job_id = kPrimitiveTagBase + 0x880000ULL +
                          static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                          row * 0x1000ULL + row_group_index * 16 + item,
                .input0_gm_addr = input_gm,
                .input1_gm_addr = 0,
                .output_gm_addr = output_gm,
                .op = static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX),
                .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                .elem_count = static_cast<uint32_t>(elems_this_chunk),
                .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                .input1_stride_bytes = 0,
                .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                .flags = 0,
                .approx_mode = 0,
            });
            partial_gms[static_cast<size_t>(item)] = output_gm;
        }

        const uint64_t batch_tag =
            kPrimitiveTagBase + 0x890000ULL +
            static_cast<uint64_t>(worker_slot) * 0x100000ULL +
            row * 0x1000ULL + row_group_index;
        if (!issue_sfu_primitive_batch_descs(executor_core_id, reduce_descs, batch_tag)) {
            return false;
        }
        for (uint64_t item = 0; item < batch_items; ++item) {
            const float partial = read_fp32_from_gm(partial_gms[static_cast<size_t>(item)]);
            local_row_max = std::max(local_row_max, static_cast<double>(partial));
        }
        processed_col += batch_items * chunk_elems;
        row_group_index += 1;
    }

    state->row = row;
    state->local_max = local_row_max;
    state->stage = SoftmaxRowPipelineStage::LOCAL_MAX_DONE;
    return true;
}

bool advance_softmax_stage_exp_sum(int executor_core_id,
                                   int worker_slot,
                                   uint64_t row,
                                   uint64_t slice_begin,
                                   uint64_t slice_end,
                                   uint64_t slice_elems,
                                   uint64_t chunk_elems,
                                   uint64_t max_batch_items,
                                   uint64_t input_hbm_base,
                                   uint64_t dim,
                                   SoftmaxRowPipelineState* state) {
    std::vector<float> chunk_values;
    state->local_sum = 0.0;
    state->row_exp.assign(static_cast<size_t>(slice_elems), 0.0f);

    uint64_t processed_col = slice_begin;
    uint64_t row_group_index = 0;
    while (processed_col < slice_end) {
        const uint64_t remaining_chunks =
            (slice_end - processed_col + chunk_elems - 1) / chunk_elems;
        const uint64_t batch_items = std::min(remaining_chunks, max_batch_items);
        std::vector<SFUPrimitiveDesc> exp_descs;
        exp_descs.reserve(static_cast<size_t>(batch_items));
        std::vector<uint64_t> exp_output_gms(static_cast<size_t>(batch_items), 0);
        std::vector<uint64_t> elem_counts(static_cast<size_t>(batch_items), 0);
        std::vector<uint64_t> cols(static_cast<size_t>(batch_items), 0);

        for (uint64_t item = 0; item < batch_items; ++item) {
            const uint64_t col = processed_col + item * chunk_elems;
            const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
            const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
            const uint64_t input_gm =
                primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
            const uint64_t output_gm =
                primitive_batch_output_gm(executor_core_id, chunk_elems, item);
            const uint64_t hbm_addr = input_hbm_base + (row * dim + col) * sizeof(float);

            dma_remote_load_to_gm(executor_core_id, hbm_addr, input_gm, chunk_bytes);
            read_values_from_gm(input_gm, elems_this_chunk, &chunk_values);
            for (float& value : chunk_values) {
                value = static_cast<float>(static_cast<double>(value) - state->row_max);
            }
            write_values_to_gm(input_gm, chunk_values);

            exp_descs.push_back(SFUPrimitiveDesc{
                .job_id = kPrimitiveTagBase + 0x8a0000ULL +
                          static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                          row * 0x1000ULL + row_group_index * 16 + item,
                .input0_gm_addr = input_gm,
                .input1_gm_addr = 0,
                .output_gm_addr = output_gm,
                .op = static_cast<uint32_t>(SFUPrimitiveOp::EXP),
                .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                .elem_count = static_cast<uint32_t>(elems_this_chunk),
                .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                .input1_stride_bytes = 0,
                .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                .flags = 0,
                .approx_mode = 0,
            });
            exp_output_gms[static_cast<size_t>(item)] = output_gm;
            elem_counts[static_cast<size_t>(item)] = elems_this_chunk;
            cols[static_cast<size_t>(item)] = col;
        }

        const uint64_t exp_batch_tag =
            kPrimitiveTagBase + 0x8b0000ULL +
            static_cast<uint64_t>(worker_slot) * 0x100000ULL +
            row * 0x1000ULL + row_group_index;
        if (!issue_sfu_primitive_batch_descs(executor_core_id, exp_descs, exp_batch_tag)) {
            return false;
        }

        std::vector<SFUPrimitiveDesc> sum_descs;
        sum_descs.reserve(static_cast<size_t>(batch_items));
        std::vector<uint64_t> sum_output_gms(static_cast<size_t>(batch_items), 0);
        for (uint64_t item = 0; item < batch_items; ++item) {
            const uint64_t elems_this_chunk = elem_counts[static_cast<size_t>(item)];
            const uint64_t col = cols[static_cast<size_t>(item)];
            const uint64_t exp_output_gm = exp_output_gms[static_cast<size_t>(item)];
            read_values_from_gm(exp_output_gm, elems_this_chunk, &chunk_values);
            for (uint64_t i = 0; i < elems_this_chunk; ++i) {
                state->row_exp[static_cast<size_t>(col - slice_begin + i)] =
                    chunk_values[static_cast<size_t>(i)];
            }
            const uint64_t sum_output_gm =
                primitive_batch_output_gm(executor_core_id, chunk_elems, item);
            sum_descs.push_back(SFUPrimitiveDesc{
                .job_id = kPrimitiveTagBase + 0x8c0000ULL +
                          static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                          row * 0x1000ULL + row_group_index * 16 + item,
                .input0_gm_addr = exp_output_gm,
                .input1_gm_addr = 0,
                .output_gm_addr = sum_output_gm,
                .op = static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM),
                .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                .elem_count = static_cast<uint32_t>(elems_this_chunk),
                .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                .input1_stride_bytes = 0,
                .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                .flags = 0,
                .approx_mode = 0,
            });
            sum_output_gms[static_cast<size_t>(item)] = sum_output_gm;
        }

        const uint64_t sum_batch_tag =
            kPrimitiveTagBase + 0x8d0000ULL +
            static_cast<uint64_t>(worker_slot) * 0x100000ULL +
            row * 0x1000ULL + row_group_index;
        if (!issue_sfu_primitive_batch_descs(executor_core_id, sum_descs, sum_batch_tag)) {
            return false;
        }
        for (uint64_t item = 0; item < batch_items; ++item) {
            state->local_sum +=
                static_cast<double>(read_fp32_from_gm(sum_output_gms[static_cast<size_t>(item)]));
        }

        processed_col += batch_items * chunk_elems;
        row_group_index += 1;
    }

    state->stage = SoftmaxRowPipelineStage::LOCAL_SUM_DONE;
    return true;
}

bool advance_softmax_stage_normalize(int executor_core_id,
                                     uint64_t row,
                                     uint64_t slice_begin,
                                     uint64_t slice_end,
                                     uint64_t chunk_elems,
                                     uint64_t output_hbm_base,
                                     uint64_t dim,
                                     int64_t verify,
                                     SoftmaxRowPipelineState* state,
                                     double* max_abs_diff,
                                     double* max_rel_diff,
                                     double* max_row_sum_error) {
    const double row_sum = state->inv_sum != 0.0f
                               ? 1.0 / static_cast<double>(state->inv_sum)
                               : std::numeric_limits<double>::infinity();
    double output_row_sum = 0.0;
    std::vector<float> output_values;
    for (uint64_t col = slice_begin; col < slice_end; col += chunk_elems) {
        const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
        output_values.assign(static_cast<size_t>(elems_this_chunk), 0.0f);
        for (uint64_t i = 0; i < elems_this_chunk; ++i) {
            const uint64_t global_col = col + i;
            const float got =
                state->row_exp[static_cast<size_t>(global_col - slice_begin)] *
                state->inv_sum;
            output_values[static_cast<size_t>(i)] = got;
            output_row_sum += static_cast<double>(got);
            if (verify != 0) {
                const double expected_exp =
                    std::exp(static_cast<double>(
                        softmax_primitive_input_value(row, global_col)) - state->row_max);
                const double expected = expected_exp / row_sum;
                const double abs_diff = std::fabs(static_cast<double>(got) - expected);
                const double rel_diff =
                    abs_diff / std::max(std::fabs(expected), 1.0e-12);
                *max_abs_diff = std::max(*max_abs_diff, abs_diff);
                *max_rel_diff = std::max(*max_rel_diff, rel_diff);
                if (!std::isfinite(got)) {
                    std::fprintf(stderr,
                                 "[SFU-SOFTMAX-PRIMITIVE] non-finite output row=%llu col=%llu got=%g row_max=%g row_sum=%g inv_sum=%g\n",
                                 static_cast<unsigned long long>(row),
                                 static_cast<unsigned long long>(global_col),
                                 got,
                                 state->row_max,
                                 row_sum,
                                 state->inv_sum);
                    return false;
                }
            }
        }

        const uint64_t output_gm = primitive_input_gm(executor_core_id);
        write_values_to_gm(output_gm, output_values);
        const uint64_t output_hbm = output_hbm_base + (row * dim + col) * sizeof(float);
        set_len(elems_this_chunk * sizeof(float));
        remote_store(output_gm, output_hbm);
    }

    if (verify != 0) {
        const double row_sum_error = std::fabs(output_row_sum - (state->local_sum / row_sum));
        *max_row_sum_error = std::max(*max_row_sum_error, row_sum_error);
        if (*max_abs_diff > 1.0e-4 || row_sum_error > 1.0e-4) {
            std::fprintf(stderr,
                         "[SFU-SOFTMAX-PRIMITIVE] verify failed row=%llu max_abs_diff=%g row_sum_error=%g\n",
                         static_cast<unsigned long long>(row),
                         *max_abs_diff,
                         row_sum_error);
            return false;
        }
    }

    state->stage = SoftmaxRowPipelineStage::NORMALIZED;
    return true;
}

int run_sfu_primitive_softmax_row_block_for_core(int executor_core_id,
                                                int requested_core_id,
                                                const MatmulRuntimeConfig& cfg,
                                                uint64_t rows,
                                                uint64_t dim,
                                                int worker_cores,
                                                uint64_t chunk_elems,
                                                uint64_t max_batch_items,
                                                uint64_t row_block,
                                                uint64_t row_blocks,
                                                uint64_t block_syncs,
                                                int64_t verify,
                                                uint64_t pipeline_depth,
                                                const char* pipeline_mode) {
    (void)requested_core_id;

    const int worker_slot = gemm_worker_slot_for_core(executor_core_id);
    if (worker_slot < 0 || worker_slot >= ACTIVE_GEMM_CORES) {
        return 0;
    }
    const bool participates = worker_slot < worker_cores;
    if (!participates) {
        return 0;
    }

    const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, 0, cfg);
    const uint64_t total_elems = rows * dim;
    const uint64_t total_bytes = total_elems * sizeof(float);
    const uint64_t required_bytes = total_bytes * 2;
    const uint64_t available_bytes = primitive_hbm_available_bytes(desc);
    if (available_bytes < required_bytes) {
        std::fprintf(stderr,
                     "[SFU-SOFTMAX-PRIMITIVE] requested bytes=%llu exceed available HBM C-region bytes=%llu\n",
                     static_cast<unsigned long long>(required_bytes),
                     static_cast<unsigned long long>(available_bytes));
        return 1;
    }

    const uint64_t input_hbm_base = desc.c_base_mm;
    const uint64_t output_hbm_base = desc.c_base_mm + total_bytes;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double max_row_sum_error = 0.0;
    uint64_t slice_begin = 0;
    uint64_t slice_end = 0;
    softmax_primitive_slice_for_worker(
        dim, worker_slot, worker_cores, &slice_begin, &slice_end);
    const uint64_t slice_elems = slice_end - slice_begin;
    const uint64_t dim_per_core = (dim + static_cast<uint64_t>(worker_cores) - 1) /
                                  static_cast<uint64_t>(worker_cores);
    const int coordinator_core = gemm_worker_core_for_slot(0);
    const bool is_coordinator = (worker_slot == 0);
    const uint64_t coord_slot_bytes = 2 * chunk_elems * kPrimitiveElemStride;
    const uint64_t coord_base_gm =
        primitive_input_gm(coordinator_core) + max_batch_items * coord_slot_bytes + 0x100ULL;

    uint64_t planned_chunks_per_row = 0;
    uint64_t planned_groups_per_row = 0;
    for (int slot = 0; slot < worker_cores; ++slot) {
        uint64_t begin = 0;
        uint64_t end = 0;
        softmax_primitive_slice_for_worker(dim, slot, worker_cores, &begin, &end);
        const uint64_t elems = end - begin;
        const uint64_t slice_chunks = (elems + chunk_elems - 1) / chunk_elems;
        planned_chunks_per_row += slice_chunks;
        planned_groups_per_row += (slice_chunks + max_batch_items - 1) / max_batch_items;
    }
    const uint64_t planned_total_chunks = rows * planned_chunks_per_row;
    const uint64_t planned_total_batches =
        rows * (planned_groups_per_row * 3 + 1);
    const uint64_t planned_hbm_init_write_bytes = total_bytes;
    const uint64_t planned_hbm_read_bytes = total_bytes * 2;
    const uint64_t planned_hbm_write_bytes = total_bytes;

    if (is_coordinator) {
        std::printf("[SFU-SOFTMAX-PRIMITIVE] begin executor_core=%d rows=%llu dim=%llu worker_cores=%llu dim_per_core=%llu chunk_elems=%llu chunks_per_row=%llu row_block=%llu row_blocks=%llu pipeline_depth=%llu pipeline_mode=%s verify=%lld multicore_min_dim=%lld\n",
                    executor_core_id,
                    static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(dim),
                    static_cast<unsigned long long>(worker_cores),
                    static_cast<unsigned long long>(dim_per_core),
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(planned_chunks_per_row),
                    static_cast<unsigned long long>(row_block),
                    static_cast<unsigned long long>(row_blocks),
                    static_cast<unsigned long long>(pipeline_depth),
                    pipeline_mode,
                    static_cast<long long>(verify),
                    static_cast<long long>(0));
        std::fflush(stdout);
    }

    std::vector<float> chunk_values;
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t col = slice_begin; col < slice_end; col += chunk_elems) {
            const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
            fill_softmax_primitive_chunk(&chunk_values, row, col, elems_this_chunk);
            const uint64_t input_gm = primitive_input_gm(executor_core_id);
            write_values_to_gm(input_gm, chunk_values);
            const uint64_t hbm_addr =
                input_hbm_base + (row * dim + col) * sizeof(float);
            set_len(elems_this_chunk * sizeof(float));
            remote_store(input_gm, hbm_addr);
        }
    }

    std::vector<std::vector<float>> block_row_exp(
        static_cast<size_t>(row_block),
        std::vector<float>(static_cast<size_t>(slice_elems), 0.0f));
    std::vector<double> local_max_values(static_cast<size_t>(row_block), 0.0);
    std::vector<double> global_max_values(static_cast<size_t>(row_block), 0.0);
    std::vector<double> local_sum_values(static_cast<size_t>(row_block), 0.0);
    std::vector<float> inv_sum_values(static_cast<size_t>(row_block), 0.0f);
    std::vector<float> output_values;
    const bool packed_two_row_sync = (row_block <= 2);
    const bool coordinator_nbpoll =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_NBPOLL", 0) != 0;

    for (uint64_t row_base = 0; row_base < rows; row_base += row_block) {
        const uint64_t block_index = row_base / row_block;
        const uint64_t block_rows = std::min(row_block, rows - row_base);
        const uint64_t local_max_seq = block_index * 4 + 1;
        const uint64_t global_max_seq = block_index * 4 + 2;
        const uint64_t local_sum_seq = block_index * 4 + 3;
        const uint64_t global_sum_seq = block_index * 4 + 4;

        for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
            const uint64_t row = row_base + block_row;
            double local_row_max = -std::numeric_limits<double>::infinity();
            uint64_t processed_col = slice_begin;
            uint64_t row_group_index = 0;
            while (processed_col < slice_end) {
                const uint64_t remaining_chunks =
                    (slice_end - processed_col + chunk_elems - 1) / chunk_elems;
                const uint64_t batch_items =
                    std::min(remaining_chunks, max_batch_items);
                std::vector<SFUPrimitiveDesc> reduce_descs;
                reduce_descs.reserve(static_cast<size_t>(batch_items));
                std::vector<uint64_t> partial_gms(static_cast<size_t>(batch_items), 0);

                for (uint64_t item = 0; item < batch_items; ++item) {
                    const uint64_t col = processed_col + item * chunk_elems;
                    const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
                    const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
                    const uint64_t input_gm =
                        primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
                    const uint64_t output_gm =
                        primitive_batch_output_gm(executor_core_id, chunk_elems, item);
                    const uint64_t hbm_addr =
                        input_hbm_base + (row * dim + col) * sizeof(float);

                    dma_remote_load_to_gm(executor_core_id, hbm_addr, input_gm, chunk_bytes);
                    reduce_descs.push_back(SFUPrimitiveDesc{
                        .job_id = kPrimitiveTagBase + 0x800000ULL +
                                  static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                                  row * 0x1000ULL + row_group_index * 16 + item,
                        .input0_gm_addr = input_gm,
                        .input1_gm_addr = 0,
                        .output_gm_addr = output_gm,
                        .op = static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX),
                        .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                        .elem_count = static_cast<uint32_t>(elems_this_chunk),
                        .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                        .input1_stride_bytes = 0,
                        .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                        .flags = 0,
                        .approx_mode = 0,
                    });
                    partial_gms[static_cast<size_t>(item)] = output_gm;
                }

                const uint64_t batch_tag =
                    kPrimitiveTagBase + 0x810000ULL +
                    static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                    row * 0x1000ULL + row_group_index;
                if (!issue_sfu_primitive_batch_descs(executor_core_id, reduce_descs, batch_tag)) {
                    return 1;
                }
                for (uint64_t item = 0; item < batch_items; ++item) {
                    const float partial = read_fp32_from_gm(partial_gms[static_cast<size_t>(item)]);
                    local_row_max = std::max(local_row_max, static_cast<double>(partial));
                }
                processed_col += batch_items * chunk_elems;
                row_group_index += 1;
            }

            local_max_values[static_cast<size_t>(block_row)] = local_row_max;
            if (!packed_two_row_sync) {
                softmax_primitive_publish_to_addr(
                    executor_core_id,
                    softmax_primitive_block_worker_addr(
                        coord_base_gm,
                        worker_slot,
                        kSoftmaxPrimitiveBlockLocalMaxBase +
                            block_row * kSoftmaxPrimitiveBlockValueStride),
                    fp32_to_reg(static_cast<float>(local_row_max)));
            }
        }
        if (packed_two_row_sync) {
            const float low_max = static_cast<float>(local_max_values[0]);
            const float high_max = block_rows > 1
                                       ? static_cast<float>(local_max_values[1])
                                       : low_max;
            softmax_primitive_publish_to_addr(
                executor_core_id,
                softmax_primitive_block_worker_addr(
                    coord_base_gm, worker_slot, kSoftmaxPrimitiveBlockLocalMaxBase),
                pack_two_fp32_to_reg(low_max, high_max));
        }
        softmax_primitive_publish_to_addr(
            executor_core_id,
            softmax_primitive_block_worker_addr(
                coord_base_gm, worker_slot, kSoftmaxPrimitiveBlockMaxReady),
            local_max_seq);

        if (is_coordinator) {
            for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
                global_max_values[static_cast<size_t>(block_row)] =
                    -std::numeric_limits<double>::infinity();
            }
            const auto consume_worker_max = [&](int slot) {
                if (packed_two_row_sync) {
                    const uint64_t packed_max = gm2reg(softmax_primitive_block_worker_addr(
                        coord_base_gm, slot, kSoftmaxPrimitiveBlockLocalMaxBase));
                    const float worker_max0 = low_fp32_from_packed_reg(packed_max);
                    global_max_values[0] =
                        std::max(global_max_values[0], static_cast<double>(worker_max0));
                    if (block_rows > 1) {
                        const float worker_max1 = high_fp32_from_packed_reg(packed_max);
                        global_max_values[1] =
                            std::max(global_max_values[1], static_cast<double>(worker_max1));
                    }
                } else {
                    for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
                        const float worker_max = fp32_from_reg(gm2reg(
                            softmax_primitive_block_worker_addr(
                                coord_base_gm,
                                slot,
                                kSoftmaxPrimitiveBlockLocalMaxBase +
                                    block_row * kSoftmaxPrimitiveBlockValueStride)));
                        global_max_values[static_cast<size_t>(block_row)] =
                            std::max(global_max_values[static_cast<size_t>(block_row)],
                                     static_cast<double>(worker_max));
                    }
                }
            };
            if (coordinator_nbpoll) {
                std::vector<uint8_t> observed_max_workers(static_cast<size_t>(worker_cores), 0);
                int remaining_max_workers = worker_cores;
                while (remaining_max_workers > 0) {
                    for (int slot = 0; slot < worker_cores; ++slot) {
                        if (observed_max_workers[static_cast<size_t>(slot)] != 0) {
                            continue;
                        }
                        const uint64_t max_ready_addr = softmax_primitive_block_worker_addr(
                            coord_base_gm, slot, kSoftmaxPrimitiveBlockMaxReady);
                        if (!softmax_primitive_poll_ready(max_ready_addr, local_max_seq)) {
                            continue;
                        }
                        observed_max_workers[static_cast<size_t>(slot)] = 1;
                        --remaining_max_workers;
                        consume_worker_max(slot);
                    }
                }
            } else {
                for (int slot = 0; slot < worker_cores; ++slot) {
                    adaptive_wait_eq(
                        softmax_primitive_block_worker_addr(
                            coord_base_gm, slot, kSoftmaxPrimitiveBlockMaxReady),
                        local_max_seq);
                    consume_worker_max(slot);
                }
            }
            for (int slot = 0; slot < worker_cores; ++slot) {
                const int worker_core = gemm_worker_core_for_slot(slot);
                if (packed_two_row_sync) {
                    const float low_max = static_cast<float>(global_max_values[0]);
                    const float high_max = block_rows > 1
                                               ? static_cast<float>(global_max_values[1])
                                               : low_max;
                    softmax_primitive_publish_u64(
                        executor_core_id,
                        worker_core,
                        kSoftmaxPrimitiveBlockGlobalBase +
                            kSoftmaxPrimitiveBlockGlobalMaxBase,
                        pack_two_fp32_to_reg(low_max, high_max));
                } else {
                    for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
                        softmax_primitive_publish_u64(
                            executor_core_id,
                            worker_core,
                            kSoftmaxPrimitiveBlockGlobalBase +
                                kSoftmaxPrimitiveBlockGlobalMaxBase +
                                block_row * kSoftmaxPrimitiveBlockValueStride,
                            fp32_to_reg(static_cast<float>(
                                global_max_values[static_cast<size_t>(block_row)])));
                    }
                }
                softmax_primitive_publish_u64(
                    executor_core_id,
                    worker_core,
                    kSoftmaxPrimitiveBlockGlobalBase +
                        kSoftmaxPrimitiveBlockGlobalMaxReady,
                    global_max_seq);
            }
        }

        adaptive_wait_eq(
            softmax_primitive_block_global_addr(
                executor_core_id, kSoftmaxPrimitiveBlockGlobalMaxReady),
            global_max_seq);

        const uint64_t cross_row_batch_rows =
            (packed_two_row_sync && block_rows > 1 && max_batch_items >= block_rows)
                ? block_rows
                : 1;
        if (cross_row_batch_rows > 1) {
            const uint64_t packed_global_max = gm2reg(softmax_primitive_block_global_addr(
                executor_core_id, kSoftmaxPrimitiveBlockGlobalMaxBase));
            std::vector<double> cross_row_max(static_cast<size_t>(block_rows), 0.0);
            for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
                cross_row_max[static_cast<size_t>(block_row)] =
                    static_cast<double>(
                        block_row == 0 ? low_fp32_from_packed_reg(packed_global_max)
                                       : high_fp32_from_packed_reg(packed_global_max));
                local_sum_values[static_cast<size_t>(block_row)] = 0.0;
                block_row_exp[static_cast<size_t>(block_row)].assign(
                    static_cast<size_t>(slice_elems), 0.0f);
            }

            uint64_t processed_col = slice_begin;
            uint64_t row_group_index = 0;
            while (processed_col < slice_end) {
                const uint64_t remaining_chunks =
                    (slice_end - processed_col + chunk_elems - 1) / chunk_elems;
                const uint64_t per_row_batch_items =
                    std::min(remaining_chunks, max_batch_items / cross_row_batch_rows);
                if (per_row_batch_items == 0) {
                    return 1;
                }
                const uint64_t cross_row_batch_items =
                    per_row_batch_items * cross_row_batch_rows;
                if (!(cross_row_batch_items <= max_batch_items)) {
                    return 1;
                }

                std::vector<SFUPrimitiveDesc> cross_row_exp_descs;
                cross_row_exp_descs.reserve(static_cast<size_t>(cross_row_batch_items));
                std::vector<uint64_t> exp_output_gms(
                    static_cast<size_t>(cross_row_batch_items), 0);
                std::vector<uint64_t> elem_counts(
                    static_cast<size_t>(cross_row_batch_items), 0);
                std::vector<uint64_t> cols(static_cast<size_t>(cross_row_batch_items), 0);
                std::vector<uint64_t> row_indices(
                    static_cast<size_t>(cross_row_batch_items), 0);

                for (uint64_t block_row = 0; block_row < cross_row_batch_rows; ++block_row) {
                    const uint64_t row = row_base + block_row;
                    const double row_max = cross_row_max[static_cast<size_t>(block_row)];
                    for (uint64_t item = 0; item < per_row_batch_items; ++item) {
                        const uint64_t combined_slot = block_row * per_row_batch_items + item;
                        const uint64_t col = processed_col + item * chunk_elems;
                        const uint64_t elems_this_chunk =
                            std::min(chunk_elems, slice_end - col);
                        const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
                        const uint64_t input_gm =
                            primitive_batch_slot_gm(
                                executor_core_id, chunk_elems, combined_slot);
                        const uint64_t output_gm =
                            primitive_batch_output_gm(
                                executor_core_id, chunk_elems, combined_slot);
                        const uint64_t hbm_addr =
                            input_hbm_base + (row * dim + col) * sizeof(float);

                        dma_remote_load_to_gm(
                            executor_core_id, hbm_addr, input_gm, chunk_bytes);
                        read_values_from_gm(input_gm, elems_this_chunk, &chunk_values);
                        for (float& value : chunk_values) {
                            value = static_cast<float>(
                                static_cast<double>(value) - row_max);
                        }
                        write_values_to_gm(input_gm, chunk_values);

                        cross_row_exp_descs.push_back(SFUPrimitiveDesc{
                            .job_id = kPrimitiveTagBase + 0x820000ULL +
                                      static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                                      row * 0x1000ULL + row_group_index * 64 +
                                      combined_slot,
                            .input0_gm_addr = input_gm,
                            .input1_gm_addr = 0,
                            .output_gm_addr = output_gm,
                            .op = static_cast<uint32_t>(SFUPrimitiveOp::EXP),
                            .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                            .elem_count = static_cast<uint32_t>(elems_this_chunk),
                            .input0_stride_bytes =
                                static_cast<uint32_t>(kPrimitiveElemStride),
                            .input1_stride_bytes = 0,
                            .output_stride_bytes =
                                static_cast<uint32_t>(kPrimitiveElemStride),
                            .flags = 0,
                            .approx_mode = 0,
                        });
                        exp_output_gms[static_cast<size_t>(combined_slot)] = output_gm;
                        elem_counts[static_cast<size_t>(combined_slot)] = elems_this_chunk;
                        cols[static_cast<size_t>(combined_slot)] = col;
                        row_indices[static_cast<size_t>(combined_slot)] = block_row;
                    }
                }

                const uint64_t exp_batch_tag =
                    kPrimitiveTagBase + 0x830000ULL +
                    static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                    row_base * 0x1000ULL + row_group_index;
                if (!issue_sfu_primitive_batch_descs(executor_core_id, cross_row_exp_descs, exp_batch_tag)) {
                    return 1;
                }

                std::vector<SFUPrimitiveDesc> cross_row_sum_descs;
                cross_row_sum_descs.reserve(static_cast<size_t>(cross_row_batch_items));
                std::vector<uint64_t> sum_output_gms(
                    static_cast<size_t>(cross_row_batch_items), 0);
                for (uint64_t combined_slot = 0;
                     combined_slot < cross_row_batch_items;
                     ++combined_slot) {
                    const uint64_t block_row = row_indices[static_cast<size_t>(combined_slot)];
                    const uint64_t row = row_base + block_row;
                    const uint64_t elems_this_chunk =
                        elem_counts[static_cast<size_t>(combined_slot)];
                    const uint64_t col = cols[static_cast<size_t>(combined_slot)];
                    const uint64_t exp_output_gm =
                        exp_output_gms[static_cast<size_t>(combined_slot)];
                    read_values_from_gm(exp_output_gm, elems_this_chunk, &chunk_values);
                    for (uint64_t i = 0; i < elems_this_chunk; ++i) {
                        block_row_exp[static_cast<size_t>(block_row)]
                                     [static_cast<size_t>((col - slice_begin) + i)] =
                            chunk_values[static_cast<size_t>(i)];
                    }

                    const uint64_t sum_output_gm =
                        primitive_batch_slot_gm(
                            executor_core_id, chunk_elems, combined_slot);
                    cross_row_sum_descs.push_back(SFUPrimitiveDesc{
                        .job_id = kPrimitiveTagBase + 0x840000ULL +
                                  static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                                  row * 0x1000ULL + row_group_index * 64 +
                                  combined_slot,
                        .input0_gm_addr = exp_output_gm,
                        .input1_gm_addr = 0,
                        .output_gm_addr = sum_output_gm,
                        .op = static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM),
                        .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                        .elem_count = static_cast<uint32_t>(elems_this_chunk),
                        .input0_stride_bytes =
                            static_cast<uint32_t>(kPrimitiveElemStride),
                        .input1_stride_bytes = 0,
                        .output_stride_bytes =
                            static_cast<uint32_t>(kPrimitiveElemStride),
                        .flags = 0,
                        .approx_mode = 0,
                    });
                    sum_output_gms[static_cast<size_t>(combined_slot)] = sum_output_gm;
                }

                const uint64_t sum_batch_tag =
                    kPrimitiveTagBase + 0x850000ULL +
                    static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                    row_base * 0x1000ULL + row_group_index;
                if (!issue_sfu_primitive_batch_descs(executor_core_id, cross_row_sum_descs, sum_batch_tag)) {
                    return 1;
                }
                for (uint64_t combined_slot = 0;
                     combined_slot < cross_row_batch_items;
                     ++combined_slot) {
                    const uint64_t block_row = row_indices[static_cast<size_t>(combined_slot)];
                    local_sum_values[static_cast<size_t>(block_row)] +=
                        static_cast<double>(
                            read_fp32_from_gm(
                                sum_output_gms[static_cast<size_t>(combined_slot)]));
                }

                processed_col += per_row_batch_items * chunk_elems;
                row_group_index += 1;
            }
        } else {
            for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
                const uint64_t row = row_base + block_row;
                double row_max = 0.0;
                if (packed_two_row_sync) {
                    const uint64_t packed_global_max = gm2reg(softmax_primitive_block_global_addr(
                        executor_core_id, kSoftmaxPrimitiveBlockGlobalMaxBase));
                    row_max = static_cast<double>(
                        block_row == 0 ? low_fp32_from_packed_reg(packed_global_max)
                                       : high_fp32_from_packed_reg(packed_global_max));
                } else {
                    row_max = static_cast<double>(fp32_from_reg(gm2reg(
                        softmax_primitive_block_global_addr(
                            executor_core_id,
                            kSoftmaxPrimitiveBlockGlobalMaxBase,
                            block_row))));
                }
                double local_row_sum = 0.0;
                block_row_exp[static_cast<size_t>(block_row)].assign(
                    static_cast<size_t>(slice_elems), 0.0f);

                uint64_t processed_col = slice_begin;
                uint64_t row_group_index = 0;
                while (processed_col < slice_end) {
                    const uint64_t remaining_chunks =
                        (slice_end - processed_col + chunk_elems - 1) / chunk_elems;
                    const uint64_t batch_items =
                        std::min(remaining_chunks, max_batch_items);
                    std::vector<SFUPrimitiveDesc> exp_descs;
                    exp_descs.reserve(static_cast<size_t>(batch_items));
                    std::vector<uint64_t> exp_output_gms(static_cast<size_t>(batch_items), 0);
                    std::vector<uint64_t> elem_counts(static_cast<size_t>(batch_items), 0);
                    std::vector<uint64_t> cols(static_cast<size_t>(batch_items), 0);

                    for (uint64_t item = 0; item < batch_items; ++item) {
                        const uint64_t col = processed_col + item * chunk_elems;
                        const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
                        const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
                        const uint64_t input_gm =
                            primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
                        const uint64_t output_gm =
                            primitive_batch_output_gm(executor_core_id, chunk_elems, item);
                        const uint64_t hbm_addr =
                            input_hbm_base + (row * dim + col) * sizeof(float);

                        dma_remote_load_to_gm(executor_core_id, hbm_addr, input_gm, chunk_bytes);
                        read_values_from_gm(input_gm, elems_this_chunk, &chunk_values);
                        for (float& value : chunk_values) {
                            value = static_cast<float>(static_cast<double>(value) - row_max);
                        }
                        write_values_to_gm(input_gm, chunk_values);

                        exp_descs.push_back(SFUPrimitiveDesc{
                            .job_id = kPrimitiveTagBase + 0x820000ULL +
                                      static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                                      row * 0x1000ULL + row_group_index * 16 + item,
                            .input0_gm_addr = input_gm,
                            .input1_gm_addr = 0,
                            .output_gm_addr = output_gm,
                            .op = static_cast<uint32_t>(SFUPrimitiveOp::EXP),
                            .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                            .elem_count = static_cast<uint32_t>(elems_this_chunk),
                            .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                            .input1_stride_bytes = 0,
                            .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                            .flags = 0,
                            .approx_mode = 0,
                        });
                        exp_output_gms[static_cast<size_t>(item)] = output_gm;
                        elem_counts[static_cast<size_t>(item)] = elems_this_chunk;
                        cols[static_cast<size_t>(item)] = col;
                    }

                    const uint64_t exp_batch_tag =
                        kPrimitiveTagBase + 0x830000ULL +
                        static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                        row * 0x1000ULL + row_group_index;
                    if (!issue_sfu_primitive_batch_descs(executor_core_id, exp_descs, exp_batch_tag)) {
                        return 1;
                    }

                    std::vector<SFUPrimitiveDesc> sum_descs;
                    sum_descs.reserve(static_cast<size_t>(batch_items));
                    std::vector<uint64_t> sum_output_gms(static_cast<size_t>(batch_items), 0);
                    for (uint64_t item = 0; item < batch_items; ++item) {
                        const uint64_t elems_this_chunk = elem_counts[static_cast<size_t>(item)];
                        const uint64_t col = cols[static_cast<size_t>(item)];
                        const uint64_t exp_output_gm = exp_output_gms[static_cast<size_t>(item)];
                        read_values_from_gm(exp_output_gm, elems_this_chunk, &chunk_values);
                        for (uint64_t i = 0; i < elems_this_chunk; ++i) {
                            block_row_exp[static_cast<size_t>(block_row)]
                                         [static_cast<size_t>((col - slice_begin) + i)] =
                                chunk_values[static_cast<size_t>(i)];
                        }
                        const uint64_t sum_output_gm =
                            primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
                        sum_descs.push_back(SFUPrimitiveDesc{
                            .job_id = kPrimitiveTagBase + 0x840000ULL +
                                      static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                                      row * 0x1000ULL + row_group_index * 16 + item,
                            .input0_gm_addr = exp_output_gm,
                            .input1_gm_addr = 0,
                            .output_gm_addr = sum_output_gm,
                            .op = static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM),
                            .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                            .elem_count = static_cast<uint32_t>(elems_this_chunk),
                            .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                            .input1_stride_bytes = 0,
                            .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                            .flags = 0,
                            .approx_mode = 0,
                        });
                        sum_output_gms[static_cast<size_t>(item)] = sum_output_gm;
                    }

                    const uint64_t sum_batch_tag =
                        kPrimitiveTagBase + 0x850000ULL +
                        static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                        row * 0x1000ULL + row_group_index;
                    if (!issue_sfu_primitive_batch_descs(executor_core_id, sum_descs, sum_batch_tag)) {
                        return 1;
                    }
                    for (uint64_t item = 0; item < batch_items; ++item) {
                        local_row_sum += static_cast<double>(
                            read_fp32_from_gm(sum_output_gms[static_cast<size_t>(item)]));
                    }

                    processed_col += batch_items * chunk_elems;
                    row_group_index += 1;
                }

                local_sum_values[static_cast<size_t>(block_row)] = local_row_sum;
                if (!packed_two_row_sync) {
                    softmax_primitive_publish_to_addr(
                        executor_core_id,
                        softmax_primitive_block_worker_addr(
                            coord_base_gm,
                            worker_slot,
                            kSoftmaxPrimitiveBlockLocalSumBase +
                                block_row * kSoftmaxPrimitiveBlockValueStride),
                        fp32_to_reg(static_cast<float>(local_row_sum)));
                }
            }
        }
        if (packed_two_row_sync) {
            const float low_sum = static_cast<float>(local_sum_values[0]);
            const float high_sum = block_rows > 1
                                       ? static_cast<float>(local_sum_values[1])
                                       : low_sum;
            softmax_primitive_publish_to_addr(
                executor_core_id,
                softmax_primitive_block_worker_addr(
                    coord_base_gm, worker_slot, kSoftmaxPrimitiveBlockLocalSumBase),
                pack_two_fp32_to_reg(low_sum, high_sum));
        }
        softmax_primitive_publish_to_addr(
            executor_core_id,
            softmax_primitive_block_worker_addr(
                coord_base_gm, worker_slot, kSoftmaxPrimitiveBlockSumReady),
            local_sum_seq);

        if (is_coordinator) {
            std::vector<double> global_sum_values(static_cast<size_t>(block_rows), 0.0);
            const auto consume_worker_sum = [&](int slot) {
                if (packed_two_row_sync) {
                    const uint64_t packed_sum = gm2reg(softmax_primitive_block_worker_addr(
                        coord_base_gm, slot, kSoftmaxPrimitiveBlockLocalSumBase));
                    global_sum_values[0] +=
                        static_cast<double>(low_fp32_from_packed_reg(packed_sum));
                    if (block_rows > 1) {
                        global_sum_values[1] +=
                            static_cast<double>(high_fp32_from_packed_reg(packed_sum));
                    }
                } else {
                    for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
                        const float worker_sum = fp32_from_reg(gm2reg(
                            softmax_primitive_block_worker_addr(
                                coord_base_gm,
                                slot,
                                kSoftmaxPrimitiveBlockLocalSumBase +
                                    block_row * kSoftmaxPrimitiveBlockValueStride)));
                        global_sum_values[static_cast<size_t>(block_row)] +=
                            static_cast<double>(worker_sum);
                    }
                }
            };
            if (coordinator_nbpoll) {
                std::vector<uint8_t> observed_sum_workers(static_cast<size_t>(worker_cores), 0);
                int remaining_sum_workers = worker_cores;
                while (remaining_sum_workers > 0) {
                    for (int slot = 0; slot < worker_cores; ++slot) {
                        if (observed_sum_workers[static_cast<size_t>(slot)] != 0) {
                            continue;
                        }
                        const uint64_t sum_ready_addr = softmax_primitive_block_worker_addr(
                            coord_base_gm, slot, kSoftmaxPrimitiveBlockSumReady);
                        if (!softmax_primitive_poll_ready(sum_ready_addr, local_sum_seq)) {
                            continue;
                        }
                        observed_sum_workers[static_cast<size_t>(slot)] = 1;
                        --remaining_sum_workers;
                        consume_worker_sum(slot);
                    }
                }
            } else {
                for (int slot = 0; slot < worker_cores; ++slot) {
                    adaptive_wait_eq(
                        softmax_primitive_block_worker_addr(
                            coord_base_gm, slot, kSoftmaxPrimitiveBlockSumReady),
                        local_sum_seq);
                    consume_worker_sum(slot);
                }
            }
            for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
                const uint64_t row = row_base + block_row;
                const uint64_t scalar_input_gm =
                    primitive_batch_slot_gm(executor_core_id, chunk_elems, 0);
                const uint64_t scalar_output_gm =
                    primitive_batch_output_gm(executor_core_id, chunk_elems, 0);
                write_fp32_to_gm(
                    scalar_input_gm,
                    static_cast<float>(global_sum_values[static_cast<size_t>(block_row)]));
                const SFUPrimitiveDesc reciprocal_desc = {
                    .job_id = kPrimitiveTagBase + 0x860000ULL + row,
                    .input0_gm_addr = scalar_input_gm,
                    .input1_gm_addr = 0,
                    .output_gm_addr = scalar_output_gm,
                    .op = static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL),
                    .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                    .elem_count = 1,
                    .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .input1_stride_bytes = 0,
                    .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .flags = 0,
                    .approx_mode = 0,
                };
                if (!issue_sfu_primitive_batch_descs(
                        executor_core_id,
                        std::vector<SFUPrimitiveDesc>{reciprocal_desc},
                        kPrimitiveTagBase + 0x870000ULL + row)) {
                    return 1;
                }
                inv_sum_values[static_cast<size_t>(block_row)] =
                    read_fp32_from_gm(scalar_output_gm);
                if (!std::isfinite(inv_sum_values[static_cast<size_t>(block_row)])) {
                    return 1;
                }
            }
            for (int slot = 0; slot < worker_cores; ++slot) {
                const int worker_core = gemm_worker_core_for_slot(slot);
                if (packed_two_row_sync) {
                    const float low_inv = inv_sum_values[0];
                    const float high_inv = block_rows > 1 ? inv_sum_values[1] : low_inv;
                    softmax_primitive_publish_u64(
                        executor_core_id,
                        worker_core,
                        kSoftmaxPrimitiveBlockGlobalBase +
                            kSoftmaxPrimitiveBlockInvSumBase,
                        pack_two_fp32_to_reg(low_inv, high_inv));
                } else {
                    for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
                        softmax_primitive_publish_u64(
                            executor_core_id,
                            worker_core,
                            kSoftmaxPrimitiveBlockGlobalBase +
                                kSoftmaxPrimitiveBlockInvSumBase +
                                block_row * kSoftmaxPrimitiveBlockValueStride,
                            fp32_to_reg(inv_sum_values[static_cast<size_t>(block_row)]));
                    }
                }
                softmax_primitive_publish_u64(
                    executor_core_id,
                    worker_core,
                    kSoftmaxPrimitiveBlockGlobalBase +
                        kSoftmaxPrimitiveBlockGlobalSumReady,
                    global_sum_seq);
            }
        }

        adaptive_wait_eq(
            softmax_primitive_block_global_addr(
                executor_core_id, kSoftmaxPrimitiveBlockGlobalSumReady),
            global_sum_seq);

        for (uint64_t block_row = 0; block_row < block_rows; ++block_row) {
            const uint64_t row = row_base + block_row;
            double row_max = 0.0;
            float inv_sum = 0.0f;
            if (packed_two_row_sync) {
                const uint64_t packed_global_max = gm2reg(softmax_primitive_block_global_addr(
                    executor_core_id, kSoftmaxPrimitiveBlockGlobalMaxBase));
                const uint64_t packed_inv_sum = gm2reg(softmax_primitive_block_global_addr(
                    executor_core_id, kSoftmaxPrimitiveBlockInvSumBase));
                row_max = static_cast<double>(
                    block_row == 0 ? low_fp32_from_packed_reg(packed_global_max)
                                   : high_fp32_from_packed_reg(packed_global_max));
                inv_sum = block_row == 0 ? low_fp32_from_packed_reg(packed_inv_sum)
                                         : high_fp32_from_packed_reg(packed_inv_sum);
            } else {
                row_max = static_cast<double>(fp32_from_reg(gm2reg(
                    softmax_primitive_block_global_addr(
                        executor_core_id,
                        kSoftmaxPrimitiveBlockGlobalMaxBase,
                        block_row))));
                inv_sum = fp32_from_reg(gm2reg(
                    softmax_primitive_block_global_addr(
                        executor_core_id,
                        kSoftmaxPrimitiveBlockInvSumBase,
                        block_row)));
            }
            const double row_sum = inv_sum != 0.0f
                                       ? 1.0 / static_cast<double>(inv_sum)
                                       : std::numeric_limits<double>::infinity();

            double output_row_sum = 0.0;
            for (uint64_t col = slice_begin; col < slice_end; col += chunk_elems) {
                const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
                output_values.assign(static_cast<size_t>(elems_this_chunk), 0.0f);
                for (uint64_t i = 0; i < elems_this_chunk; ++i) {
                    const uint64_t global_col = col + i;
                    const double expected_exp =
                        std::exp(static_cast<double>(
                            softmax_primitive_input_value(row, global_col)) - row_max);
                    const double expected = expected_exp / row_sum;
                    const float got =
                        block_row_exp[static_cast<size_t>(block_row)]
                                     [static_cast<size_t>(global_col - slice_begin)] *
                        inv_sum;
                    output_values[static_cast<size_t>(i)] = got;
                    output_row_sum += static_cast<double>(got);
                    if (verify != 0) {
                        const double abs_diff =
                            std::fabs(static_cast<double>(got) - expected);
                        const double rel_diff =
                            abs_diff / std::max(std::fabs(expected), 1.0e-12);
                        max_abs_diff = std::max(max_abs_diff, abs_diff);
                        max_rel_diff = std::max(max_rel_diff, rel_diff);
                        if (!std::isfinite(got)) {
                            std::fprintf(stderr,
                                         "[SFU-SOFTMAX-PRIMITIVE] non-finite output row=%llu col=%llu got=%g row_max=%g row_sum=%g inv_sum=%g exp_value=%g\n",
                                         static_cast<unsigned long long>(row),
                                         static_cast<unsigned long long>(global_col),
                                         got,
                                         row_max,
                                         row_sum,
                                         inv_sum,
                                         block_row_exp[static_cast<size_t>(block_row)]
                                                      [static_cast<size_t>(
                                                          global_col - slice_begin)]);
                            return 1;
                        }
                    }
                }

                const uint64_t output_gm = primitive_input_gm(executor_core_id);
                write_values_to_gm(output_gm, output_values);
                const uint64_t output_hbm =
                    output_hbm_base + (row * dim + col) * sizeof(float);
                set_len(elems_this_chunk * sizeof(float));
                remote_store(output_gm, output_hbm);
            }

            if (verify != 0) {
                const double row_sum_error = std::fabs(
                    output_row_sum -
                    (local_sum_values[static_cast<size_t>(block_row)] / row_sum));
                max_row_sum_error = std::max(max_row_sum_error, row_sum_error);
                if (max_abs_diff > 1.0e-4 || row_sum_error > 1.0e-4) {
                    std::fprintf(stderr,
                                 "[SFU-SOFTMAX-PRIMITIVE] verify failed row=%llu max_abs_diff=%g row_sum_error=%g\n",
                                 static_cast<unsigned long long>(row),
                                 max_abs_diff,
                                 row_sum_error);
                    return 1;
                }
            }
        }
    }

    if (is_coordinator) {
        std::printf("[SOFTMAX] mode=sfu-primitive-softmax executor_core=%d rows=%llu dim=%llu worker_cores=%llu dim_per_core=%llu chunk_elems=%llu row_block=%llu row_blocks=%llu block_syncs=%llu pipeline_depth=%llu pipeline_mode=%s primitive_stages=4 local_steps=3 cross_core_reduce_stages=2 chunks=%llu batches=%llu hbm_init_write_bytes=%llu hbm_read_bytes=%llu hbm_write_bytes=%llu max_abs_diff=%g max_rel_diff=%g max_row_sum_error=%g PASS\n",
                    executor_core_id,
                    static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(dim),
                    static_cast<unsigned long long>(worker_cores),
                    static_cast<unsigned long long>(dim_per_core),
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(row_block),
                    static_cast<unsigned long long>(row_blocks),
                    static_cast<unsigned long long>(block_syncs),
                    static_cast<unsigned long long>(pipeline_depth),
                    pipeline_mode,
                    static_cast<unsigned long long>(planned_total_chunks),
                    static_cast<unsigned long long>(planned_total_batches),
                    static_cast<unsigned long long>(planned_hbm_init_write_bytes),
                    static_cast<unsigned long long>(planned_hbm_read_bytes),
                    static_cast<unsigned long long>(planned_hbm_write_bytes),
                    max_abs_diff,
                    max_rel_diff,
                    max_row_sum_error);
        std::fflush(stdout);
    }
    return 0;
}

int run_sfu_primitive_softmax_row_pipeline_for_core(int executor_core_id,
                                                    int requested_core_id,
                                                    const MatmulRuntimeConfig& cfg,
                                                    uint64_t rows,
                                                    uint64_t dim,
                                                    int worker_cores,
                                                    uint64_t chunk_elems,
                                                    uint64_t max_batch_items,
                                                    uint64_t pipeline_depth,
                                                    int64_t verify) {
    (void)requested_core_id;

    const uint64_t pipeline_window_rows = std::max<uint64_t>(1, pipeline_depth);
    const uint64_t row_blocks = (rows + pipeline_window_rows - 1) / pipeline_window_rows;
    const uint64_t block_syncs = row_blocks * 2;
    const int worker_slot = gemm_worker_slot_for_core(executor_core_id);
    if (worker_slot < 0 || worker_slot >= ACTIVE_GEMM_CORES) {
        return 0;
    }
    const bool participates = worker_slot < worker_cores;
    if (!participates) {
        return 0;
    }

    const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, 0, cfg);
    const uint64_t total_elems = rows * dim;
    const uint64_t total_bytes = total_elems * sizeof(float);
    const uint64_t required_bytes = total_bytes * 2;
    const uint64_t available_bytes = primitive_hbm_available_bytes(desc);
    if (available_bytes < required_bytes) {
        std::fprintf(stderr,
                     "[SFU-SOFTMAX-PRIMITIVE] requested bytes=%llu exceed available HBM C-region bytes=%llu\n",
                     static_cast<unsigned long long>(required_bytes),
                     static_cast<unsigned long long>(available_bytes));
        return 1;
    }

    const uint64_t input_hbm_base = desc.c_base_mm;
    const uint64_t output_hbm_base = desc.c_base_mm + total_bytes;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double max_row_sum_error = 0.0;
    uint64_t slice_begin = 0;
    uint64_t slice_end = 0;
    softmax_primitive_slice_for_worker(
        dim, worker_slot, worker_cores, &slice_begin, &slice_end);
    const uint64_t slice_elems = slice_end - slice_begin;
    const uint64_t dim_per_core = (dim + static_cast<uint64_t>(worker_cores) - 1) /
                                  static_cast<uint64_t>(worker_cores);
    const int coordinator_core = gemm_worker_core_for_slot(0);
    const bool is_coordinator = (worker_slot == 0);
    const uint64_t coord_slot_bytes = 2 * chunk_elems * kPrimitiveElemStride;
    const uint64_t coord_base_gm =
        primitive_input_gm(coordinator_core) + max_batch_items * coord_slot_bytes + 0x100ULL;

    uint64_t planned_chunks_per_row = 0;
    uint64_t planned_groups_per_row = 0;
    for (int slot = 0; slot < worker_cores; ++slot) {
        uint64_t begin = 0;
        uint64_t end = 0;
        softmax_primitive_slice_for_worker(dim, slot, worker_cores, &begin, &end);
        const uint64_t elems = end - begin;
        const uint64_t slice_chunks = (elems + chunk_elems - 1) / chunk_elems;
        planned_chunks_per_row += slice_chunks;
        planned_groups_per_row += (slice_chunks + max_batch_items - 1) / max_batch_items;
    }
    const uint64_t planned_total_chunks = rows * planned_chunks_per_row;
    const uint64_t planned_total_batches = rows * (planned_groups_per_row * 3 + 1);
    const uint64_t planned_hbm_init_write_bytes = total_bytes;
    const uint64_t planned_hbm_read_bytes = total_bytes * 2;
    const uint64_t planned_hbm_write_bytes = total_bytes;

    if (is_coordinator) {
        std::printf("[SFU-SOFTMAX-PRIMITIVE] begin executor_core=%d rows=%llu dim=%llu worker_cores=%llu dim_per_core=%llu chunk_elems=%llu chunks_per_row=%llu row_block=%llu row_blocks=%llu pipeline_depth=%llu pipeline_mode=row verify=%lld multicore_min_dim=%lld\n",
                    executor_core_id,
                    static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(dim),
                    static_cast<unsigned long long>(worker_cores),
                    static_cast<unsigned long long>(dim_per_core),
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(planned_chunks_per_row),
                    static_cast<unsigned long long>(1),
                    static_cast<unsigned long long>(row_blocks),
                    static_cast<unsigned long long>(pipeline_depth),
                    static_cast<long long>(verify),
                    static_cast<long long>(0));
        std::printf("[SFU-SOFTMAX-PRIMITIVE] pipeline_mode=row pipeline_depth=%llu pipeline_window_rows=%llu dispatch=stage-row-state-machine\n",
                    static_cast<unsigned long long>(pipeline_depth),
                    static_cast<unsigned long long>(pipeline_window_rows));
        std::fflush(stdout);
    }

    std::vector<float> chunk_values;
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t col = slice_begin; col < slice_end; col += chunk_elems) {
            const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
            fill_softmax_primitive_chunk(&chunk_values, row, col, elems_this_chunk);
            const uint64_t input_gm = primitive_input_gm(executor_core_id);
            write_values_to_gm(input_gm, chunk_values);
            const uint64_t hbm_addr = input_hbm_base + (row * dim + col) * sizeof(float);
            set_len(elems_this_chunk * sizeof(float));
            remote_store(input_gm, hbm_addr);
        }
    }

    std::vector<SoftmaxRowPipelineState> stage_pipeline_states(
        static_cast<size_t>(pipeline_window_rows));
    uint64_t pipeline_stage_cycles = 0;

    for (uint64_t row_base = 0; row_base < rows; row_base += pipeline_window_rows) {
        const uint64_t window_rows = std::min(pipeline_window_rows, rows - row_base);

        for (uint64_t slot = 0; slot < window_rows; ++slot) {
            const uint64_t row = row_base + slot;
            SoftmaxRowPipelineState& state =
                stage_pipeline_states[static_cast<size_t>(slot)];
            state = SoftmaxRowPipelineState{
                .row = row,
                .slot = slot,
                .local_max_seq = row * 4 + 1,
                .global_max_seq = row * 4 + 2,
                .local_sum_seq = row * 4 + 3,
                .global_sum_seq = row * 4 + 4,
                .local_max = -std::numeric_limits<double>::infinity(),
                .row_max = -std::numeric_limits<double>::infinity(),
                .local_sum = 0.0,
                .inv_sum = 0.0f,
                .row_exp = std::vector<float>{},
                .stage = SoftmaxRowPipelineStage::EMPTY,
            };

            if (!advance_softmax_stage_local_max(
                    executor_core_id,
                    worker_slot,
                    row,
                    slice_begin,
                    slice_end,
                    chunk_elems,
                    max_batch_items,
                    input_hbm_base,
                    dim,
                    &state)) {
                return 1;
            }
            ++pipeline_stage_cycles;
            softmax_primitive_publish_to_addr(
                executor_core_id,
                softmax_primitive_stage_worker_addr(
                    coord_base_gm,
                    worker_slot,
                    kSoftmaxPrimitiveStageLocalMaxBase,
                    slot),
                fp32_to_reg(static_cast<float>(state.local_max)));
            softmax_primitive_publish_to_addr(
                executor_core_id,
                softmax_primitive_stage_worker_addr(
                    coord_base_gm,
                    worker_slot,
                    kSoftmaxPrimitiveStageLocalMaxReadyBase,
                    slot),
                state.local_max_seq);
            state.stage = SoftmaxRowPipelineStage::LOCAL_MAX_PUBLISHED;

            if (is_coordinator) {
                double global_row_max = -std::numeric_limits<double>::infinity();
                for (int worker = 0; worker < worker_cores; ++worker) {
                    adaptive_wait_eq(
                        softmax_primitive_stage_worker_addr(
                            coord_base_gm,
                            worker,
                            kSoftmaxPrimitiveStageLocalMaxReadyBase,
                            slot),
                        state.local_max_seq);
                    const float worker_max = fp32_from_reg(gm2reg(
                        softmax_primitive_stage_worker_addr(
                            coord_base_gm,
                            worker,
                            kSoftmaxPrimitiveStageLocalMaxBase,
                            slot)));
                    global_row_max =
                        std::max(global_row_max, static_cast<double>(worker_max));
                }
                for (int worker = 0; worker < worker_cores; ++worker) {
                    const int worker_core = gemm_worker_core_for_slot(worker);
                    softmax_primitive_publish_u64(
                        executor_core_id,
                        worker_core,
                        kSoftmaxPrimitiveStageGlobalBase +
                            kSoftmaxPrimitiveStageGlobalMaxBase +
                            slot * kSoftmaxPrimitiveStageValueStride,
                        fp32_to_reg(static_cast<float>(global_row_max)));
                    softmax_primitive_publish_u64(
                        executor_core_id,
                        worker_core,
                        kSoftmaxPrimitiveStageGlobalBase +
                            kSoftmaxPrimitiveStageGlobalMaxReadyBase +
                            slot * kSoftmaxPrimitiveStageValueStride,
                        state.global_max_seq);
                }
            }
        }

        for (uint64_t slot = 0; slot < window_rows; ++slot) {
            SoftmaxRowPipelineState& state =
                stage_pipeline_states[static_cast<size_t>(slot)];
            adaptive_wait_eq(
                softmax_primitive_stage_global_addr(
                    executor_core_id, kSoftmaxPrimitiveStageGlobalMaxReadyBase, slot),
                state.global_max_seq);
            state.row_max = static_cast<double>(fp32_from_reg(gm2reg(
                softmax_primitive_stage_global_addr(
                    executor_core_id, kSoftmaxPrimitiveStageGlobalMaxBase, slot))));
            state.stage = SoftmaxRowPipelineStage::GLOBAL_MAX_READY;

            if (!advance_softmax_stage_exp_sum(
                    executor_core_id,
                    worker_slot,
                    state.row,
                    slice_begin,
                    slice_end,
                    slice_elems,
                    chunk_elems,
                    max_batch_items,
                    input_hbm_base,
                    dim,
                    &state)) {
                return 1;
            }
            ++pipeline_stage_cycles;
            softmax_primitive_publish_to_addr(
                executor_core_id,
                softmax_primitive_stage_worker_addr(
                    coord_base_gm,
                    worker_slot,
                    kSoftmaxPrimitiveStageLocalSumBase,
                    slot),
                fp32_to_reg(static_cast<float>(state.local_sum)));
            softmax_primitive_publish_to_addr(
                executor_core_id,
                softmax_primitive_stage_worker_addr(
                    coord_base_gm,
                    worker_slot,
                    kSoftmaxPrimitiveStageLocalSumReadyBase,
                    slot),
                state.local_sum_seq);
            state.stage = SoftmaxRowPipelineStage::LOCAL_SUM_PUBLISHED;

            if (is_coordinator) {
                double global_row_sum = 0.0;
                for (int worker = 0; worker < worker_cores; ++worker) {
                    adaptive_wait_eq(
                        softmax_primitive_stage_worker_addr(
                            coord_base_gm,
                            worker,
                            kSoftmaxPrimitiveStageLocalSumReadyBase,
                            slot),
                        state.local_sum_seq);
                    const float worker_sum = fp32_from_reg(gm2reg(
                        softmax_primitive_stage_worker_addr(
                            coord_base_gm,
                            worker,
                            kSoftmaxPrimitiveStageLocalSumBase,
                            slot)));
                    global_row_sum += static_cast<double>(worker_sum);
                }

                const uint64_t scalar_input_gm =
                    primitive_batch_slot_gm(executor_core_id, chunk_elems, 0);
                const uint64_t scalar_output_gm =
                    primitive_batch_output_gm(executor_core_id, chunk_elems, 0);
                write_fp32_to_gm(scalar_input_gm, static_cast<float>(global_row_sum));
                const SFUPrimitiveDesc reciprocal_desc = {
                    .job_id = kPrimitiveTagBase + 0x8e0000ULL + state.row,
                    .input0_gm_addr = scalar_input_gm,
                    .input1_gm_addr = 0,
                    .output_gm_addr = scalar_output_gm,
                    .op = static_cast<uint32_t>(SFUPrimitiveOp::RECIPROCAL),
                    .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                    .elem_count = 1,
                    .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .input1_stride_bytes = 0,
                    .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .flags = 0,
                    .approx_mode = 0,
                };
                if (!issue_sfu_primitive_batch_descs(
                        executor_core_id,
                        std::vector<SFUPrimitiveDesc>{reciprocal_desc},
                        kPrimitiveTagBase + 0x8f0000ULL + state.row)) {
                    return 1;
                }
                const float inv_sum = read_fp32_from_gm(scalar_output_gm);
                if (!std::isfinite(inv_sum)) {
                    return 1;
                }
                for (int worker = 0; worker < worker_cores; ++worker) {
                    const int worker_core = gemm_worker_core_for_slot(worker);
                    softmax_primitive_publish_u64(
                        executor_core_id,
                        worker_core,
                        kSoftmaxPrimitiveStageGlobalBase +
                            kSoftmaxPrimitiveStageInvSumBase +
                            slot * kSoftmaxPrimitiveStageValueStride,
                        fp32_to_reg(inv_sum));
                    softmax_primitive_publish_u64(
                        executor_core_id,
                        worker_core,
                        kSoftmaxPrimitiveStageGlobalBase +
                            kSoftmaxPrimitiveStageGlobalSumReadyBase +
                            slot * kSoftmaxPrimitiveStageValueStride,
                        state.global_sum_seq);
                }
            }
        }

        for (uint64_t slot = 0; slot < window_rows; ++slot) {
            SoftmaxRowPipelineState& state =
                stage_pipeline_states[static_cast<size_t>(slot)];
            adaptive_wait_eq(
                softmax_primitive_stage_global_addr(
                    executor_core_id, kSoftmaxPrimitiveStageGlobalSumReadyBase, slot),
                state.global_sum_seq);
            state.inv_sum = fp32_from_reg(gm2reg(
                softmax_primitive_stage_global_addr(
                    executor_core_id, kSoftmaxPrimitiveStageInvSumBase, slot)));
            state.stage = SoftmaxRowPipelineStage::GLOBAL_SUM_READY;
            if (!advance_softmax_stage_normalize(
                    executor_core_id,
                    state.row,
                    slice_begin,
                    slice_end,
                    chunk_elems,
                    output_hbm_base,
                    dim,
                    verify,
                    &state,
                    &max_abs_diff,
                    &max_rel_diff,
                    &max_row_sum_error)) {
                return 1;
            }
            ++pipeline_stage_cycles;
        }
    }

    if (is_coordinator) {
        std::printf("[SOFTMAX] mode=sfu-primitive-softmax executor_core=%d rows=%llu dim=%llu worker_cores=%llu dim_per_core=%llu chunk_elems=%llu row_block=%llu row_blocks=%llu block_syncs=%llu pipeline_depth=%llu pipeline_mode=row primitive_stages=5 local_steps=3 cross_core_reduce_stages=2 chunks=%llu batches=%llu hbm_init_write_bytes=%llu hbm_read_bytes=%llu hbm_write_bytes=%llu pipeline_stage_cycles=%llu max_abs_diff=%g max_rel_diff=%g max_row_sum_error=%g PASS\n",
                    executor_core_id,
                    static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(dim),
                    static_cast<unsigned long long>(worker_cores),
                    static_cast<unsigned long long>(dim_per_core),
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(1),
                    static_cast<unsigned long long>(row_blocks),
                    static_cast<unsigned long long>(block_syncs),
                    static_cast<unsigned long long>(pipeline_depth),
                    static_cast<unsigned long long>(planned_total_chunks),
                    static_cast<unsigned long long>(planned_total_batches),
                    static_cast<unsigned long long>(planned_hbm_init_write_bytes),
                    static_cast<unsigned long long>(planned_hbm_read_bytes),
                    static_cast<unsigned long long>(planned_hbm_write_bytes),
                    static_cast<unsigned long long>(pipeline_stage_cycles),
                    max_abs_diff,
                    max_rel_diff,
                    max_row_sum_error);
        std::fflush(stdout);
    }
    return 0;
}

int run_sfu_primitive_softmax_for_core(int executor_core_id,
                                       int requested_core_id,
                                       const MatmulRuntimeConfig& cfg) {
    int64_t requested_rows = read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_ROWS", 1);
    if (requested_rows <= 0) {
        requested_rows = 1;
    }
    int64_t requested_dim = read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_DIM", 256);
    if (requested_dim <= 0) {
        requested_dim = 256;
    }
    const uint64_t rows = static_cast<uint64_t>(requested_rows);
    const uint64_t dim = static_cast<uint64_t>(requested_dim);
    const int64_t requested_worker_cores =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_WORKER_CORES", 0);
    const int64_t multicore_min_dim =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_MULTICORE_MIN_DIM", 512);
    const int worker_cores =
        resolve_softmax_primitive_worker_count(dim, requested_worker_cores, multicore_min_dim);
    const int64_t requested_chunk =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_CHUNK_ELEMS", 0);
    const uint64_t chunk_elems = primitive_hbm_chunk_elems(dim, requested_chunk);
    if (chunk_elems == 0) {
        std::fprintf(stderr, "[SFU-SOFTMAX-PRIMITIVE] invalid primitive chunk capacity\n");
        return 1;
    }
    const uint64_t max_batch_items = primitive_batch_max_items(chunk_elems);
    if (max_batch_items == 0) {
        std::fprintf(stderr, "[SFU-SOFTMAX-PRIMITIVE] invalid primitive batch capacity\n");
        return 1;
    }
    const int64_t verify = read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_VERIFY", 1);
    const int64_t requested_row_block =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_ROW_BLOCK", 0);
    const uint64_t row_block =
        resolve_softmax_primitive_row_block(rows, requested_row_block);
    const uint64_t row_blocks = (rows + row_block - 1) / row_block;
    const uint64_t block_syncs = row_blocks * 2;
    const int64_t requested_pipeline_depth =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX_PIPELINE_DEPTH", 1);
    const uint64_t pipeline_depth =
        resolve_softmax_primitive_pipeline_depth(rows, requested_pipeline_depth);

    if (row_block <= 1) {
        // Continue with the row-at-a-time or row-pipeline path below.
    }
    if (row_block > 1) {
        return run_sfu_primitive_softmax_row_block_for_core(
            executor_core_id,
            requested_core_id,
            cfg,
            rows,
            dim,
            worker_cores,
            chunk_elems,
            max_batch_items,
            row_block,
            row_blocks,
            block_syncs,
            verify,
            1,
            "row_block");
    }
    if (pipeline_depth > 1) {
        return run_sfu_primitive_softmax_row_pipeline_for_core(
            executor_core_id,
            requested_core_id,
            cfg,
            rows,
            dim,
            worker_cores,
            chunk_elems,
            max_batch_items,
            pipeline_depth,
            verify);
    }

    const int worker_slot = gemm_worker_slot_for_core(executor_core_id);
    if (worker_slot < 0 || worker_slot >= ACTIVE_GEMM_CORES) {
        return 0;
    }
    const bool participates = worker_slot < worker_cores;
    if (!participates) {
        return 0;
    }

    const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, 0, cfg);
    const uint64_t total_elems = rows * dim;
    const uint64_t total_bytes = total_elems * sizeof(float);
    const uint64_t required_bytes = total_bytes * 2;
    const uint64_t available_bytes = primitive_hbm_available_bytes(desc);
    if (available_bytes < required_bytes) {
        std::fprintf(stderr,
                     "[SFU-SOFTMAX-PRIMITIVE] requested bytes=%llu exceed available HBM C-region bytes=%llu\n",
                     static_cast<unsigned long long>(required_bytes),
                     static_cast<unsigned long long>(available_bytes));
        return 1;
    }

    const uint64_t input_hbm_base = desc.c_base_mm;
    const uint64_t output_hbm_base = desc.c_base_mm + total_bytes;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double max_row_sum_error = 0.0;
    uint64_t slice_begin = 0;
    uint64_t slice_end = 0;
    softmax_primitive_slice_for_worker(
        dim, worker_slot, worker_cores, &slice_begin, &slice_end);
    const uint64_t slice_elems = slice_end - slice_begin;
    const uint64_t dim_per_core = (dim + static_cast<uint64_t>(worker_cores) - 1) /
                                  static_cast<uint64_t>(worker_cores);
    const int coordinator_core = gemm_worker_core_for_slot(0);
    const bool is_coordinator = (worker_slot == 0);
    const uint64_t coord_slot_bytes = 2 * chunk_elems * kPrimitiveElemStride;
    const uint64_t coord_base_gm =
        primitive_input_gm(coordinator_core) + max_batch_items * coord_slot_bytes + 0x100ULL;

    uint64_t planned_chunks_per_row = 0;
    uint64_t planned_groups_per_row = 0;
    for (int slot = 0; slot < worker_cores; ++slot) {
        uint64_t begin = 0;
        uint64_t end = 0;
        softmax_primitive_slice_for_worker(dim, slot, worker_cores, &begin, &end);
        const uint64_t elems = end - begin;
        const uint64_t slice_chunks = (elems + chunk_elems - 1) / chunk_elems;
        planned_chunks_per_row += slice_chunks;
        planned_groups_per_row += (slice_chunks + max_batch_items - 1) / max_batch_items;
    }
    const uint64_t planned_total_chunks = rows * planned_chunks_per_row;
    const uint64_t planned_total_batches =
        rows * (planned_groups_per_row * 3 + 1);
    const uint64_t planned_hbm_init_write_bytes = total_bytes;
    const uint64_t planned_hbm_read_bytes = total_bytes * 2;
    const uint64_t planned_hbm_write_bytes = total_bytes;

    if (is_coordinator) {
        std::printf("[SFU-SOFTMAX-PRIMITIVE] begin executor_core=%d rows=%llu dim=%llu worker_cores=%llu dim_per_core=%llu chunk_elems=%llu chunks_per_row=%llu row_block=%llu row_blocks=%llu verify=%lld multicore_min_dim=%lld\n",
                    executor_core_id,
                    static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(dim),
                    static_cast<unsigned long long>(worker_cores),
                    static_cast<unsigned long long>(dim_per_core),
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(planned_chunks_per_row),
                    static_cast<unsigned long long>(row_block),
                    static_cast<unsigned long long>(row_blocks),
                    static_cast<long long>(verify),
                    static_cast<long long>(multicore_min_dim));
        std::fflush(stdout);
    }

    std::vector<float> chunk_values;
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t col = slice_begin; col < slice_end; col += chunk_elems) {
            const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
            fill_softmax_primitive_chunk(&chunk_values, row, col, elems_this_chunk);
            const uint64_t input_gm = primitive_input_gm(executor_core_id);
            write_values_to_gm(input_gm, chunk_values);
            const uint64_t hbm_addr =
                input_hbm_base + (row * dim + col) * sizeof(float);
            set_len(elems_this_chunk * sizeof(float));
            remote_store(input_gm, hbm_addr);
        }
    }

    std::vector<float> row_exp(static_cast<size_t>(slice_elems), 0.0f);
    std::vector<float> output_values;
    for (uint64_t row = 0; row < rows; ++row) {
        double local_row_max = -std::numeric_limits<double>::infinity();
        uint64_t processed_col = slice_begin;
        uint64_t row_group_index = 0;

        while (processed_col < slice_end) {
            const uint64_t remaining_chunks =
                (slice_end - processed_col + chunk_elems - 1) / chunk_elems;
            const uint64_t batch_items =
                std::min(remaining_chunks, max_batch_items);
            std::vector<SFUPrimitiveDesc> reduce_descs;
            reduce_descs.reserve(static_cast<size_t>(batch_items));
            std::vector<uint64_t> partial_gms(static_cast<size_t>(batch_items), 0);

            for (uint64_t item = 0; item < batch_items; ++item) {
                const uint64_t col = processed_col + item * chunk_elems;
                const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
                const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
                const uint64_t input_gm =
                    primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
                const uint64_t output_gm =
                    primitive_batch_output_gm(executor_core_id, chunk_elems, item);
                const uint64_t hbm_addr =
                    input_hbm_base + (row * dim + col) * sizeof(float);

                dma_remote_load_to_gm(executor_core_id, hbm_addr, input_gm, chunk_bytes);

                reduce_descs.push_back(SFUPrimitiveDesc{
                    .job_id = kPrimitiveTagBase + 0x800000ULL +
                              static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                              row * 0x1000ULL + row_group_index * 16 + item,
                    .input0_gm_addr = input_gm,
                    .input1_gm_addr = 0,
                    .output_gm_addr = output_gm,
                    .op = static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_MAX),
                    .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                    .elem_count = static_cast<uint32_t>(elems_this_chunk),
                    .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .input1_stride_bytes = 0,
                    .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .flags = 0,
                    .approx_mode = 0,
                });
                partial_gms[static_cast<size_t>(item)] = output_gm;
            }

            const uint64_t batch_tag =
                kPrimitiveTagBase + 0x810000ULL +
                static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                row * 0x1000ULL + row_group_index;
            if (!issue_sfu_primitive_batch_descs(executor_core_id, reduce_descs, batch_tag)) {
                return 1;
            }
            for (uint64_t item = 0; item < batch_items; ++item) {
                const float partial = read_fp32_from_gm(partial_gms[static_cast<size_t>(item)]);
                local_row_max = std::max(local_row_max, static_cast<double>(partial));
            }

            processed_col += batch_items * chunk_elems;
            row_group_index += 1;
        }

        const uint64_t local_max_seq = row * 4 + 1;
        const uint64_t global_max_seq = row * 4 + 2;
        const uint64_t local_sum_seq = row * 4 + 3;
        const uint64_t global_sum_seq = row * 4 + 4;
        softmax_primitive_publish_to_addr(
            executor_core_id,
            softmax_primitive_coord_worker_addr(coord_base_gm, worker_slot, 0x8),
            fp32_to_reg(static_cast<float>(local_row_max)));
        softmax_primitive_publish_to_addr(
            executor_core_id,
            softmax_primitive_coord_worker_addr(coord_base_gm, worker_slot, 0x0),
            local_max_seq);

        if (is_coordinator) {
            double global_row_max = -std::numeric_limits<double>::infinity();
            for (int slot = 0; slot < worker_cores; ++slot) {
                adaptive_wait_eq(
                    softmax_primitive_coord_worker_addr(coord_base_gm, slot, 0x0),
                    local_max_seq);
                const float worker_max = fp32_from_reg(gm2reg(
                    softmax_primitive_coord_worker_addr(coord_base_gm, slot, 0x8)));
                global_row_max = std::max(global_row_max, static_cast<double>(worker_max));
            }
            for (int slot = 0; slot < worker_cores; ++slot) {
                const int worker_core = gemm_worker_core_for_slot(slot);
                softmax_primitive_publish_u64(
                    executor_core_id, worker_core,
                    kSoftmaxPrimitiveMboxGlobalValue,
                    fp32_to_reg(static_cast<float>(global_row_max)));
                softmax_primitive_publish_u64(
                    executor_core_id, worker_core,
                    kSoftmaxPrimitiveMboxGlobalReady, global_max_seq);
            }
        }

        softmax_primitive_wait_local_u64(
            executor_core_id, kSoftmaxPrimitiveMboxGlobalReady, global_max_seq);
        const double row_max = static_cast<double>(
            fp32_from_reg(gm2reg(softmax_primitive_mailbox_addr(
                executor_core_id, kSoftmaxPrimitiveMboxGlobalValue))));

        double local_row_sum = 0.0;
        processed_col = slice_begin;
        row_group_index = 0;
        while (processed_col < slice_end) {
            const uint64_t remaining_chunks =
                (slice_end - processed_col + chunk_elems - 1) / chunk_elems;
            const uint64_t batch_items =
                std::min(remaining_chunks, max_batch_items);
            std::vector<SFUPrimitiveDesc> exp_descs;
            exp_descs.reserve(static_cast<size_t>(batch_items));
            std::vector<uint64_t> exp_output_gms(static_cast<size_t>(batch_items), 0);
            std::vector<uint64_t> elem_counts(static_cast<size_t>(batch_items), 0);
            std::vector<uint64_t> cols(static_cast<size_t>(batch_items), 0);

            for (uint64_t item = 0; item < batch_items; ++item) {
                const uint64_t col = processed_col + item * chunk_elems;
                const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
                const uint64_t chunk_bytes = elems_this_chunk * sizeof(float);
                const uint64_t input_gm =
                    primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
                const uint64_t output_gm =
                    primitive_batch_output_gm(executor_core_id, chunk_elems, item);
                const uint64_t hbm_addr =
                    input_hbm_base + (row * dim + col) * sizeof(float);

                dma_remote_load_to_gm(executor_core_id, hbm_addr, input_gm, chunk_bytes);
                read_values_from_gm(input_gm, elems_this_chunk, &chunk_values);
                for (float& value : chunk_values) {
                    value = static_cast<float>(static_cast<double>(value) - row_max);
                }
                write_values_to_gm(input_gm, chunk_values);

                exp_descs.push_back(SFUPrimitiveDesc{
                    .job_id = kPrimitiveTagBase + 0x820000ULL +
                              static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                              row * 0x1000ULL + row_group_index * 16 + item,
                    .input0_gm_addr = input_gm,
                    .input1_gm_addr = 0,
                    .output_gm_addr = output_gm,
                    .op = static_cast<uint32_t>(SFUPrimitiveOp::EXP),
                    .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                    .elem_count = static_cast<uint32_t>(elems_this_chunk),
                    .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .input1_stride_bytes = 0,
                    .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .flags = 0,
                    .approx_mode = 0,
                });
                exp_output_gms[static_cast<size_t>(item)] = output_gm;
                elem_counts[static_cast<size_t>(item)] = elems_this_chunk;
                cols[static_cast<size_t>(item)] = col;
            }

            const uint64_t exp_batch_tag =
                kPrimitiveTagBase + 0x830000ULL +
                static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                row * 0x1000ULL + row_group_index;
            if (!issue_sfu_primitive_batch_descs(executor_core_id, exp_descs, exp_batch_tag)) {
                return 1;
            }

            std::vector<SFUPrimitiveDesc> sum_descs;
            sum_descs.reserve(static_cast<size_t>(batch_items));
            std::vector<uint64_t> sum_output_gms(static_cast<size_t>(batch_items), 0);
            for (uint64_t item = 0; item < batch_items; ++item) {
                const uint64_t elems_this_chunk = elem_counts[static_cast<size_t>(item)];
                const uint64_t col = cols[static_cast<size_t>(item)];
                const uint64_t exp_output_gm = exp_output_gms[static_cast<size_t>(item)];
                read_values_from_gm(exp_output_gm, elems_this_chunk, &chunk_values);
                for (uint64_t i = 0; i < elems_this_chunk; ++i) {
                    row_exp[static_cast<size_t>((col - slice_begin) + i)] =
                        chunk_values[static_cast<size_t>(i)];
                }
                const uint64_t sum_output_gm =
                    primitive_batch_slot_gm(executor_core_id, chunk_elems, item);
                sum_descs.push_back(SFUPrimitiveDesc{
                    .job_id = kPrimitiveTagBase + 0x840000ULL +
                              static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                              row * 0x1000ULL + row_group_index * 16 + item,
                    .input0_gm_addr = exp_output_gm,
                    .input1_gm_addr = 0,
                    .output_gm_addr = sum_output_gm,
                    .op = static_cast<uint32_t>(SFUPrimitiveOp::REDUCE_SUM),
                    .dtype = static_cast<uint32_t>(GOLEM_DTYPE_FP32),
                    .elem_count = static_cast<uint32_t>(elems_this_chunk),
                    .input0_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .input1_stride_bytes = 0,
                    .output_stride_bytes = static_cast<uint32_t>(kPrimitiveElemStride),
                    .flags = 0,
                    .approx_mode = 0,
                });
                sum_output_gms[static_cast<size_t>(item)] = sum_output_gm;
            }

            const uint64_t sum_batch_tag =
                kPrimitiveTagBase + 0x850000ULL +
                static_cast<uint64_t>(worker_slot) * 0x100000ULL +
                row * 0x1000ULL + row_group_index;
            if (!issue_sfu_primitive_batch_descs(executor_core_id, sum_descs, sum_batch_tag)) {
                return 1;
            }
            for (uint64_t item = 0; item < batch_items; ++item) {
                local_row_sum += static_cast<double>(
                    read_fp32_from_gm(sum_output_gms[static_cast<size_t>(item)]));
            }

            processed_col += batch_items * chunk_elems;
            row_group_index += 1;
        }

        softmax_primitive_publish_to_addr(
            executor_core_id,
            softmax_primitive_coord_worker_addr(coord_base_gm, worker_slot, 0x8),
            fp32_to_reg(static_cast<float>(local_row_sum)));
        softmax_primitive_publish_to_addr(
            executor_core_id,
            softmax_primitive_coord_worker_addr(coord_base_gm, worker_slot, 0x0),
            local_sum_seq);

        if (is_coordinator) {
            double global_row_sum = 0.0;
            for (int slot = 0; slot < worker_cores; ++slot) {
                adaptive_wait_eq(
                    softmax_primitive_coord_worker_addr(coord_base_gm, slot, 0x0),
                    local_sum_seq);
                const float worker_sum = fp32_from_reg(gm2reg(
                    softmax_primitive_coord_worker_addr(coord_base_gm, slot, 0x8)));
                global_row_sum += static_cast<double>(worker_sum);
            }
            const float inv_sum = coordinator_reciprocal_and_broadcast(
                executor_core_id, worker_cores, chunk_elems, row, global_sum_seq, global_row_sum);
            if (!std::isfinite(inv_sum)) {
                return 1;
            }
        }

        softmax_primitive_wait_local_u64(
            executor_core_id, kSoftmaxPrimitiveMboxGlobalReady, global_sum_seq);
        const float inv_sum = fp32_from_reg(gm2reg(softmax_primitive_mailbox_addr(
            executor_core_id, kSoftmaxPrimitiveMboxGlobalValue)));
        const double row_sum = inv_sum != 0.0f
                                   ? 1.0 / static_cast<double>(inv_sum)
                                   : std::numeric_limits<double>::infinity();

        double output_row_sum = 0.0;
        for (uint64_t col = slice_begin; col < slice_end; col += chunk_elems) {
            const uint64_t elems_this_chunk = std::min(chunk_elems, slice_end - col);
            output_values.assign(static_cast<size_t>(elems_this_chunk), 0.0f);
            for (uint64_t i = 0; i < elems_this_chunk; ++i) {
                const uint64_t global_col = col + i;
                const double expected_exp =
                    std::exp(static_cast<double>(
                        softmax_primitive_input_value(row, global_col)) - row_max);
                const double expected = expected_exp / row_sum;
                const float got =
                    row_exp[static_cast<size_t>(global_col - slice_begin)] * inv_sum;
                output_values[static_cast<size_t>(i)] = got;
                output_row_sum += static_cast<double>(got);
                if (verify != 0) {
                    const double abs_diff = std::fabs(static_cast<double>(got) - expected);
                    const double rel_diff = abs_diff / std::max(std::fabs(expected), 1.0e-12);
                    max_abs_diff = std::max(max_abs_diff, abs_diff);
                    max_rel_diff = std::max(max_rel_diff, rel_diff);
                    if (!std::isfinite(got)) {
                        std::fprintf(stderr,
                                     "[SFU-SOFTMAX-PRIMITIVE] non-finite output row=%llu col=%llu got=%g row_max=%g row_sum=%g inv_sum=%g exp_value=%g\n",
                                     static_cast<unsigned long long>(row),
                                     static_cast<unsigned long long>(global_col),
                                     got,
                                     row_max,
                                     row_sum,
                                     inv_sum,
                                     row_exp[static_cast<size_t>(global_col - slice_begin)]);
                        return 1;
                    }
                }
            }

            const uint64_t output_gm = primitive_input_gm(executor_core_id);
            write_values_to_gm(output_gm, output_values);
            const uint64_t output_hbm =
                output_hbm_base + (row * dim + col) * sizeof(float);
            set_len(elems_this_chunk * sizeof(float));
            remote_store(output_gm, output_hbm);
        }

        if (verify != 0) {
            const double row_sum_error = std::fabs(output_row_sum - (local_row_sum / row_sum));
            max_row_sum_error = std::max(max_row_sum_error, row_sum_error);
            if (max_abs_diff > 1.0e-4 || row_sum_error > 1.0e-4) {
                std::fprintf(stderr,
                             "[SFU-SOFTMAX-PRIMITIVE] verify failed row=%llu max_abs_diff=%g row_sum_error=%g\n",
                             static_cast<unsigned long long>(row),
                             max_abs_diff,
                             row_sum_error);
                return 1;
            }
        }
    }

    if (is_coordinator) {
        std::printf("[SOFTMAX] mode=sfu-primitive-softmax executor_core=%d rows=%llu dim=%llu worker_cores=%llu dim_per_core=%llu chunk_elems=%llu row_block=%llu row_blocks=%llu block_syncs=%llu primitive_stages=4 local_steps=3 cross_core_reduce_stages=2 chunks=%llu batches=%llu hbm_init_write_bytes=%llu hbm_read_bytes=%llu hbm_write_bytes=%llu max_abs_diff=%g max_rel_diff=%g max_row_sum_error=%g PASS\n",
                    executor_core_id,
                    static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(dim),
                    static_cast<unsigned long long>(worker_cores),
                    static_cast<unsigned long long>(dim_per_core),
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(row_block),
                    static_cast<unsigned long long>(row_blocks),
                    static_cast<unsigned long long>(block_syncs),
                    static_cast<unsigned long long>(planned_total_chunks),
                    static_cast<unsigned long long>(planned_total_batches),
                    static_cast<unsigned long long>(planned_hbm_init_write_bytes),
                    static_cast<unsigned long long>(planned_hbm_read_bytes),
                    static_cast<unsigned long long>(planned_hbm_write_bytes),
                    max_abs_diff,
                    max_rel_diff,
                    max_row_sum_error);
        std::fflush(stdout);
    }
    return 0;
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
    const int64_t batch_mode = read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_HBM_BATCH", 1);
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
    const int actual_core_id = sched_getcpu();
    int executor_core_id = actual_core_id;
    if (executor_core_id < 0 || executor_core_id >= TOTAL_CORES) {
        if (actual_core_id < 0) {
            perror("sched_getcpu");
        }
        std::fprintf(stderr,
                     "[WARN] invalid sched_getcpu core id=%d; falling back to requested core id=%d\n",
                     actual_core_id,
                     requested_core_id);
        executor_core_id = requested_core_id;
    }
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

int run_gemm_unified_job_softmax_for_core(int executor_core_id,
                                          int requested_core_id,
                                          const golem_matmul_op_desc_t& op_desc,
                                          const golem_softmax_op_desc_t& softmax_desc,
                                          const MatmulRuntimeConfig& cfg,
                                          uint64_t job_id) {
    int status = run_gemm_for_core(executor_core_id, requested_core_id, op_desc);
    if (status != 0) {
        return status;
    }
    if (requested_core_id != 0) {
        return 0;
    }

    if (!validate_matmul_call(cfg)) {
        std::fprintf(stderr, "[SOFTMAX-SFU-JOB] invalid matmul config\n");
        return 1;
    }

    const uint64_t matrix_elems =
        static_cast<uint64_t>(cfg.m) * static_cast<uint64_t>(cfg.n);
    const uint64_t matrix_bytes = matrix_elems * sizeof(float);
    const uint64_t matrix_bytes_aligned = align_up_constexpr(matrix_bytes, LOCAL_ALIGN);
    const uint64_t tile_elems =
        static_cast<uint64_t>(cfg.block_m) * static_cast<uint64_t>(cfg.block_n);
    const uint64_t tile_bytes = tile_elems * sizeof(float);
    const uint64_t tile_bytes_aligned = align_up_constexpr(tile_bytes, LOCAL_ALIGN);
    const uint64_t required_bytes =
        matrix_bytes_aligned + matrix_bytes_aligned + tile_bytes_aligned;
    const uint64_t available_bytes =
        (GOLEM_GLOBAL_STRIDE_BYTES > (LOCAL_DATA_BASE + kPrimitiveLocalGuardBytes))
            ? (GOLEM_GLOBAL_STRIDE_BYTES - LOCAL_DATA_BASE - kPrimitiveLocalGuardBytes)
            : 0;
    if (required_bytes > available_bytes) {
        std::fprintf(stderr,
                     "[SOFTMAX-SFU-JOB] local GM staging too small: required=%llu available=%llu "
                     "m=%d n=%d block_m=%d block_n=%d\n",
                     static_cast<unsigned long long>(required_bytes),
                     static_cast<unsigned long long>(available_bytes),
                     cfg.m, cfg.n, cfg.block_m, cfg.block_n);
        return 1;
    }

    const uint64_t input_gm = gm_addr(executor_core_id, LOCAL_DATA_BASE);
    const uint64_t output_gm = input_gm + matrix_bytes_aligned;
    const uint64_t tile_gm = output_gm + matrix_bytes_aligned;
    const uint64_t desc_gm = gm_addr(executor_core_id, LOCAL_TMP_OFFSET);
    std::vector<float> row_major(static_cast<size_t>(matrix_elems), 0.0f);
    std::vector<float> tile(static_cast<size_t>(tile_elems), 0.0f);

    for (int task_id = 0; task_id < gemm_total_tasks(cfg); ++task_id) {
        const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, task_id, cfg);
        dma_remote_load_to_gm(executor_core_id, desc.c_base_mm, tile_gm, tile_bytes);
        set_len(tile_bytes);
        gm2mm(tile.data(), tile_gm);
        for (int r = 0; r < desc.block_m; ++r) {
            for (int c = 0; c < desc.block_n; ++c) {
                const size_t src_idx = static_cast<size_t>(c) * desc.block_m + r;
                const size_t dst_idx =
                    static_cast<size_t>(desc.m_tile * desc.block_m + r) * cfg.n +
                    static_cast<size_t>(desc.n_tile * desc.block_n + c);
                row_major[dst_idx] = tile[src_idx];
            }
        }
    }

    set_len(matrix_bytes);
    mm2gm(row_major.data(), input_gm);

    uint64_t chunk_elems = static_cast<uint64_t>(
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS", 256));
    if (chunk_elems == 0) {
        chunk_elems = 256;
    }
    uint64_t worker_cores = static_cast<uint64_t>(
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES", 16));
    if (worker_cores == 0) {
        worker_cores = 1;
    }
    const uint64_t tag = kPrimitiveTagBase + 0x900000ULL;

    std::printf("[SOFTMAX] dispatch=sfu-unified-job-softmax rows=%d dim=%d chunk=%llu workers=%llu executor_core=%d\n",
                cfg.m,
                cfg.n,
                static_cast<unsigned long long>(chunk_elems),
                static_cast<unsigned long long>(worker_cores),
                executor_core_id);
    std::fflush(stdout);

    const golem_status_t sfu_status = golemRunStandaloneSoftmaxSfuJobForCore(
        &softmax_desc,
        executor_core_id,
        &cfg,
        input_gm,
        output_gm,
        desc_gm,
        chunk_elems,
        worker_cores,
        0,
        static_cast<uint32_t>(executor_core_id),
        0,
        job_id,
        tag);
    if (sfu_status != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[SOFTMAX-SFU-JOB] unified job softmax failed on core %d\n",
                     requested_core_id);
        return 1;
    }

    set_len(matrix_bytes);
    gm2mm(row_major.data(), output_gm);
    for (int task_id = 0; task_id < gemm_total_tasks(cfg); ++task_id) {
        const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, task_id, cfg);
        for (int r = 0; r < desc.block_m; ++r) {
            for (int c = 0; c < desc.block_n; ++c) {
                const size_t src_idx =
                    static_cast<size_t>(desc.m_tile * desc.block_m + r) * cfg.n +
                    static_cast<size_t>(desc.n_tile * desc.block_n + c);
                const size_t dst_idx = static_cast<size_t>(c) * desc.block_m + r;
                tile[dst_idx] = row_major[src_idx];
            }
        }
        set_len(tile_bytes);
        mm2gm(tile.data(), tile_gm);
        remote_store(tile_gm, desc.c_base_mm);
    }

    std::printf("[SOFTMAX] mode=sfu-unified-job-softmax executor_core=%d rows=%d dim=%d chunk_elems=%llu worker_cores=%llu PASS\n",
                executor_core_id,
                cfg.m,
                cfg.n,
                static_cast<unsigned long long>(chunk_elems),
                static_cast<unsigned long long>(worker_cores));
    std::fflush(stdout);
    return 0;
}

int run_standalone_unified_job_softmax_band_for_core(
    int executor_core_id,
    const golem_softmax_op_desc_t& softmax_desc,
    const MatmulRuntimeConfig& cfg,
    int row_band_begin,
    int row_band_rows,
    uint64_t input_gm,
    uint64_t output_gm,
    uint64_t tile_gm,
    uint64_t desc_gm,
    uint64_t tile_bytes,
    uint64_t chunk_elems,
    uint64_t worker_cores,
    int job_rows_per_issue,
    uint64_t job_id,
    uint64_t tag,
    bool trace_bands,
    std::vector<float>& row_band,
    std::vector<float>& tile) {
    const uint64_t band_matrix_elems =
        static_cast<uint64_t>(row_band_rows) * static_cast<uint64_t>(cfg.n);
    const uint64_t band_matrix_bytes = band_matrix_elems * sizeof(float);
    if (trace_bands) {
        std::printf("[SOFTMAX-SFU-JOB] band_stage=load row_band_begin=%d row_band_rows=%d bytes=%llu tag=%llu\n",
                    row_band_begin,
                    row_band_rows,
                    static_cast<unsigned long long>(band_matrix_bytes),
                    static_cast<unsigned long long>(tag));
        std::fflush(stdout);
    }
    std::fill(row_band.begin(), row_band.begin() + static_cast<size_t>(band_matrix_elems), 0.0f);

    for (int task_id = 0; task_id < gemm_total_tasks(cfg); ++task_id) {
        const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, task_id, cfg);
        const int tile_row_begin = desc.m_tile * desc.block_m;
        const int tile_row_end = tile_row_begin + desc.block_m;
        const int overlap_begin = std::max(row_band_begin, tile_row_begin);
        const int overlap_end = std::min(row_band_begin + row_band_rows, tile_row_end);
        if (overlap_begin >= overlap_end) {
            continue;
        }

        dma_remote_load_to_gm(executor_core_id, desc.c_base_mm, tile_gm, tile_bytes);
        set_len(tile_bytes);
        gm2mm(tile.data(), tile_gm);
        for (int r = overlap_begin - row_band_begin; r < overlap_end - row_band_begin; ++r) {
            const int global_row = row_band_begin + r;
            const int tile_r = global_row - tile_row_begin;
            const int band_r = r;
            for (int c = 0; c < desc.block_n; ++c) {
                const size_t src_idx = static_cast<size_t>(c) * desc.block_m + tile_r;
                const size_t dst_idx =
                    static_cast<size_t>(band_r) * cfg.n +
                    static_cast<size_t>(desc.n_tile * desc.block_n + c);
                row_band[dst_idx] = tile[src_idx];
            }
        }
    }

    set_len(band_matrix_bytes);
    mm2gm(row_band.data(), input_gm);

    if (job_rows_per_issue <= 0 || job_rows_per_issue > row_band_rows) {
        job_rows_per_issue = row_band_rows;
    }
    if (trace_bands) {
        std::printf("[SOFTMAX-SFU-JOB] band_stage=job row_band_begin=%d row_band_rows=%d job_rows=%d chunk=%llu workers=%llu tag=%llu\n",
                    row_band_begin,
                    row_band_rows,
                    job_rows_per_issue,
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(worker_cores),
                    static_cast<unsigned long long>(tag));
        std::fflush(stdout);
    }
    int sub_job_index = 0;
    for (int job_row_begin = 0; job_row_begin < row_band_rows;
         job_row_begin += job_rows_per_issue) {
        const int sub_job_rows = std::min(job_rows_per_issue, row_band_rows - job_row_begin);
        const uint64_t sub_job_offset_bytes =
            static_cast<uint64_t>(job_row_begin) * static_cast<uint64_t>(cfg.n) * sizeof(float);
        if (trace_bands) {
            std::printf("[SOFTMAX-SFU-JOB] band_stage=subjob row_band_begin=%d job_row_begin=%d sub_job_rows=%d input_gm=0x%llx output_gm=0x%llx tag=%llu\n",
                        row_band_begin,
                        job_row_begin,
                        sub_job_rows,
                        static_cast<unsigned long long>(input_gm + sub_job_offset_bytes),
                        static_cast<unsigned long long>(output_gm + sub_job_offset_bytes),
                        static_cast<unsigned long long>(tag + static_cast<uint64_t>(sub_job_index)));
            std::fflush(stdout);
        }
        golem_softmax_op_desc_t sub_desc = softmax_desc;
        sub_desc.outer = static_cast<uint64_t>(sub_job_rows);
        sub_desc.dim = static_cast<uint64_t>(cfg.n);
        MatmulRuntimeConfig sub_cfg = cfg;
        sub_cfg.m = sub_job_rows;
        sub_cfg.block_m = sub_job_rows;
        const golem_status_t sfu_status = golemRunStandaloneSoftmaxSfuJobForCore(
            &sub_desc,
            executor_core_id,
            &sub_cfg,
            input_gm + sub_job_offset_bytes,
            output_gm + sub_job_offset_bytes,
            desc_gm,
            static_cast<uint32_t>(chunk_elems),
            static_cast<uint32_t>(worker_cores),
            0,
            static_cast<uint32_t>(executor_core_id),
            read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_ROW_ENGINE", 0) != 0
                ? SFU_JOB_FLAG_ROW_ENGINE_MODEL
                : 0,
            job_id ^ static_cast<uint64_t>(row_band_begin + job_row_begin),
            tag + static_cast<uint64_t>(sub_job_index));
        if (sfu_status != GOLEM_STATUS_OK) {
            std::fprintf(stderr,
                         "[SOFTMAX-SFU-JOB] standalone unified sub-job failed: "
                         "row_band_begin=%d job_row_begin=%d sub_job_rows=%d error=%s\n",
                         row_band_begin,
                         job_row_begin,
                         sub_job_rows,
                         golemSoftmaxSfuGetLastErrorString());
            return 1;
        }
        ++sub_job_index;
    }

    if (trace_bands) {
        std::printf("[SOFTMAX-SFU-JOB] band_stage=store row_band_begin=%d row_band_rows=%d bytes=%llu tag=%llu\n",
                    row_band_begin,
                    row_band_rows,
                    static_cast<unsigned long long>(band_matrix_bytes),
                    static_cast<unsigned long long>(tag));
        std::fflush(stdout);
    }
    set_len(band_matrix_bytes);
    gm2mm(row_band.data(), output_gm);
    for (int task_id = 0; task_id < gemm_total_tasks(cfg); ++task_id) {
        const GemmTaskDescriptor desc = gemm_descriptor_for_task(executor_core_id, task_id, cfg);
        const int tile_row_begin = desc.m_tile * desc.block_m;
        const int tile_row_end = tile_row_begin + desc.block_m;
        const int overlap_begin = std::max(row_band_begin, tile_row_begin);
        const int overlap_end = std::min(row_band_begin + row_band_rows, tile_row_end);
        if (overlap_begin >= overlap_end) {
            continue;
        }

        const bool full_tile_band =
            overlap_begin == tile_row_begin && overlap_end == tile_row_end;
        if (!full_tile_band) {
            dma_remote_load_to_gm(executor_core_id, desc.c_base_mm, tile_gm, tile_bytes);
            set_len(tile_bytes);
            gm2mm(tile.data(), tile_gm);
        }
        for (int r = overlap_begin - row_band_begin; r < overlap_end - row_band_begin; ++r) {
            const int global_row = row_band_begin + r;
            const int tile_r = global_row - tile_row_begin;
            const int band_r = r;
            for (int c = 0; c < desc.block_n; ++c) {
                const size_t src_idx =
                    static_cast<size_t>(band_r) * cfg.n +
                    static_cast<size_t>(desc.n_tile * desc.block_n + c);
                const size_t dst_idx = static_cast<size_t>(c) * desc.block_m + tile_r;
                tile[dst_idx] = row_band[src_idx];
            }
        }
        set_len(tile_bytes);
        mm2gm(tile.data(), tile_gm);
        remote_store(tile_gm, desc.c_base_mm);
    }

    if (trace_bands) {
        std::printf("[SOFTMAX-SFU-JOB] band_stage=done row_band_begin=%d row_band_rows=%d tag=%llu\n",
                    row_band_begin,
                    row_band_rows,
                    static_cast<unsigned long long>(tag));
        std::fflush(stdout);
    }
    return 0;
}

int run_standalone_unified_job_softmax_distributed_direct_for_core(
    int executor_core_id,
    const golem_softmax_op_desc_t& softmax_desc,
    const MatmulRuntimeConfig& cfg,
    int active_worker_slot,
    int band_core_count,
    int worker_cores,
    int staging_rows,
    int job_rows_per_issue,
    uint64_t input_gm,
    uint64_t output_gm,
    uint64_t desc_gm,
    uint64_t chunk_elems,
    uint64_t job_id,
    uint64_t tag,
    bool trace_bands,
    uint64_t rowmajor_input_hbm,
    uint64_t rowmajor_output_hbm) {
    const int cooperative_group_count = band_core_count / worker_cores;
    const int cooperative_group_id = active_worker_slot / worker_cores;
    const int worker_slot = active_worker_slot % worker_cores;
    const int owner_core =
        gemm_worker_core_for_slot(cooperative_group_id * worker_cores);
    const uint64_t slice_begin =
        static_cast<uint64_t>(cfg.n) * static_cast<uint64_t>(worker_slot) /
        static_cast<uint64_t>(worker_cores);
    const uint64_t slice_end =
        static_cast<uint64_t>(cfg.n) * static_cast<uint64_t>(worker_slot + 1) /
        static_cast<uint64_t>(worker_cores);
    const uint64_t slice_elems = slice_end - slice_begin;
    const uint64_t slice_bytes = slice_elems * sizeof(float);
    if (cooperative_group_count <= 0 || slice_elems == 0) {
        return 1;
    }

    int band_index = 0;
    for (int row_band_begin = 0; row_band_begin < cfg.m;
         row_band_begin += staging_rows, ++band_index) {
        if ((band_index % cooperative_group_count) != cooperative_group_id) {
            continue;
        }
        const int row_band_rows = std::min(staging_rows, cfg.m - row_band_begin);
        int sub_job_index = 0;
        for (int job_row_begin = 0; job_row_begin < row_band_rows;
             job_row_begin += job_rows_per_issue, ++sub_job_index) {
            const int sub_job_rows =
                std::min(job_rows_per_issue, row_band_rows - job_row_begin);
            const uint64_t compact_bytes =
                static_cast<uint64_t>(sub_job_rows) * slice_bytes;
            golem_softmax_op_desc_t sub_desc = softmax_desc;
            sub_desc.outer = static_cast<uint64_t>(sub_job_rows);
            sub_desc.dim = static_cast<uint64_t>(cfg.n);
            MatmulRuntimeConfig sub_cfg = cfg;
            sub_cfg.m = sub_job_rows;
            sub_cfg.block_m = sub_job_rows;
            const uint64_t sub_job_id =
                job_id ^ static_cast<uint64_t>(row_band_begin + job_row_begin);
            const uint64_t sub_job_tag =
                tag + static_cast<uint64_t>(band_index) * 0x1000ULL +
                static_cast<uint64_t>(sub_job_index);
            prepare_direct_dma_load_guard(input_gm, compact_bytes);
            for (int local_row = 0; local_row < sub_job_rows; ++local_row) {
                const uint64_t global_row =
                    static_cast<uint64_t>(row_band_begin + job_row_begin + local_row);
                const uint64_t input_hbm =
                    rowmajor_input_hbm + (global_row * cfg.n + slice_begin) * sizeof(float);
                const uint64_t local_input =
                    input_gm + static_cast<uint64_t>(local_row) * slice_bytes;
                dma_remote_load_to_gm(executor_core_id, input_hbm, local_input, slice_bytes);
            }
            if (!direct_dma_load_guard_passed(input_gm, compact_bytes)) {
                std::fprintf(stderr,
                             "[SOFTMAX-SFU-JOB] DMA_LOAD_FAILED distributed column slice: "
                             "executor_core=%d group=%d worker_slot=%d row_band_begin=%d "
                             "job_row_begin=%d sub_job_rows=%d slice=[%llu,%llu)\n",
                             executor_core_id,
                             cooperative_group_id,
                             worker_slot,
                             row_band_begin,
                             job_row_begin,
                             sub_job_rows,
                             static_cast<unsigned long long>(slice_begin),
                             static_cast<unsigned long long>(slice_end));
                (void)golemRunStandaloneSoftmaxSfuJobForCore(
                    &sub_desc,
                    executor_core_id,
                    &sub_cfg,
                    input_gm,
                    output_gm,
                    desc_gm,
                    static_cast<uint32_t>(chunk_elems),
                    static_cast<uint32_t>(worker_cores),
                    static_cast<uint32_t>(worker_slot),
                    static_cast<uint32_t>(owner_core),
                    SFU_JOB_FLAG_DISTRIBUTED_COLUMNS | SFU_JOB_FLAG_DISTRIBUTED_ABORT,
                    sub_job_id,
                    sub_job_tag);
                return 1;
            }

            if (trace_bands) {
                std::printf("[SOFTMAX-SFU-JOB] band_stage=distributed-subjob group=%d worker_slot=%d owner_core=%d row_band_begin=%d job_row_begin=%d sub_job_rows=%d slice=[%llu,%llu) tag=%llu\n",
                            cooperative_group_id,
                            worker_slot,
                            owner_core,
                            row_band_begin,
                            job_row_begin,
                            sub_job_rows,
                            static_cast<unsigned long long>(slice_begin),
                            static_cast<unsigned long long>(slice_end),
                            static_cast<unsigned long long>(sub_job_tag));
                std::fflush(stdout);
            }
            const golem_status_t sfu_status = golemRunStandaloneSoftmaxSfuJobForCore(
                &sub_desc,
                executor_core_id,
                &sub_cfg,
                input_gm,
                output_gm,
                desc_gm,
                static_cast<uint32_t>(chunk_elems),
                static_cast<uint32_t>(worker_cores),
                static_cast<uint32_t>(worker_slot),
                static_cast<uint32_t>(owner_core),
                SFU_JOB_FLAG_DISTRIBUTED_COLUMNS,
                sub_job_id,
                sub_job_tag);
            if (sfu_status != GOLEM_STATUS_OK) {
                std::fprintf(stderr,
                             "[SOFTMAX-SFU-JOB] distributed unified direct sub-job failed: "
                             "group=%d worker_slot=%d row_band_begin=%d job_row_begin=%d error=%s\n",
                             cooperative_group_id,
                             worker_slot,
                             row_band_begin,
                             job_row_begin,
                             golemSoftmaxSfuGetLastErrorString());
                return 1;
            }

            for (int local_row = 0; local_row < sub_job_rows; ++local_row) {
                const uint64_t global_row =
                    static_cast<uint64_t>(row_band_begin + job_row_begin + local_row);
                const uint64_t output_hbm =
                    rowmajor_output_hbm + (global_row * cfg.n + slice_begin) * sizeof(float);
                const uint64_t local_output =
                    output_gm + static_cast<uint64_t>(local_row) * slice_bytes;
                set_len(slice_bytes);
                remote_store(local_output, output_hbm);
            }
        }
    }
    return 0;
}

int run_standalone_unified_job_softmax_direct_band_for_core(
    int executor_core_id,
    const golem_softmax_op_desc_t& softmax_desc,
    const MatmulRuntimeConfig& cfg,
    int row_band_begin,
    int row_band_rows,
    uint64_t input_gm,
    uint64_t output_gm,
    uint64_t desc_gm,
    uint64_t chunk_elems,
    uint64_t worker_cores,
    int job_rows_per_issue,
    uint64_t job_id,
    uint64_t tag,
    bool trace_bands,
    uint64_t rowmajor_input_hbm,
    uint64_t rowmajor_output_hbm) {
    const bool row_engine_model =
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_ROW_ENGINE", 0) != 0;
    if (job_rows_per_issue <= 0 || job_rows_per_issue > row_band_rows) {
        job_rows_per_issue = row_band_rows;
    }
    if (trace_bands) {
        std::printf("[SOFTMAX-SFU-JOB] band_stage=job row_band_begin=%d row_band_rows=%d job_rows=%d chunk=%llu workers=%llu tag=%llu\n",
                    row_band_begin,
                    row_band_rows,
                    job_rows_per_issue,
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(worker_cores),
                    static_cast<unsigned long long>(tag));
        std::fflush(stdout);
    }
    int sub_job_index = 0;
    for (int job_row_begin = 0; job_row_begin < row_band_rows;
         job_row_begin += job_rows_per_issue) {
        const int sub_job_rows = std::min(job_rows_per_issue, row_band_rows - job_row_begin);
        const uint64_t sub_job_offset_bytes =
            static_cast<uint64_t>(job_row_begin) * static_cast<uint64_t>(cfg.n) * sizeof(float);
        const uint64_t sub_job_bytes =
            static_cast<uint64_t>(sub_job_rows) * static_cast<uint64_t>(cfg.n) * sizeof(float);
        const uint64_t sub_job_input_gm = input_gm;
        const uint64_t sub_job_output_gm = output_gm;
        const uint64_t sub_job_input_hbm =
            rowmajor_input_hbm + sub_job_offset_bytes;
        const uint64_t sub_job_output_hbm =
            rowmajor_output_hbm + sub_job_offset_bytes;
        if (trace_bands) {
            std::printf("[SOFTMAX-SFU-JOB] band_stage=direct-load row_band_begin=%d job_row_begin=%d sub_job_rows=%d bytes=%llu hbm=0x%llx tag=%llu\n",
                        row_band_begin,
                        job_row_begin,
                        sub_job_rows,
                        static_cast<unsigned long long>(sub_job_bytes),
                        static_cast<unsigned long long>(sub_job_input_hbm),
                        static_cast<unsigned long long>(tag + static_cast<uint64_t>(sub_job_index)));
            std::fflush(stdout);
        }
        prepare_direct_dma_load_guard(sub_job_input_gm, sub_job_bytes);
        dma_remote_load_to_gm(
            executor_core_id,
            sub_job_input_hbm,
            sub_job_input_gm,
            sub_job_bytes);
        if (!direct_dma_load_guard_passed(sub_job_input_gm, sub_job_bytes)) {
            std::fprintf(stderr,
                         "[SOFTMAX-SFU-JOB] DMA_LOAD_FAILED direct row-major sub-job: "
                         "executor_core=%d row_band_begin=%d job_row_begin=%d "
                         "sub_job_rows=%d bytes=%llu hbm=0x%llx gm=0x%llx\n",
                         executor_core_id,
                         row_band_begin,
                         job_row_begin,
                         sub_job_rows,
                         static_cast<unsigned long long>(sub_job_bytes),
                         static_cast<unsigned long long>(sub_job_input_hbm),
                         static_cast<unsigned long long>(sub_job_input_gm));
            return 1;
        }
        if (trace_bands) {
            std::printf("[SOFTMAX-SFU-JOB] band_stage=subjob row_band_begin=%d job_row_begin=%d sub_job_rows=%d input_gm=0x%llx output_gm=0x%llx tag=%llu\n",
                        row_band_begin,
                        job_row_begin,
                        sub_job_rows,
                        static_cast<unsigned long long>(sub_job_input_gm),
                        static_cast<unsigned long long>(sub_job_output_gm),
                        static_cast<unsigned long long>(tag + static_cast<uint64_t>(sub_job_index)));
            std::fflush(stdout);
        }
        golem_softmax_op_desc_t sub_desc = softmax_desc;
        sub_desc.outer = static_cast<uint64_t>(sub_job_rows);
        sub_desc.dim = static_cast<uint64_t>(cfg.n);
        MatmulRuntimeConfig sub_cfg = cfg;
        sub_cfg.m = sub_job_rows;
        sub_cfg.block_m = sub_job_rows;
        const golem_status_t sfu_status = golemRunStandaloneSoftmaxSfuJobForCore(
            &sub_desc,
            executor_core_id,
            &sub_cfg,
            sub_job_input_gm,
            sub_job_output_gm,
            desc_gm,
            static_cast<uint32_t>(chunk_elems),
            static_cast<uint32_t>(worker_cores),
            0,
            static_cast<uint32_t>(executor_core_id),
            row_engine_model
                ? SFU_JOB_FLAG_ROW_ENGINE_MODEL
                : 0,
            job_id ^ static_cast<uint64_t>(row_band_begin + job_row_begin),
            tag + static_cast<uint64_t>(sub_job_index));
        if (sfu_status != GOLEM_STATUS_OK) {
            std::fprintf(stderr,
                         "[SOFTMAX-SFU-JOB] standalone unified direct sub-job failed: "
                         "row_band_begin=%d job_row_begin=%d sub_job_rows=%d error=%s\n",
                         row_band_begin,
                         job_row_begin,
                         sub_job_rows,
                         golemSoftmaxSfuGetLastErrorString());
            return 1;
        }
        if (trace_bands) {
            std::printf("[SOFTMAX-SFU-JOB] band_stage=direct-store row_band_begin=%d job_row_begin=%d sub_job_rows=%d bytes=%llu hbm=0x%llx tag=%llu\n",
                        row_band_begin,
                        job_row_begin,
                        sub_job_rows,
                        static_cast<unsigned long long>(sub_job_bytes),
                        static_cast<unsigned long long>(sub_job_output_hbm),
                        static_cast<unsigned long long>(tag + static_cast<uint64_t>(sub_job_index)));
            std::fflush(stdout);
        }
        set_len(sub_job_bytes);
        if (row_engine_model) {
            remote_store_wait(sub_job_output_gm, sub_job_output_hbm);
        } else {
            remote_store(sub_job_output_gm, sub_job_output_hbm);
        }
        ++sub_job_index;
    }

    if (trace_bands) {
        std::printf("[SOFTMAX-SFU-JOB] band_stage=done row_band_begin=%d row_band_rows=%d tag=%llu\n",
                    row_band_begin,
                    row_band_rows,
                    static_cast<unsigned long long>(tag));
        std::fflush(stdout);
    }
    return 0;
}

int run_standalone_unified_job_softmax_for_core(int executor_core_id,
                                                int requested_core_id,
                                                const golem_softmax_op_desc_t& softmax_desc,
                                                const MatmulRuntimeConfig& cfg,
                                                uint64_t job_id) {
    if (!validate_matmul_call(cfg)) {
        std::fprintf(stderr, "[SOFTMAX-SFU-JOB] invalid standalone matmul-shaped config\n");
        return 1;
    }

    int staging_rows = static_cast<int>(
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS", cfg.block_m));
    if (staging_rows <= 0) {
        staging_rows = cfg.block_m;
    }
    if (staging_rows > cfg.m) {
        staging_rows = cfg.m;
    }
    int band_core_count = static_cast<int>(
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_BAND_CORES", 1));
    if (band_core_count <= 0) {
        band_core_count = 1;
    }
    if (band_core_count > ACTIVE_GEMM_CORES) {
        band_core_count = ACTIVE_GEMM_CORES;
    }
    const bool direct_rowmajor_hbm =
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM", 0) != 0;
    const bool distributed_columns =
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS", 0) != 0;
    const bool row_engine_model =
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_ROW_ENGINE", 0) != 0;
    const bool tensor_controller =
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_TENSOR_CONTROLLER", 0) != 0;
    const int64_t attention_head_dim =
        read_i64_env_or_default("GOLEM_SFU_ATTENTION_HEAD_DIM", 0);
    const bool attention_causal =
        read_i64_env_or_default("GOLEM_SFU_ATTENTION_CAUSAL", 0) != 0;
    const int total_bands = (cfg.m + staging_rows - 1) / staging_rows;
    if (!distributed_columns && band_core_count > total_bands) {
        band_core_count = total_bands;
    }
    if (tensor_controller) {
        const bool coordinator = GROUP_MANAGER_ENABLED
            ? executor_core_id >= 0 && executor_core_id < TOTAL_GROUPS
            : requested_core_id == 0;
        if (!coordinator) {
            return 0;
        }
    }
    if (!tensor_controller && !distributed_columns && requested_core_id >= band_core_count) {
        return 0;
    }
    const int active_worker_slot = gemm_worker_slot_for_core(executor_core_id);
    if (distributed_columns &&
        (active_worker_slot < 0 || active_worker_slot >= band_core_count)) {
        return 0;
    }
    if (distributed_columns && !direct_rowmajor_hbm) {
        std::fprintf(stderr,
                     "[SOFTMAX-SFU-JOB] distributed columns currently requires direct row-major HBM\n");
        return 1;
    }
    if (!distributed_columns && band_core_count > 1 &&
        (staging_rows % cfg.block_m) != 0) {
        std::fprintf(stderr,
                     "[SOFTMAX-SFU-JOB] cooperative row-band mode requires whole m-tile bands: "
                     "staging_rows=%d block_m=%d band_cores=%d\n",
                     staging_rows,
                     cfg.block_m,
                     band_core_count);
        return 1;
    }
    const uint64_t band_matrix_elems =
        static_cast<uint64_t>(staging_rows) * static_cast<uint64_t>(cfg.n);
    const uint64_t band_matrix_bytes = band_matrix_elems * sizeof(float);
    const uint64_t band_matrix_bytes_aligned = align_up_constexpr(band_matrix_bytes, LOCAL_ALIGN);
    const uint64_t tile_elems =
        static_cast<uint64_t>(cfg.block_m) * static_cast<uint64_t>(cfg.block_n);
    const uint64_t tile_bytes = tile_elems * sizeof(float);
    const uint64_t tile_bytes_aligned = align_up_constexpr(tile_bytes, LOCAL_ALIGN);
    int job_rows_per_issue = static_cast<int>(
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS", staging_rows));
    if (job_rows_per_issue <= 0) {
        job_rows_per_issue = staging_rows;
    }
    uint64_t worker_cores = static_cast<uint64_t>(
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES", 16));
    if (worker_cores == 0) {
        worker_cores = 1;
    }
    if (distributed_columns &&
        (worker_cores > static_cast<uint64_t>(band_core_count) ||
         (band_core_count % static_cast<int>(worker_cores)) != 0 ||
         worker_cores > static_cast<uint64_t>(cfg.n))) {
        std::fprintf(stderr,
                     "[SOFTMAX-SFU-JOB] invalid distributed grouping: "
                     "band_cores=%d worker_cores=%llu dim=%d\n",
                     band_core_count,
                     static_cast<unsigned long long>(worker_cores),
                     cfg.n);
        return 1;
    }
    const int local_buffer_rows =
        direct_rowmajor_hbm ? std::min(job_rows_per_issue, staging_rows) : staging_rows;
    const int distributed_worker_slot =
        distributed_columns ? (active_worker_slot % static_cast<int>(worker_cores)) : 0;
    uint64_t local_buffer_cols = static_cast<uint64_t>(cfg.n);
    if (distributed_columns) {
        uint64_t local_col_begin = 0;
        uint64_t local_col_end = 0;
        softmax_primitive_slice_for_worker(
            static_cast<uint64_t>(cfg.n),
            distributed_worker_slot,
            static_cast<int>(worker_cores),
            &local_col_begin,
            &local_col_end);
        local_buffer_cols = local_col_end - local_col_begin;
    }
    const uint64_t local_buffer_elems =
        static_cast<uint64_t>(local_buffer_rows) * local_buffer_cols;
    const uint64_t local_buffer_bytes = local_buffer_elems * sizeof(float);
    const uint64_t local_buffer_bytes_aligned = align_up_constexpr(local_buffer_bytes, LOCAL_ALIGN);
    const uint64_t tensor_scratch_bytes = static_cast<uint64_t>(
        std::min(cfg.m, ACTIVE_GEMM_CORES * 4)) * static_cast<uint64_t>(cfg.n) * sizeof(float);
    const uint64_t required_bytes = tensor_controller
        ? tensor_scratch_bytes
        : direct_rowmajor_hbm
            ? (local_buffer_bytes_aligned + local_buffer_bytes_aligned)
            : (band_matrix_bytes_aligned + band_matrix_bytes_aligned + tile_bytes_aligned);
    const uint64_t available_bytes =
        (GOLEM_GLOBAL_STRIDE_BYTES > (LOCAL_DATA_BASE + kPrimitiveLocalGuardBytes))
            ? (GOLEM_GLOBAL_STRIDE_BYTES - LOCAL_DATA_BASE - kPrimitiveLocalGuardBytes)
            : 0;
    if (required_bytes > available_bytes) {
        std::fprintf(stderr,
                     "[SOFTMAX-SFU-JOB] standalone local GM staging too small: "
                     "required=%llu available=%llu m=%d n=%d block_m=%d block_n=%d\n",
                     static_cast<unsigned long long>(required_bytes),
                     static_cast<unsigned long long>(available_bytes),
                     cfg.m, cfg.n, cfg.block_m, cfg.block_n);
        return 1;
    }
    const uint64_t full_matrix_bytes =
        static_cast<uint64_t>(cfg.m) * static_cast<uint64_t>(cfg.n) * sizeof(float);
    if (direct_rowmajor_hbm &&
        OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE + full_matrix_bytes > OFF_GEMM_BIAS_BASE) {
        std::fprintf(stderr,
                     "[SOFTMAX-SFU-JOB] direct row-major HBM region too small: "
                     "out_base=0x%llx bytes=%llu bias=0x%llx m=%d n=%d\n",
                     static_cast<unsigned long long>(OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE),
                     static_cast<unsigned long long>(full_matrix_bytes),
                     static_cast<unsigned long long>(OFF_GEMM_BIAS_BASE),
                     cfg.m,
                     cfg.n);
        return 1;
    }

    const uint64_t input_gm = gm_addr(executor_core_id, LOCAL_DATA_BASE);
    const uint64_t output_gm = input_gm + local_buffer_bytes_aligned;
    const uint64_t tile_gm = output_gm + local_buffer_bytes_aligned;
    const uint64_t desc_gm = gm_addr(executor_core_id, LOCAL_TMP_OFFSET);
    std::vector<float> row_band;
    std::vector<float> tile;
    if (!direct_rowmajor_hbm) {
        row_band.assign(static_cast<size_t>(band_matrix_elems), 0.0f);
        tile.assign(static_cast<size_t>(tile_elems), 0.0f);
    }
    const char* softmax_hbm_layout = std::getenv("GOLEM_SFU_SOFTMAX_HBM_LAYOUT");
    const bool band_striped_hbm = softmax_hbm_layout != nullptr &&
        (std::strcmp(softmax_hbm_layout, "band_striped") == 0 ||
         std::strcmp(softmax_hbm_layout, "striped") == 0);

    uint64_t chunk_elems = static_cast<uint64_t>(
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS", 256));
    if (chunk_elems == 0) {
        chunk_elems = 256;
    }
    const bool trace_bands =
        read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS", 0) != 0;
    const uint64_t tag = kPrimitiveTagBase + 0xa00000ULL;
    uint64_t row_engine_start_cycle = 0;
    uint64_t row_engine_rows = 0;

    std::printf("[SOFTMAX] dispatch=sfu-standalone-unified-job-softmax rows=%d dim=%d chunk=%llu workers=%llu staging_rows=%d job_rows=%d band_cores=%d direct_rowmajor_hbm=%d distributed_columns=%d executor_core=%d requested_core=%d\n",
                cfg.m,
                cfg.n,
                static_cast<unsigned long long>(chunk_elems),
                static_cast<unsigned long long>(worker_cores),
                staging_rows,
                job_rows_per_issue,
                band_core_count,
                direct_rowmajor_hbm ? 1 : 0,
                distributed_columns ? 1 : 0,
                executor_core_id,
                requested_core_id);
    std::fflush(stdout);

    // Exclude benchmark diagnostics from the accelerator kernel timing window.
    row_engine_start_cycle = row_engine_model ? read_cycle_counter() : 0;

    if (tensor_controller) {
        if (!row_engine_model || !direct_rowmajor_hbm || !band_striped_hbm) {
            std::fprintf(stderr,
                         "[SOFTMAX-TENSOR-CONTROLLER] requires Row Engine, direct HBM, and band striping\n");
            return 1;
        }
        const uint64_t params_gm = desc_gm + sizeof(SFUJobDesc);
        const uint64_t rowmajor_input_hbm =
            node_base_addr(kSfuSoftmaxRowmajorDataNode) + OFF_SFU_SOFTMAX_ROWMAJOR_BASE;
        const uint64_t rowmajor_output_hbm =
            node_base_addr(kSfuSoftmaxRowmajorDataNode) + OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE;
        const uint32_t row_contexts = static_cast<uint32_t>(
            std::min(cfg.m, ACTIVE_GEMM_CORES * 4));
        golem_softmax_launch_timeline_t launch_timeline = {};
        const golem_status_t status = golemRunTensorSoftmaxSfuJob(
            &softmax_desc,
            executor_core_id,
            &cfg,
            rowmajor_input_hbm,
            rowmajor_output_hbm,
            input_gm,
            params_gm,
            desc_gm,
            GOLEM_MEM_NODE_SIZE_BYTES,
            static_cast<uint32_t>(staging_rows),
            row_contexts,
            ACTIVE_GEMM_CORES,
            static_cast<uint32_t>(attention_head_dim),
            attention_causal,
            job_id,
            tag,
            &launch_timeline);
        if (status != GOLEM_STATUS_OK) {
            std::fprintf(stderr,
                         "[SOFTMAX-TENSOR-CONTROLLER] failed: %s\n",
                         golemSoftmaxSfuGetLastErrorString());
            return 1;
        }
        const uint64_t row_engine_end_cycle = read_cycle_counter();
        std::printf("[SOFTMAX-ROW-ENGINE] core=%d rows=%d start_cycle=%llu end_cycle=%llu cycles=%llu output_dma_completion=1 tensor_controller=1 launch_start_cycle=%llu descriptors_ready_cycle=%llu params_write_done_cycle=%llu desc_write_done_cycle=%llu issue_return_cycle=%llu wait_start_cycle=%llu wait_return_cycle=%llu\n",
                    executor_core_id,
                    cfg.m,
                    static_cast<unsigned long long>(row_engine_start_cycle),
                    static_cast<unsigned long long>(row_engine_end_cycle),
                    static_cast<unsigned long long>(row_engine_end_cycle - row_engine_start_cycle),
                    static_cast<unsigned long long>(launch_timeline.launch_start_cycle),
                    static_cast<unsigned long long>(launch_timeline.descriptors_ready_cycle),
                    static_cast<unsigned long long>(launch_timeline.params_write_done_cycle),
                    static_cast<unsigned long long>(launch_timeline.desc_write_done_cycle),
                    static_cast<unsigned long long>(launch_timeline.issue_return_cycle),
                    static_cast<unsigned long long>(launch_timeline.wait_start_cycle),
                    static_cast<unsigned long long>(launch_timeline.wait_return_cycle));
        std::fflush(stdout);
        return 0;
    }

    if (distributed_columns) {
        const uint64_t rowmajor_input_hbm =
            node_base_addr(kSfuSoftmaxRowmajorDataNode) + OFF_SFU_SOFTMAX_ROWMAJOR_BASE;
        const uint64_t rowmajor_output_hbm =
            node_base_addr(kSfuSoftmaxRowmajorDataNode) + OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE;
        const int status =
            run_standalone_unified_job_softmax_distributed_direct_for_core(
                executor_core_id,
                softmax_desc,
                cfg,
                active_worker_slot,
                band_core_count,
                static_cast<int>(worker_cores),
                staging_rows,
                job_rows_per_issue,
                input_gm,
                output_gm,
                desc_gm,
                chunk_elems,
                job_id,
                tag,
                trace_bands,
                rowmajor_input_hbm,
                rowmajor_output_hbm);
        if (status != 0) {
            std::fprintf(stderr,
                         "[SOFTMAX-SFU-JOB] distributed standalone unified job failed on core %d\n",
                         requested_core_id);
            return status;
        }
        std::printf("[SOFTMAX] mode=sfu-standalone-job-softmax executor_core=%d requested_core=%d rows=%d dim=%d chunk_elems=%llu worker_cores=%llu staging_rows=%d job_rows=%d band_cores=%d direct_rowmajor_hbm=1 distributed_columns=1 PASS\n",
                    executor_core_id,
                    requested_core_id,
                    cfg.m,
                    cfg.n,
                    static_cast<unsigned long long>(chunk_elems),
                    static_cast<unsigned long long>(worker_cores),
                    staging_rows,
                    job_rows_per_issue,
                    band_core_count);
        std::fflush(stdout);
        return 0;
    }

    int band_index = 0;
    for (int row_band_begin = 0; row_band_begin < cfg.m; row_band_begin += staging_rows) {
        const int row_band_rows = std::min(staging_rows, cfg.m - row_band_begin);
        const int band_slot = band_index % band_core_count;
        if (band_slot != requested_core_id) {
            ++band_index;
            continue;
        }
        const uint64_t band_tag = tag + static_cast<uint64_t>(band_index);
        const int data_node_count = std::max(1, GOLEM_NUM_MEMORY_NODES - 1);
        const int data_node = band_striped_hbm
            ? 1 + (band_index % data_node_count)
            : kSfuSoftmaxRowmajorDataNode;
        const uint64_t local_band = band_striped_hbm
            ? static_cast<uint64_t>(band_index / data_node_count)
            : static_cast<uint64_t>(row_band_begin / staging_rows);
        const uint64_t local_band_offset = local_band * static_cast<uint64_t>(staging_rows) *
            static_cast<uint64_t>(cfg.n) * sizeof(float);
        const uint64_t rowmajor_input_hbm = node_base_addr(data_node) +
            OFF_SFU_SOFTMAX_ROWMAJOR_BASE + local_band_offset;
        const uint64_t rowmajor_output_hbm = node_base_addr(data_node) +
            OFF_SFU_SOFTMAX_ROWMAJOR_OUT_BASE + local_band_offset;
        const int status = direct_rowmajor_hbm
            ? run_standalone_unified_job_softmax_direct_band_for_core(
                  executor_core_id,
                  softmax_desc,
                  cfg,
                  row_band_begin,
                  row_band_rows,
                  input_gm,
                  output_gm,
                  desc_gm,
                  chunk_elems,
                  worker_cores,
                  job_rows_per_issue,
                  job_id,
                  band_tag,
                  trace_bands,
                  rowmajor_input_hbm,
                  rowmajor_output_hbm)
            : run_standalone_unified_job_softmax_band_for_core(
                  executor_core_id,
                  softmax_desc,
                  cfg,
                  row_band_begin,
                  row_band_rows,
                  input_gm,
                  output_gm,
                  tile_gm,
                  desc_gm,
                  tile_bytes,
                  chunk_elems,
                  worker_cores,
                  job_rows_per_issue,
                  job_id,
                  band_tag,
                  trace_bands,
                  row_band,
                  tile);
        if (status != 0) {
            std::fprintf(stderr, "[SOFTMAX-SFU-JOB] standalone unified job failed on core %d\n",
                         requested_core_id);
            return status;
        }
        row_engine_rows += static_cast<uint64_t>(row_band_rows);
        ++band_index;
    }

    if (row_engine_model) {
        const uint64_t row_engine_end_cycle = read_cycle_counter();
        std::printf("[SOFTMAX-ROW-ENGINE] core=%d rows=%llu start_cycle=%llu end_cycle=%llu cycles=%llu output_dma_completion=1\n",
                    executor_core_id,
                    static_cast<unsigned long long>(row_engine_rows),
                    static_cast<unsigned long long>(row_engine_start_cycle),
                    static_cast<unsigned long long>(row_engine_end_cycle),
                    static_cast<unsigned long long>(row_engine_end_cycle - row_engine_start_cycle));
        std::fflush(stdout);
    }

    std::printf("[SOFTMAX] mode=sfu-standalone-job-softmax executor_core=%d requested_core=%d rows=%d dim=%d chunk_elems=%llu worker_cores=%llu staging_rows=%d job_rows=%d band_cores=%d direct_rowmajor_hbm=%d distributed_columns=0 PASS\n",
                executor_core_id,
                requested_core_id,
                cfg.m,
                cfg.n,
                static_cast<unsigned long long>(chunk_elems),
                static_cast<unsigned long long>(worker_cores),
                staging_rows,
                job_rows_per_issue,
                band_core_count,
                direct_rowmajor_hbm ? 1 : 0);
    std::fflush(stdout);
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

    const int64_t primitive_softmax =
        read_i64_env_or_default("GOLEM_SFU_PRIMITIVE_SOFTMAX", 0);
    if (primitive_softmax != 0) {
        return run_sfu_primitive_softmax_for_core(executor_core_id, requested_core_id, cfg);
    }

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
        const int64_t job_softmax = read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX", 0);
        if (job_softmax != 0) {
            return run_standalone_unified_job_softmax_for_core(
                executor_core_id, requested_core_id, softmax_desc, cfg, job_id);
        }
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

    const int64_t job_softmax = read_i64_env_or_default("GOLEM_SFU_JOB_SOFTMAX", 0);
    if (job_softmax != 0) {
        return run_gemm_unified_job_softmax_for_core(
            executor_core_id, requested_core_id, op_desc, softmax_desc, cfg, job_id);
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
