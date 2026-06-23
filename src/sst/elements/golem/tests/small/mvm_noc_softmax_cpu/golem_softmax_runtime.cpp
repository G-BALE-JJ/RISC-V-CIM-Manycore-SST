#include "golem_softmax_runtime.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(__riscv)
#include "operators.h"
#endif

namespace {

constexpr int64_t SOFTMAX_RISCV_MAX_DIM = 64;

thread_local std::string g_softmax_last_error;

void set_softmax_last_error(const char* fmt, ...) {
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    g_softmax_last_error = buffer;
}

bool is_supported_axis(int64_t axis) {
    return axis == -1 || axis == 1;
}

bool is_row_major_contiguous_2d(const golem_tensor_desc_t* tensor) {
    return tensor->stride[1] == 1 && tensor->stride[0] == tensor->shape[1];
}

golem_status_t validate_softmax_op(const golem_softmax_op_desc_t* op) {
    if (op == nullptr) {
        set_softmax_last_error("softmax op_desc is null");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (op->outer <= 0 || op->dim <= 0) {
        set_softmax_last_error("softmax outer/dim must be positive, got (%lld,%lld)",
                               static_cast<long long>(op->outer),
                               static_cast<long long>(op->dim));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (!is_supported_axis(op->axis)) {
        set_softmax_last_error("softmax v1 only supports axis=-1 or axis=1, got %lld",
                               static_cast<long long>(op->axis));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (op->dtype != GOLEM_DTYPE_FP32) {
        set_softmax_last_error("softmax v1 only supports GOLEM_DTYPE_FP32");
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (op->layout != GOLEM_LAYOUT_ROW_MAJOR) {
        set_softmax_last_error("softmax v1 only supports GOLEM_LAYOUT_ROW_MAJOR");
        return GOLEM_STATUS_UNSUPPORTED;
    }
    return GOLEM_STATUS_OK;
}

golem_status_t validate_softmax_tensor(const golem_tensor_desc_t* tensor,
                                       const golem_softmax_op_desc_t* op,
                                       const char* name) {
    if (tensor == nullptr) {
        set_softmax_last_error("%s tensor descriptor is null", name);
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (tensor->data == nullptr) {
        set_softmax_last_error("%s tensor data is null", name);
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (tensor->ndim != 2) {
        set_softmax_last_error("%s tensor ndim must be 2, got %lld",
                               name,
                               static_cast<long long>(tensor->ndim));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (tensor->shape[0] != op->outer || tensor->shape[1] != op->dim) {
        set_softmax_last_error("%s tensor shape mismatch, expected [%lld,%lld], got [%lld,%lld]",
                               name,
                               static_cast<long long>(op->outer),
                               static_cast<long long>(op->dim),
                               static_cast<long long>(tensor->shape[0]),
                               static_cast<long long>(tensor->shape[1]));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (tensor->dtype != GOLEM_DTYPE_FP32) {
        set_softmax_last_error("%s tensor dtype must be GOLEM_DTYPE_FP32", name);
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (tensor->layout != GOLEM_LAYOUT_ROW_MAJOR) {
        set_softmax_last_error("%s tensor layout must be GOLEM_LAYOUT_ROW_MAJOR", name);
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (!is_row_major_contiguous_2d(tensor)) {
        set_softmax_last_error("%s tensor must be row-major contiguous, got stride=(%lld,%lld)",
                               name,
                               static_cast<long long>(tensor->stride[0]),
                               static_cast<long long>(tensor->stride[1]));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    return GOLEM_STATUS_OK;
}

void run_softmax_fp32_rows(const golem_softmax_op_desc_t* op,
                           const golem_tensor_desc_t* input,
                           const golem_tensor_desc_t* output) {
    const float* x = reinterpret_cast<const float*>(input->data);
    float* y = reinterpret_cast<float*>(output->data);

    for (int64_t row = 0; row < op->outer; ++row) {
        const float* x_row = x + row * input->stride[0];
        float* y_row = y + row * output->stride[0];

        float max_value = x_row[0];
        for (int64_t col = 1; col < op->dim; ++col) {
            if (x_row[col] > max_value) {
                max_value = x_row[col];
            }
        }

        float sum = 0.0f;
        for (int64_t col = 0; col < op->dim; ++col) {
            const float value = std::exp(x_row[col] - max_value);
            y_row[col] = value;
            sum += value;
        }

        const float inv_sum = 1.0f / sum;
        for (int64_t col = 0; col < op->dim; ++col) {
            y_row[col] *= inv_sum;
        }
    }
}

void run_softmax_fp32_buffer(const golem_softmax_op_desc_t* op,
                             const float* input,
                             float* output,
                             int64_t input_stride,
                             int64_t output_stride) {
    for (int64_t row = 0; row < op->outer; ++row) {
        const float* x_row = input + row * input_stride;
        float* y_row = output + row * output_stride;

        float max_value = x_row[0];
        for (int64_t col = 1; col < op->dim; ++col) {
            if (x_row[col] > max_value) {
                max_value = x_row[col];
            }
        }

        float sum = 0.0f;
        for (int64_t col = 0; col < op->dim; ++col) {
            const float value = std::exp(x_row[col] - max_value);
            y_row[col] = value;
            sum += value;
        }

        const float inv_sum = 1.0f / sum;
        for (int64_t col = 0; col < op->dim; ++col) {
            y_row[col] *= inv_sum;
        }
    }
}

golem_status_t validate_softmax_gm_request(const golem_softmax_op_desc_t* op,
                                           int core_id,
                                           uint64_t input_gm_addr,
                                           uint64_t output_gm_addr,
                                           int64_t input_stride,
                                           int64_t output_stride) {
    golem_status_t status = validate_softmax_op(op);
    if (status != GOLEM_STATUS_OK) {
        return status;
    }
    if (core_id < 0) {
        set_softmax_last_error("softmax GM core_id must be non-negative, got %d", core_id);
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (input_gm_addr == 0 || output_gm_addr == 0) {
        set_softmax_last_error("softmax GM input/output address must be nonzero");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (input_stride < op->dim || output_stride < op->dim) {
        set_softmax_last_error("softmax GM stride must be >= dim, got input_stride=%lld output_stride=%lld dim=%lld",
                               static_cast<long long>(input_stride),
                               static_cast<long long>(output_stride),
                               static_cast<long long>(op->dim));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    return GOLEM_STATUS_OK;
}

}  // namespace

extern "C" golem_status_t golemRunSoftmaxCpu(
    const golem_softmax_op_desc_t* op_desc,
    const golem_tensor_desc_t* input,
    const golem_tensor_desc_t* output) {
    golem_status_t status = validate_softmax_op(op_desc);
    if (status != GOLEM_STATUS_OK) {
        return status;
    }
    status = validate_softmax_tensor(input, op_desc, "input");
    if (status != GOLEM_STATUS_OK) {
        return status;
    }
    status = validate_softmax_tensor(output, op_desc, "output");
    if (status != GOLEM_STATUS_OK) {
        return status;
    }

    run_softmax_fp32_rows(op_desc, input, output);
    return GOLEM_STATUS_OK;
}

extern "C" golem_status_t golemRunSoftmaxCpuGm(
    const golem_softmax_op_desc_t* op_desc,
    uint64_t input_gm_addr,
    uint64_t output_gm_addr,
    int64_t input_stride,
    int64_t output_stride) {
    return golemRunSoftmaxCpuGmForCore(
        op_desc, 0, input_gm_addr, output_gm_addr, input_stride, output_stride);
}

extern "C" golem_status_t golemRunSoftmaxCpuGmForCore(
    const golem_softmax_op_desc_t* op_desc,
    int core_id,
    uint64_t input_gm_addr,
    uint64_t output_gm_addr,
    int64_t input_stride,
    int64_t output_stride) {
    golem_status_t status = validate_softmax_gm_request(
        op_desc, core_id, input_gm_addr, output_gm_addr, input_stride, output_stride);
    if (status != GOLEM_STATUS_OK) {
        return status;
    }

#if defined(__riscv)
    if (op_desc->dim > SOFTMAX_RISCV_MAX_DIM) {
        set_softmax_last_error("RISC-V softmax v1 supports dim <= %lld, got %lld",
                               static_cast<long long>(SOFTMAX_RISCV_MAX_DIM),
                               static_cast<long long>(op_desc->dim));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (input_stride != op_desc->dim || output_stride != op_desc->dim) {
        set_softmax_last_error("RISC-V softmax v1 requires contiguous rows, got input_stride=%lld output_stride=%lld dim=%lld",
                               static_cast<long long>(input_stride),
                               static_cast<long long>(output_stride),
                               static_cast<long long>(op_desc->dim));
        return GOLEM_STATUS_UNSUPPORTED;
    }

    float row_in[SOFTMAX_RISCV_MAX_DIM];
    float row_out[SOFTMAX_RISCV_MAX_DIM];
    const uint64_t row_bytes = static_cast<uint64_t>(op_desc->dim * sizeof(float));
    const uint64_t local_tmp_gm = gm_addr(core_id, LOCAL_LAYOUT.accum + LOCAL_OUT_TILE_BYTES_ALIGNED);
    for (int64_t row = 0; row < op_desc->outer; ++row) {
        const uint64_t row_input_gm = input_gm_addr + static_cast<uint64_t>(row * input_stride * sizeof(float));
        const uint64_t row_output_gm = output_gm_addr + static_cast<uint64_t>(row * output_stride * sizeof(float));
        dma_remote_load_to_gm(core_id, row_input_gm, local_tmp_gm, row_bytes);
        set_len(row_bytes);
        gm2mm(row_in, local_tmp_gm);

        float max_value = row_in[0];
        for (int64_t col = 1; col < op_desc->dim; ++col) {
            if (row_in[col] > max_value) {
                max_value = row_in[col];
            }
        }
        float sum = 0.0f;
        for (int64_t col = 0; col < op_desc->dim; ++col) {
            const float value = std::exp(row_in[col] - max_value);
            row_out[col] = value;
            sum += value;
        }
        const float inv_sum = 1.0f / sum;
        for (int64_t col = 0; col < op_desc->dim; ++col) {
            row_out[col] *= inv_sum;
        }

        set_len(row_bytes);
        mm2gm(row_out, local_tmp_gm);
        remote_store(local_tmp_gm, row_output_gm);
    }
#else
    const size_t input_elems = static_cast<size_t>(op_desc->outer * input_stride);
    const size_t output_elems = static_cast<size_t>(op_desc->outer * output_stride);
    std::vector<float> input(input_elems, 0.0f);
    std::vector<float> output(output_elems, 0.0f);
    const uint64_t input_bytes = static_cast<uint64_t>(input_elems * sizeof(float));
    const uint64_t output_bytes = static_cast<uint64_t>(output_elems * sizeof(float));
    const float* input_ptr = reinterpret_cast<const float*>(input_gm_addr);
    float* output_ptr = reinterpret_cast<float*>(output_gm_addr);
    std::memcpy(input.data(), input_ptr, static_cast<size_t>(input_bytes));
    run_softmax_fp32_buffer(op_desc, input.data(), output.data(), input_stride, output_stride);
    std::memcpy(output_ptr, output.data(), static_cast<size_t>(output_bytes));
#endif

    return GOLEM_STATUS_OK;
}

extern "C" const char* golemSoftmaxGetLastErrorString(void) {
    return g_softmax_last_error.c_str();
}
