#pragma once

#include <cstdint>

#ifdef __cplusplus
#include "../golem_operator_api.h"
#endif

#include "golem_matmul_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t outer;
    int64_t dim;
    int64_t axis;
    golem_dtype_t dtype;
    golem_layout_t layout;
} golem_softmax_op_desc_t;

golem_status_t golemRunSoftmaxCpu(
    const golem_softmax_op_desc_t* op_desc,
    const golem_tensor_desc_t* input,
    const golem_tensor_desc_t* output);

golem_status_t golemRunSoftmaxCpuGm(
    const golem_softmax_op_desc_t* op_desc,
    uint64_t input_gm_addr,
    uint64_t output_gm_addr,
    int64_t input_stride,
    int64_t output_stride);

golem_status_t golemRunSoftmaxCpuGmForCore(
    const golem_softmax_op_desc_t* op_desc,
    int core_id,
    uint64_t input_gm_addr,
    uint64_t output_gm_addr,
    int64_t input_stride,
    int64_t output_stride);

const char* golemSoftmaxGetLastErrorString(void);

#ifdef __cplusplus
}

inline golem_dtype_t golemSoftmaxRuntimeDtypeFromApi(TensorDataType dtype) {
    switch (dtype) {
        case TensorDataType::FP32:
            return GOLEM_DTYPE_FP32;
        case TensorDataType::INT32:
            return GOLEM_DTYPE_INT32;
    }
    return GOLEM_DTYPE_INT32;
}

inline golem_layout_t golemSoftmaxRuntimeLayoutFromApi(TensorLayout layout) {
    switch (layout) {
        case TensorLayout::ROW_MAJOR:
            return GOLEM_LAYOUT_ROW_MAJOR;
    }
    return GOLEM_LAYOUT_ROW_MAJOR;
}

inline golem_softmax_op_desc_t golemMakeSoftmaxRuntimeDesc(const SoftmaxOpDesc& op_desc) {
    return {
        .outer = op_desc.outer,
        .dim = op_desc.dim,
        .axis = op_desc.axis,
        .dtype = golemSoftmaxRuntimeDtypeFromApi(op_desc.dtype),
        .layout = golemSoftmaxRuntimeLayoutFromApi(op_desc.layout),
    };
}
#endif
