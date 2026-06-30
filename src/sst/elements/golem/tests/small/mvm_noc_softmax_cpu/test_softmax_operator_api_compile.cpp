#include "../golem_operator_api.h"
#include "golem_softmax_runtime.h"

static_assert(static_cast<unsigned>(GolemOperatorKind::SOFTMAX) == 3,
              "softmax must be registered after existing operator kinds");
static_assert(static_cast<unsigned>(TensorDataType::FP32) != static_cast<unsigned>(GOLEM_DTYPE_FP32),
              "shared API dtype values must be converted before calling runtime C ABI");

int main() {
    SoftmaxOpDesc api_desc = {
        .version = 1,
        .outer = 64,
        .dim = 64,
        .axis = -1,
        .allow_golem = false,
        .dtype = TensorDataType::FP32,
        .layout = TensorLayout::ROW_MAJOR,
    };

    golem_softmax_op_desc_t runtime_desc = golemMakeSoftmaxRuntimeDesc(api_desc);
    return runtime_desc.outer == 64 &&
                   runtime_desc.dim == 64 &&
                   runtime_desc.axis == -1 &&
                   runtime_desc.dtype == GOLEM_DTYPE_FP32 &&
                   runtime_desc.layout == GOLEM_LAYOUT_ROW_MAJOR
               ? 0
               : 1;
}
