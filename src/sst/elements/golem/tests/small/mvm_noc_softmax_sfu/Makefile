ARCH := riscv64
BASE_DIR := ../mvm_noc_int_array
CPU_SOFTMAX_DIR := ../mvm_noc_softmax_cpu
RISCV_MUSL_TOOLCHAIN_BIN ?= /data/lzq/packages/install/riscv64_musl_toolchain/bin
CXX := $(RISCV_MUSL_TOOLCHAIN_BIN)/riscv64-linux-musl-g++

ifneq ($(filter-out riscv64,$(ARCH)),)
$(error mvm_noc_softmax_sfu only supports ARCH=riscv64; native builds are intentionally disabled)
endif

CXXFLAGS ?= -O2 -std=c++17
CPPFLAGS += -I. -I$(BASE_DIR) -I$(CPU_SOFTMAX_DIR) $(CFLAGS)
LDFLAGS_STATIC ?= -static
LDLIBS += -lm

PROG := test_noc_dma_softmax_sfu
OUT_DIR := $(ARCH)
SRCS := test_noc_dma_softmax_sfu.cpp golem_softmax_sfu_runtime.cpp $(BASE_DIR)/golem_matmul_runtime.cpp $(CPU_SOFTMAX_DIR)/golem_softmax_runtime.cpp

all: $(OUT_DIR)/$(PROG)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/$(PROG): test_noc_dma_softmax_sfu.cpp golem_softmax_sfu_runtime.cpp golem_softmax_sfu_runtime.h ex_instr.h $(BASE_DIR)/gemm_matmul_op.h | $(OUT_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS_STATIC) -o $@ $(SRCS) $(LDLIBS)

clean:
	rm -rf $(OUT_DIR)/$(PROG)
