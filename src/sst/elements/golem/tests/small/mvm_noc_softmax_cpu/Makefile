ARCH := riscv64
BASE_DIR := ../mvm_noc_int_array
RISCV_MUSL_TOOLCHAIN_BIN := /data/lzq/packages/install/riscv64_musl_toolchain/bin
CXX := $(RISCV_MUSL_TOOLCHAIN_BIN)/riscv64-linux-musl-g++

ifneq ($(filter-out riscv64,$(ARCH)),)
$(error mvm_noc_softmax_cpu only supports ARCH=riscv64; native builds are intentionally disabled)
endif

CXXFLAGS ?= -O2 -std=c++17
CPPFLAGS += -I$(BASE_DIR) $(CFLAGS)
LDFLAGS_STATIC ?= -static
LDLIBS += -lm

PROG := test_noc_dma_softmax
OUT_DIR := $(ARCH)
SRCS := test_noc_dma_softmax.cpp golem_softmax_runtime.cpp $(BASE_DIR)/golem_matmul_runtime.cpp
API_TEST_OBJ := $(OUT_DIR)/test_softmax_operator_api_compile.o

all: $(OUT_DIR)/$(PROG) $(API_TEST_OBJ)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/$(PROG): test_noc_dma_softmax.cpp golem_softmax_runtime.cpp golem_softmax_runtime.h ../golem_operator_api.h | $(OUT_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS_STATIC) -o $@ $(SRCS) $(LDLIBS)

$(API_TEST_OBJ): test_softmax_operator_api_compile.cpp golem_softmax_runtime.h ../golem_operator_api.h | $(OUT_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ test_softmax_operator_api_compile.cpp

clean:
	rm -rf $(OUT_DIR)/$(PROG) $(API_TEST_OBJ)
