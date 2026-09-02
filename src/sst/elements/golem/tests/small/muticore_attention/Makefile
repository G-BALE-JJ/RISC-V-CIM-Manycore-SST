.PHONY: all clean test scale scale-e2 scale-e3 scale-e4 scale-e5 base

BASE_TEST_DIR := ../mvm_noc_int_array
ARCH ?= riscv64

RISCV_MUSL_TOOLCHAIN_BIN ?= /local/scratch/src/riscv64-linux-musl-cross/bin
CXX := $(RISCV_MUSL_TOOLCHAIN_BIN)/riscv64-linux-musl-g++
SCALE_E2_EXE := $(ARCH)/fused_attention_scale_e2
SCALE_E3_EXE := $(ARCH)/fused_attention_scale_e3
SCALE_E4_EXE := $(ARCH)/fused_attention_scale_e4
SCALE_E5_EXE := $(ARCH)/fused_attention_scale_e5

all: scale

base:
	$(MAKE) -C $(BASE_TEST_DIR) ARCH=$(ARCH)

scale: scale-e4

scale-e2: $(SCALE_E2_EXE)

scale-e3: $(SCALE_E3_EXE)

scale-e4: $(SCALE_E4_EXE)

scale-e5: $(SCALE_E5_EXE)

$(SCALE_E2_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 -DGOLEM_ATTENTION_QUERIES=64 \
		-DGOLEM_ATTENTION_KEYS=256 -static -no-pie -O2 -o $@ golem_attention_runtime.cpp

$(SCALE_E3_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 -DGOLEM_ATTENTION_QUERIES=256 \
		-DGOLEM_ATTENTION_KEYS=1024 -DGOLEM_ATTENTION_HEAD_DIM=128 \
		-static -no-pie -O2 -o $@ golem_attention_runtime.cpp

$(SCALE_E4_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 -DGOLEM_ATTENTION_QUERIES=512 \
		-DGOLEM_ATTENTION_KEYS=2048 -DGOLEM_ATTENTION_HEAD_DIM=128 \
		-static -no-pie -O2 -o $@ golem_attention_runtime.cpp

$(SCALE_E5_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 -DGOLEM_ATTENTION_QUERIES=1024 \
		-DGOLEM_ATTENTION_KEYS=4096 -DGOLEM_ATTENTION_HEAD_DIM=128 \
		-static -no-pie -O2 -o $@ golem_attention_runtime.cpp

clean:
	rm -f $(SCALE_E2_EXE) $(SCALE_E3_EXE) $(SCALE_E4_EXE) $(SCALE_E5_EXE)

test:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_*.py'
