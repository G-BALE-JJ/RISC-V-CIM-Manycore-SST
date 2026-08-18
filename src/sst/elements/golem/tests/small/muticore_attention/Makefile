.PHONY: all clean test fused online causal partial scale scale-e2 scale-e3 scale-e4 scale-e5 base

BASE_TEST_DIR := ../mvm_noc_int_array
ARCH ?= riscv64

RISCV_MUSL_TOOLCHAIN_BIN ?= /data/lzq/packages/install/riscv64_musl_toolchain/bin
CXX := $(RISCV_MUSL_TOOLCHAIN_BIN)/riscv64-linux-musl-g++
FUSED_EXE := $(ARCH)/fused_attention_c1
ONLINE_EXE := $(ARCH)/fused_attention_d1
CAUSAL_EXE := $(ARCH)/fused_attention_d2_causal
PARTIAL_EXE := $(ARCH)/fused_attention_d3_partial
SCALE_E2_EXE := $(ARCH)/fused_attention_scale_e2
SCALE_E3_EXE := $(ARCH)/fused_attention_scale_e3
SCALE_E4_EXE := $(ARCH)/fused_attention_scale_e4
SCALE_E5_EXE := $(ARCH)/fused_attention_scale_e5

all: fused online causal partial scale

base:
	$(MAKE) -C $(BASE_TEST_DIR) ARCH=$(ARCH)

fused: $(FUSED_EXE)

online: $(ONLINE_EXE)

causal: $(CAUSAL_EXE)

partial: $(PARTIAL_EXE)

scale: scale-e4

scale-e2: $(SCALE_E2_EXE)

scale-e3: $(SCALE_E3_EXE)

scale-e4: $(SCALE_E4_EXE)

scale-e5: $(SCALE_E5_EXE)

$(FUSED_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -static -O2 -o $@ golem_attention_runtime.cpp

$(ONLINE_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_QUERIES=64 -DGOLEM_ATTENTION_KEYS=64 \
		-static -O2 -o $@ golem_attention_runtime.cpp

$(CAUSAL_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_QUERIES=64 -DGOLEM_ATTENTION_KEYS=64 \
		-DGOLEM_ATTENTION_CAUSAL=1 -static -O2 -o $@ golem_attention_runtime.cpp

$(PARTIAL_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_QUERIES=20 -DGOLEM_ATTENTION_KEYS=70 \
		-static -O2 -o $@ golem_attention_runtime.cpp

$(SCALE_E2_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 -DGOLEM_ATTENTION_QUERIES=64 \
		-DGOLEM_ATTENTION_KEYS=256 -static -O2 -o $@ golem_attention_runtime.cpp

$(SCALE_E3_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 -DGOLEM_ATTENTION_QUERIES=256 \
		-DGOLEM_ATTENTION_KEYS=1024 -DGOLEM_ATTENTION_HEAD_DIM=128 \
		-static -O2 -o $@ golem_attention_runtime.cpp

$(SCALE_E4_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 -DGOLEM_ATTENTION_QUERIES=512 \
		-DGOLEM_ATTENTION_KEYS=2048 -DGOLEM_ATTENTION_HEAD_DIM=128 \
		-static -O2 -o $@ golem_attention_runtime.cpp

$(SCALE_E5_EXE): golem_attention_runtime.cpp golem_attention_runtime.h
	mkdir -p $(ARCH)
	$(CXX) $(CFLAGS) -DGOLEM_ATTENTION_SCALE=1 -DGOLEM_ATTENTION_QUERIES=1024 \
		-DGOLEM_ATTENTION_KEYS=4096 -DGOLEM_ATTENTION_HEAD_DIM=128 \
		-static -O2 -o $@ golem_attention_runtime.cpp

clean:
	rm -f $(FUSED_EXE) $(ONLINE_EXE) $(CAUSAL_EXE) $(PARTIAL_EXE) \
		$(SCALE_E2_EXE) $(SCALE_E3_EXE) $(SCALE_E4_EXE) $(SCALE_E5_EXE)

test:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_*.py'
