#pragma once

#include "../mvm_noc_int_array/ex_instr.h"

static constexpr uint32_t GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE = 0x17;
static constexpr uint32_t GOLEM_ROCC_FUNC7_SFU_WAIT = 0x18;

static inline void sfu_softmax_tile(uint64_t desc_gm_addr, uint64_t tag) {
    asm volatile(
        ".insn r 0x0b, 7, %2, x0, %0, %1"
        :
        : "r"(desc_gm_addr), "r"(tag), "i"(GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE)
        : "memory");
}

static inline uint64_t sfu_wait(uint64_t tag) {
    uint64_t status;
    asm volatile(
        ".insn r 0x0b, 7, %2, %0, %1, x0"
        : "=r"(status)
        : "r"(tag), "i"(GOLEM_ROCC_FUNC7_SFU_WAIT)
        : "memory");
    return status;
}
