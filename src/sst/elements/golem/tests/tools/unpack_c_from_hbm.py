#!/usr/bin/env python3

import argparse
import csv
import os
import sys

if __package__ in {None, ""}:
    _tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

from golem_dtype import (
    cast_scalar,
    elem_nbytes,
    normalize_dtype,
    pack_values,
    unpack_values,
)


def align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unpack matmul C tensor from hbm_out_node*.bin"
    )
    parser.add_argument(
        "--out-file", required=True, help="Output C file (.bin int32 or .csv)"
    )
    args = parser.parse_args(argv)

    tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_root = os.getenv(
        "GOLEM_ARTIFACT_ROOT", os.path.join(tests_dir, "artifacts")
    )
    hbm_dir = os.getenv("GOLEM_HBM_DIR", os.path.join(artifact_root, "hbm"))

    array_input_size = int(os.getenv("GOLEM_ARRAY_INPUT_SIZE", "4"))
    array_output_size = int(os.getenv("GOLEM_ARRAY_OUTPUT_SIZE", "4"))
    num_arrays = int(os.getenv("GOLEM_NUM_ARRAYS", "1"))
    m = int(
        os.getenv("GOLEM_MATMUL_M", os.getenv("GOLEM_GEMM_M", str(array_output_size)))
    )
    n = int(os.getenv("GOLEM_MATMUL_N", os.getenv("GOLEM_GEMM_N", str(num_arrays))))
    k = int(
        os.getenv("GOLEM_MATMUL_K", os.getenv("GOLEM_GEMM_K", str(array_input_size)))
    )
    block_m = int(os.getenv("GOLEM_MATMUL_BLOCK_M", str(array_output_size)))
    block_n = int(os.getenv("GOLEM_MATMUL_BLOCK_N", str(num_arrays)))
    block_k = int(os.getenv("GOLEM_MATMUL_BLOCK_K", str(array_input_size)))
    dtype = normalize_dtype(os.getenv("GOLEM_MATMUL_DTYPE", "int32"))
    out_layout = os.getenv("GOLEM_GEMM_OUT_LAYOUT", "rowmajor").strip().lower()
    direct_rowmajor_softmax = (
        int(os.getenv("GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM", "0")) != 0
    )

    num_memory_nodes = int(os.getenv("GOLEM_NUM_MEMORY_NODES", "4"))
    total_groups = int(os.getenv("GOLEM_TOTAL_GROUPS", "4"))
    total_gemm_cores = int(
        os.getenv("GOLEM_TOTAL_GEMM_CORES", os.getenv("GOLEM_TOTAL_CORES", "16"))
    )
    group_manager_enabled = int(os.getenv("GOLEM_GROUP_MANAGER_ENABLE", "0")) != 0
    os_memory_node = 0
    data_nodes = [idx for idx in range(num_memory_nodes) if idx != os_memory_node]

    dedicated_manager_cores = total_groups if group_manager_enabled else 0
    active_gemm_cores = total_gemm_cores - dedicated_manager_cores
    first_worker_core = dedicated_manager_cores

    if m % block_m != 0 or n % block_n != 0 or k % block_k != 0:
        raise ValueError("M/N/K must be divisible by block_M/N/K")

    mm_align = 0x100
    elem_bytes = elem_nbytes(dtype)
    mat_bytes = block_m * block_k * elem_bytes
    vec_bytes = block_k * elem_bytes
    mm_mat_stride = align_up(mat_bytes, mm_align)
    mm_vec_stride = align_up(vec_bytes, mm_align)

    m_tiles = m // block_m
    n_tiles = n // block_n
    k_tiles = k // block_k
    total_tasks = m_tiles * n_tiles
    a_reuse_n_tiles = max(1, int(os.getenv("GOLEM_A_REUSE_N_TILES", "1")))
    b_reuse_m_tiles = max(1, int(os.getenv("GOLEM_B_REUSE_M_TILES", "1")))
    m_groups = (m_tiles + b_reuse_m_tiles - 1) // b_reuse_m_tiles
    n_groups = (n_tiles + a_reuse_n_tiles - 1) // a_reuse_n_tiles
    total_macro_tasks = m_groups * n_groups

    mat_reuse_slots = b_reuse_m_tiles if b_reuse_m_tiles > 1 else 1
    vec_reuse_slots = a_reuse_n_tiles if a_reuse_n_tiles > 1 else 1
    out_reuse_slots = mat_reuse_slots * vec_reuse_slots

    def macro_task_for_group(m_group: int, n_group: int) -> int:
        n_band = (n_group - m_group) % n_groups
        return n_band * m_groups + m_group

    def layout_owner_core_for_task(task_id: int) -> int:
        if active_gemm_cores <= 0:
            return first_worker_core
        return first_worker_core + (task_id % active_gemm_cores)

    def layout_data_node_for_task(task_id: int) -> int:
        owner_core = layout_owner_core_for_task(task_id)
        group_idx = owner_core % total_groups if total_groups > 0 else 0
        if not data_nodes:
            return os_memory_node
        return data_nodes[group_idx % len(data_nodes)]

    def layout_a_data_node_for_m_tile(m_tile: int) -> int:
        return layout_data_node_for_task(m_tile // b_reuse_m_tiles)

    def layout_b_data_node_for_n_tile(n_tile: int) -> int:
        return layout_data_node_for_task(n_tile // a_reuse_n_tiles)

    max_macro_tasks_per_data_node = max(
        (
            sum(
                1
                for task_id in range(total_macro_tasks)
                if layout_data_node_for_task(task_id) == node_idx
            )
            for node_idx in data_nodes
        ),
        default=1,
    )
    max_a_m_tiles_per_data_node = max(
        (
            sum(
                1
                for m_tile in range(m_tiles)
                if layout_a_data_node_for_m_tile(m_tile) == node_idx
            )
            for node_idx in data_nodes
        ),
        default=1,
    )
    max_b_n_tiles_per_data_node = max(
        (
            sum(
                1
                for n_tile in range(n_tiles)
                if layout_b_data_node_for_n_tile(n_tile) == node_idx
            )
            for node_idx in data_nodes
        ),
        default=1,
    )
    off_gemm_mat = 0
    off_gemm_vec_base = off_gemm_mat + max_a_m_tiles_per_data_node * k_tiles * mm_mat_stride
    off_gemm_out_base = (
        off_gemm_vec_base
        + max_b_n_tiles_per_data_node * k_tiles * block_n * mm_vec_stride
    )
    out_tile_bytes = block_m * block_n * elem_bytes
    out_tile_stride = align_up(out_tile_bytes, mm_align)
    gemm_data_region_end = (
        off_gemm_out_base + max_macro_tasks_per_data_node * out_reuse_slots * out_tile_stride
    )
    rowmajor_matrix_bytes = m * n * elem_bytes
    off_sfu_softmax_rowmajor_base = align_up(gemm_data_region_end, mm_align)
    off_sfu_softmax_rowmajor_out_base = (
        off_sfu_softmax_rowmajor_base + align_up(rowmajor_matrix_bytes, mm_align)
    )

    node_buffers = {}
    for node_idx in data_nodes:
        path = os.path.join(hbm_dir, f"hbm_out_node{node_idx}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing output backing file: {path}")
        with open(path, "rb") as f:
            node_buffers[node_idx] = f.read()

    def owner_core_for_task(task_id: int) -> int:
        if active_gemm_cores <= 0:
            return first_worker_core
        worker_slot = task_id % active_gemm_cores
        return worker_slot + first_worker_core

    def data_node_for_task(task_id: int) -> int:
        owner_core = owner_core_for_task(task_id)
        group_idx = owner_core % total_groups
        if not data_nodes:
            return os_memory_node
        return data_nodes[group_idx % len(data_nodes)]

    def task_slot_in_node(task_id: int) -> int:
        node_idx = data_node_for_task(task_id)
        slot = 0
        for prev in range(task_id):
            if data_node_for_task(prev) == node_idx:
                slot += 1
        return slot

    c = [[cast_scalar(dtype, 0) for _ in range(n)] for _ in range(m)]

    if direct_rowmajor_softmax:
        if not data_nodes:
            raise ValueError("direct row-major softmax unpack requires at least one data node")
        node_idx = data_nodes[0]
        node_data = node_buffers[node_idx]
        raw = node_data[
            off_sfu_softmax_rowmajor_out_base :
            off_sfu_softmax_rowmajor_out_base + rowmajor_matrix_bytes
        ]
        vals = unpack_values(dtype, raw)
        for r in range(m):
            row_base = r * n
            for cc in range(n):
                c[r][cc] = vals[row_base + cc]
    else:
        for task_id in range(total_tasks):
            m_tile = task_id // n_tiles
            n_tile = task_id % n_tiles
            m_group = m_tile // b_reuse_m_tiles
            n_group = n_tile // a_reuse_n_tiles
            m_offset = m_tile % b_reuse_m_tiles
            n_offset = n_tile % a_reuse_n_tiles
            macro_task_id = macro_task_for_group(m_group, n_group)
            reuse_offset = m_offset * vec_reuse_slots + n_offset
            node_idx = data_node_for_task(macro_task_id)
            slot = task_slot_in_node(macro_task_id)
            tile_off = off_gemm_out_base + (slot * out_reuse_slots + reuse_offset) * out_tile_stride

            node_data = node_buffers[node_idx]
            tile_raw = node_data[tile_off : tile_off + out_tile_bytes]
            vals = unpack_values(dtype, tile_raw)

            for r in range(block_m):
                for cc in range(block_n):
                    if out_layout in {"colmajor", "colmajor_tile", "columnmajor"}:
                        value = vals[cc * block_m + r]
                    else:
                        value = vals[r * block_n + cc]
                    c[m_tile * block_m + r][n_tile * block_n + cc] = value

    out_file = args.out_file
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    if out_file.endswith(".csv"):
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(c)
    else:
        flat = [item for row in c for item in row]
        with open(out_file, "wb") as f:
            f.write(pack_values(dtype, flat))

    print(f"[UNPACK] wrote C tensor to: {out_file}")
    print(
        f"[UNPACK] shape=({m},{n}), block=({block_m},{block_n},{block_k}), dtype={dtype}, layout={out_layout}, direct_rowmajor_softmax={int(direct_rowmajor_softmax)}, tasks={total_tasks}, macro_tasks={total_macro_tasks}, a_reuse_n={a_reuse_n_tiles}, b_reuse_m={b_reuse_m_tiles}"
    )


if __name__ == "__main__":
    main()
