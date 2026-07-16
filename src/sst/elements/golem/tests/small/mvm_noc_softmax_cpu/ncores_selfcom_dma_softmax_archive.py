#!/usr/bin/env python3
"""Local architecture shim for the softmax 16-core smoke path.

The upstream archive architecture sends memory-side DMA replies on VN1 because
memNICBase derives the response VN from num_vns. In this archive path the
GlobalMemory endpoint does not drain those replies in time, so the request side
waits forever. Keep the original architecture intact and run it with directory
MemNIC DMA replies on VN0 for this isolated softmax test folder.
"""

import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ARCHIVE_SCRIPT = os.path.join(TESTS_DIR, "architecture", "archive", "ncores_selfcom_dma.py")

with open(ARCHIVE_SCRIPT, "r", encoding="utf-8") as source_file:
    source = source_file.read()

directory_memnic_fragment = (
    '"num_vns": 3,\n'
    '            "network_input_buffer_size": os.getenv(\n'
    '                "GOLEM_DIRCTRL_HIGHLINK_INPUT_BUF_SIZE", "64KB"\n'
    '            ),\n'
    '            "network_output_buffer_size": os.getenv(\n'
    '                "GOLEM_DIRCTRL_HIGHLINK_OUTPUT_BUF_SIZE", "64KB"\n'
    '            ),\n'
    '            "golem_dma_response_chunk_bytes": os.getenv("GOLEM_DMA_RESPONSE_CHUNK_BYTES", "0"),\n'
    '            "golem_dma_response_vn": os.getenv("GOLEM_DMA_RESPONSE_VN", "1"),'
)
directory_memnic_replacement = (
    '"num_vns": 3,\n'
    '            "network_input_buffer_size": os.getenv("GOLEM_NOC_INPUT_BUF_SIZE", "512KB"),\n'
    '            "network_output_buffer_size": os.getenv("GOLEM_NOC_OUTPUT_BUF_SIZE", "512KB"),\n'
    '            "golem_dma_response_drain_limit": os.getenv("GOLEM_DMA_RESPONSE_DRAIN_LIMIT", "0"),\n'
    '            "golem_dma_response_chunk_bytes": os.getenv("GOLEM_DMA_RESPONSE_CHUNK_BYTES", "0"),\n'
    '            "golem_dma_response_vn": "0",'
)
directory_memnic_fragment_count = source.count(directory_memnic_fragment)
if directory_memnic_fragment_count != 1:
    raise RuntimeError(
        "expected exactly one archive directory MemNIC fragment, "
        f"found {directory_memnic_fragment_count}"
    )
source = source.replace(
    directory_memnic_fragment,
    directory_memnic_replacement,
    1,
)
source = source.replace(
    "sst.setStatisticLoadLevel(16)\n"
    'sst.enableAllStatisticsForAllComponents({"type": "sst.AccumulatorStatistic"})',
    'sst.setStatisticLoadLevel(int(os.getenv("GOLEM_SST_STAT_LOAD_LEVEL", "0")))\n'
    'if int(os.getenv("GOLEM_SST_ENABLE_ALL_STATS", "0")) != 0:\n'
    '    sst.enableAllStatisticsForAllComponents({"type": "sst.AccumulatorStatistic"})',
    1,
)
source = source.replace(
    "    mem_backend.enableAllStatistics()\n",
    '    if int(os.getenv("GOLEM_SST_ENABLE_ALL_STATS", "0")) != 0:\n'
    "        mem_backend.enableAllStatistics()\n",
    1,
)
source = source.replace(
    '    "GOLEM_SILENT",\n]',
    '    "GOLEM_SILENT",\n'
    '    "GOLEM_SFU_SKIP_SOFTMAX",\n]',
    1,
)
source = source.replace(
    '    "GOLEM_SFU_SKIP_SOFTMAX",\n]',
    '    "GOLEM_SFU_SKIP_SOFTMAX",\n'
    '    "GOLEM_SFU_INTERLEAVE_GEMM",\n]',
    1,
)
source = source.replace(
    '    "GOLEM_SFU_INTERLEAVE_GEMM",\n]',
    '    "GOLEM_SFU_INTERLEAVE_GEMM",\n'
    '    "GOLEM_SFU_STANDALONE_SOFTMAX",\n]',
    1,
)
source = source.replace(
    '    "GOLEM_SFU_STANDALONE_SOFTMAX",\n]',
    '    "GOLEM_SFU_STANDALONE_SOFTMAX",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX_CHUNK_ELEMS",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX_WORKER_CORES",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX_STAGING_ROWS",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX_TRACE_BANDS",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX_BAND_CORES",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX_JOB_ROWS",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX_DIRECT_ROWMAJOR_HBM",\n'
    '    "GOLEM_SFU_JOB_SOFTMAX_DISTRIBUTED_COLUMNS",\n'
    '    "GOLEM_SFU_PRIMITIVE_HBM_STREAM",\n'
    '    "GOLEM_SFU_PRIMITIVE_HBM_ELEMS",\n'
    '    "GOLEM_SFU_PRIMITIVE_HBM_CHUNK_ELEMS",\n'
    '    "GOLEM_SFU_PRIMITIVE_HBM_OPS",\n'
    '    "GOLEM_SFU_PRIMITIVE_HBM_BATCH",\n'
    '    "GOLEM_SFU_PRIMITIVE_SMOKE",\n]',
    1,
)
source = source.replace(
    '    "GOLEM_SFU_PRIMITIVE_SMOKE",\n]',
    '    "GOLEM_SFU_PRIMITIVE_SMOKE",\n'
    '    "GOLEM_SFU_PRIMITIVE_SMOKE_ELEMS",\n'
    '    "GOLEM_SFU_PRIMITIVE_SMOKE_CHUNK_ELEMS",\n]',
    1,
)

gm_buffer_length = os.getenv("GOLEM_GM_BUFFER_LENGTH", "64KB")
print(f"[GOLEM] GlobalMemory link buffer_length={gm_buffer_length}")

globals_dict = {
    "__file__": ARCHIVE_SCRIPT,
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(source, ARCHIVE_SCRIPT, "exec"), globals_dict)
