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

source = source.replace(
    '"num_vns": 3,\n'
    '            "golem_dma_response_chunk_bytes": os.getenv("GOLEM_DMA_RESPONSE_CHUNK_BYTES", "0"),\n'
    '            "golem_dma_response_vn": os.getenv("GOLEM_DMA_RESPONSE_VN", "0"),',
    '"num_vns": 1,\n'
    '            "network_input_buffer_size": os.getenv("GOLEM_NOC_INPUT_BUF_SIZE", "512KB"),\n'
    '            "network_output_buffer_size": os.getenv("GOLEM_NOC_OUTPUT_BUF_SIZE", "512KB"),\n'
    '            "golem_dma_response_drain_limit": os.getenv("GOLEM_DMA_RESPONSE_DRAIN_LIMIT", "0"),\n'
    '            "golem_dma_response_chunk_bytes": os.getenv("GOLEM_DMA_RESPONSE_CHUNK_BYTES", "0"),\n'
    '            "golem_dma_response_vn": "0",',
    1,
)

globals_dict = {
    "__file__": ARCHIVE_SCRIPT,
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(source, ARCHIVE_SCRIPT, "exec"), globals_dict)
