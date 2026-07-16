# Softmax compatibility support

This directory is not an active CPU Softmax test path. Unified SFU job
Softmax keeps only two compatibility dependencies here:

- `golem_softmax_runtime.{h,cpp}`: shared request/status definitions used by
  the SFU guest runtime.
- `ncores_selfcom_dma_softmax_archive.py`: isolated architecture shim used by
  the current explicit-NoC runner.

The former CPU Softmax executables, dimension wrappers, prototypes, tests and
historical documents were removed after the unified SFU job path became the
only active Softmax workflow. Do not add new experiments here.
