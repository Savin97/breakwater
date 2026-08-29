from pipeline.stage1 import stage1
from pipeline.incremental import run_incremental

stage1(incremental=True)
run_incremental()
