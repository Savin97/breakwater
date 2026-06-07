from pipeline.stage1 import stage1
from pipeline.incremental import run_incremental

stage1(update=True)
run_incremental()
