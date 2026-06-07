# pipeline/pipeline.py
from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.stage5 import stage5


def run_pipeline():
    stage1(update=False)
    df = stage2()
    df = stage3(df)
    df = stage4(df)
    stage5(df)