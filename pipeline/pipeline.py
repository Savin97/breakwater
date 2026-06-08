# pipeline/pipeline.py
import pandas as pd
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
    df.to_parquet("output/full_df.parquet", index=False)
    print("Wrote output/full_df.parquet")
    stage5(df)