# pipeline/pipeline.py
from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.stage5 import stage5
def run_pipeline():
    stage1(update=False) # 1. Build/Update DB
    df_2 = stage2()      # 2. Import from DB, merge and filter data (by start date, end date, stocks, etc)
    df_3 = stage3(df_2)  # 3. Engineer features
    df_4 = stage4(df_3)  # 4. Calculate Risk Score and Provide Explanations
    df_5 = stage5(df_4)  # 5. Report Generation