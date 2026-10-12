# pipeline/pipeline.py
from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.stage5 import stage5


def run_pipeline(incremental=True, ingest=True):
    """stage1 ingestion → stage2 DB to daily frame → stage3 daily preparation →
    stage4 event frame + Model B → stage5 outputs.

    `incremental` picks the ingestion provider (see stage1). `ingest=False` skips stage 1
    and re-scores the existing DB without any API call.

    Scoring and local outputs only: it never deploys, mails or writes the prediction
    archive. Publishing is scripts/full_workflow.sh (master only), which archives the
    published calls with publishing/archive_calls.py.
    """
    if ingest:
        stage1(incremental=incremental)
    df = stage2()
    daily = stage3(df)
    events = stage4(daily)
    stage5(daily, events)
