# utilities/logging_utilities.py
import logging
import sys
from config import LOG_LEVEL, NOISY_LIBRARIES

def setup_logging(level: str | None = None):
    """
        Configure logging once, at program start. Writes to stdout, the same place
        print() went, so the cron jobs' shell redirection keeps working unchanged.

        Level comes from config.LOG_LEVEL unless overridden here.
        force=True makes it safe to call more than once in a process.
    """
    logging.basicConfig(
        level=(level or LOG_LEVEL).upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Cap third-party libraries at their own levels, so LOG_LEVEL here only
    # turns up detail from our own code.
    for library, library_level in NOISY_LIBRARIES.items():
        logging.getLogger(library).setLevel(library_level)
