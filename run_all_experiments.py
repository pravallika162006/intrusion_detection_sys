"""
Master Runner Script: Executes all three phases of the Intrusion Detection System reproduction & enhancement.
- Phase 1: Full 42-feature UNSW-NB15 ML experiments (DT, ANN, kNN, LR, SVM)
- Phase 2: XGBoost Feature Selection -> exact paper 19 features -> repeated ML experiments
- Phase 3: Project Enhancement (Alternative candidate models & ensembles on validation data)
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.utils.logger import setup_logger

logger = setup_logger("MasterRunner")


def main():
    logger.info("#########################################################################")
    logger.info(" INTRUSION DETECTION SYSTEM: FULL 3-PHASE ACADEMIC SUITE EXECUTION ")
    logger.info(" Sydney M. Kasongo & Yanxia Sun (2020) Reproduction + Project Enhancement ")
    logger.info("#########################################################################\n")

    t_start = time.time()

    # Phase 1
    logger.info("=========================================================================")
    logger.info(" LAUNCHING PHASE 1: 42-FEATURE EXPERIMENTS ")
    logger.info("=========================================================================")
    import run_phase1
    run_phase1.main()

    # Phase 2
    logger.info("\n=========================================================================")
    logger.info(" LAUNCHING PHASE 2: XGBOOST FEATURE SELECTION & 19-FEATURE EXPERIMENTS ")
    logger.info("=========================================================================")
    import run_phase2
    run_phase2.main()

    # Phase 3
    logger.info("\n=========================================================================")
    logger.info(" LAUNCHING PHASE 3: PROJECT ENHANCEMENT (CANDIDATES & ENSEMBLES) ")
    logger.info("=========================================================================")
    import run_phase3
    run_phase3.main()

    total_elapsed = time.time() - t_start
    logger.info("\n#########################################################################")
    logger.info(f" ALL 3 PHASES COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS ({total_elapsed/60:.2f} MINUTES) ")
    logger.info("#########################################################################")


if __name__ == "__main__":
    main()
