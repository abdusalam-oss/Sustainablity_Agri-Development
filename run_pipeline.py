"""
run_pipeline.py
================
Master pipeline script for the GAN-based Super-Resolution Agricultural
Monitoring project (Al-Qassim, Saudi Arabia).

This script runs the full end-to-end processing chain in the correct order:

  Stage 0 — Environment check
  Stage 1 — Preprocessing (all five data sources)
  Stage 2 — Super-resolution model training & evaluation
  Stage 3 — Vegetation & water index computation
  Stage 4 — Spatiotemporal analysis
  Stage 5 — Trend & correlation analysis
  Stage 6 — Validation & model comparison
  Stage 7 — Figure generation

Each stage can be run individually via --stages, e.g.:
    python run_pipeline.py --stages 1 2
    python run_pipeline.py --stages 3 4 5 6
    python run_pipeline.py --stages 7

Run the full pipeline:
    python run_pipeline.py

Skip training and use pre-trained checkpoints:
    python run_pipeline.py --skip_training

References
----------
Paper: GAN-based super-resolution framework for sustainable agricultural
       and water resource monitoring using multisensor remote sensing.
       (Al-Qassim, Saudi Arabia)
Repository: https://github.com/<repository-to-be-confirmed>
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import platform
from pathlib import Path

# Ensure repo root is on the Python path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    RAW_LANDSAT, RAW_GRACE, RAW_CRUTS, RAW_PME, RAW_PLANETSCOPE,
    PROC_LANDSAT, PROC_GRACE, PROC_CRUTS, PROC_PME, PROC_PLANETSCOPE,
    RES_BICUBIC, RES_SRCNN, RES_SRGAN, RES_INDICES, RES_ANALYSIS,
    FIGURES_DIR, CHECKPOINTS_DIR,
    SRCNN as SRCNN_CFG,
    SRGAN as SRGAN_CFG,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(ROOT, "pipeline.log"), mode="a"),
    ],
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._start = time.time()
        log.info(">>> Starting: %s", self.label)
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        log.info("<<< Finished: %s  (%.1f s)", self.label, elapsed)


# ---------------------------------------------------------------------------
# Stage 0 — Environment check
# ---------------------------------------------------------------------------

def stage0_env_check() -> bool:
    """Verify that all required packages and data directories are present."""
    log.info("=" * 60)
    log.info("STAGE 0 — Environment Check")
    log.info("=" * 60)
    log.info("Python  : %s", sys.version.split()[0])
    log.info("Platform: %s", platform.platform())
    log.info("Root dir: %s", ROOT)

    required_packages = [
        "numpy", "pandas", "scipy", "sklearn",
        "rasterio", "shapely", "netCDF4",
        "skimage", "tifffile", "statsmodels",
        "matplotlib",
    ]
    optional_packages = ["tensorflow"]

    all_ok = True
    for pkg in required_packages:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
            log.info("  [OK] %s", pkg)
        except ImportError:
            log.error("  [MISSING] %s — install with: pip install %s", pkg, pkg)
            all_ok = False

    for pkg in optional_packages:
        try:
            __import__(pkg)
            log.info("  [OK] %s (optional)", pkg)
        except ImportError:
            log.warning("  [OPTIONAL MISSING] %s — SR model training will be skipped.", pkg)

    # Check raw data directories exist and have content
    raw_dirs = {
        "Landsat"   : RAW_LANDSAT,
        "GRACE"     : RAW_GRACE,
        "CRU-TS"    : RAW_CRUTS,
        "PME"       : RAW_PME,
        "PlanetScope": RAW_PLANETSCOPE,
    }
    for name, path in raw_dirs.items():
        n_files = len(list(Path(path).rglob("*"))) if Path(path).exists() else 0
        if n_files == 0:
            log.warning("  [EMPTY] %s raw data directory: %s", name, path)
        else:
            log.info("  [OK] %s — %d file(s) in %s", name, n_files, path)

    if all_ok:
        log.info("Environment check passed.")
    else:
        log.error("Environment check failed — fix missing packages before continuing.")
    return all_ok


# ---------------------------------------------------------------------------
# Stage 1 — Preprocessing
# ---------------------------------------------------------------------------

def stage1_preprocessing(force: bool = False) -> None:
    log.info("=" * 60)
    log.info("STAGE 1 — Preprocessing")
    log.info("=" * 60)

    # 1a — Landsat
    with _Timer("Landsat preprocessing"):
        from preprocessing.landsat_preprocess import main as landsat_main
        sys.argv = ["landsat_preprocess.py",
                    "--input",  RAW_LANDSAT,
                    "--output", PROC_LANDSAT]
        try:
            landsat_main()
        except SystemExit:
            pass
        except Exception as exc:
            log.error("Landsat preprocessing failed: %s", exc)

    # 1b — GRACE
    with _Timer("GRACE preprocessing"):
        from preprocessing.grace_preprocess import process_grace_directory
        try:
            process_grace_directory(RAW_GRACE, PROC_GRACE)
        except Exception as exc:
            log.error("GRACE preprocessing failed: %s", exc)

    # 1c — CRU-TS
    with _Timer("CRU-TS reprojection"):
        from preprocessing.cru_ts_reproject import process_all
        try:
            process_all(RAW_CRUTS, PROC_CRUTS)
        except Exception as exc:
            log.error("CRU-TS preprocessing failed: %s", exc)

    # 1d — PME Met Station
    with _Timer("PME quality control"):
        from preprocessing.pme_qc import process_pme
        try:
            process_pme(RAW_PME, PROC_PME)
        except Exception as exc:
            log.error("PME QC failed: %s", exc)

    log.info("Stage 1 complete.")


# ---------------------------------------------------------------------------
# Stage 2 — Super-resolution training & evaluation
# ---------------------------------------------------------------------------

def stage2_super_resolution(skip_training: bool = False) -> None:
    log.info("=" * 60)
    log.info("STAGE 2 — Super-Resolution Models")
    log.info("=" * 60)

    # 2a — Bicubic baseline (no training)
    with _Timer("Bicubic interpolation evaluation"):
        from models.bicubic_interpolation import evaluate_on_test_set
        try:
            evaluate_on_test_set(PROC_LANDSAT, RES_BICUBIC)
        except Exception as exc:
            log.error("Bicubic evaluation failed: %s", exc)

    if skip_training:
        log.info("Training skipped (--skip_training). "
                 "Using existing checkpoints for SRCNN and SRGAN.")
        _eval_only_srcnn()
        _eval_only_srgan()
        return

    # Check TensorFlow availability
    try:
        import tensorflow as tf
        log.info("TensorFlow %s detected — proceeding with training.", tf.__version__)
    except ImportError:
        log.error("TensorFlow not found. Install it with: pip install tensorflow==2.10")
        log.warning("Skipping SRCNN and SRGAN training.")
        return

    # 2b — SRCNN
    with _Timer(f"SRCNN training ({SRCNN_CFG['epochs']} epochs)"):
        from models.srcnn import train as srcnn_train, evaluate as srcnn_eval
        try:
            srcnn_train(
                data_dir   = PROC_LANDSAT,
                output_dir = RES_SRCNN,
                epochs     = SRCNN_CFG["epochs"],
                batch_size = SRCNN_CFG["batch_size"],
                lr         = SRCNN_CFG["lr"],
            )
            srcnn_eval(PROC_LANDSAT, RES_SRCNN)
        except Exception as exc:
            log.error("SRCNN training/eval failed: %s", exc)

    # 2c — SRGAN
    with _Timer(f"SRGAN training ({SRGAN_CFG['epochs']} epochs)"):
        from models.srgan import train as srgan_train, evaluate as srgan_eval
        try:
            srgan_train(
                data_dir   = PROC_LANDSAT,
                output_dir = RES_SRGAN,
                epochs     = SRGAN_CFG["epochs"],
                batch_size = SRGAN_CFG["batch_size"],
                lr_gen     = SRGAN_CFG["lr_gen"],
                lr_disc    = SRGAN_CFG["lr_disc"],
            )
            srgan_eval(PROC_LANDSAT, RES_SRGAN)
        except Exception as exc:
            log.error("SRGAN training/eval failed: %s", exc)

    log.info("Stage 2 complete.")


def _eval_only_srcnn() -> None:
    try:
        from models.srcnn import evaluate as srcnn_eval
        srcnn_eval(PROC_LANDSAT, RES_SRCNN)
    except Exception as exc:
        log.error("SRCNN evaluation failed: %s", exc)


def _eval_only_srgan() -> None:
    try:
        from models.srgan import evaluate as srgan_eval
        srgan_eval(PROC_LANDSAT, RES_SRGAN)
    except Exception as exc:
        log.error("SRGAN evaluation failed: %s", exc)


# ---------------------------------------------------------------------------
# Stage 3 — Vegetation & water indices
# ---------------------------------------------------------------------------

def stage3_indices() -> None:
    log.info("=" * 60)
    log.info("STAGE 3 — Vegetation & Water Indices")
    log.info("=" * 60)

    with _Timer("Vegetation indices (NDVI, SAVI, MSAVI2)"):
        from indices.vegetation_indices import build_vi_timeseries
        try:
            build_vi_timeseries(RES_SRGAN, RES_INDICES)
        except Exception as exc:
            log.error("Vegetation index computation failed: %s", exc)

    with _Timer("Water indices (NDWI, MNDWI, AWEI)"):
        from indices.water_indices import water_area_timeseries
        try:
            water_area_timeseries(RES_SRGAN, RES_INDICES)
        except Exception as exc:
            log.error("Water index computation failed: %s", exc)

    log.info("Stage 3 complete.")


# ---------------------------------------------------------------------------
# Stage 4 — Spatiotemporal analysis
# ---------------------------------------------------------------------------

def stage4_spatiotemporal() -> None:
    log.info("=" * 60)
    log.info("STAGE 4 — Spatiotemporal Analysis")
    log.info("=" * 60)

    vi_csv    = os.path.join(RES_INDICES, "vegetation_indices_timeseries.csv")
    grace_csv = os.path.join(PROC_GRACE,  "grace_tws_alqassim_timeseries.csv")
    tmp_csv   = os.path.join(PROC_CRUTS,  "cruts_tmp_alqassim_timeseries.csv")
    pre_csv   = os.path.join(PROC_CRUTS,  "cruts_pre_alqassim_timeseries.csv")

    with _Timer("Spatiotemporal analysis"):
        from analysis.spatiotemporal_analysis import run_full_analysis
        try:
            run_full_analysis(vi_csv, grace_csv, tmp_csv, pre_csv, RES_ANALYSIS)
        except Exception as exc:
            log.error("Spatiotemporal analysis failed: %s", exc)

    log.info("Stage 4 complete.")


# ---------------------------------------------------------------------------
# Stage 5 — Trend & correlation analysis
# ---------------------------------------------------------------------------

def stage5_trend_correlation() -> None:
    log.info("=" * 60)
    log.info("STAGE 5 — Trend & Correlation Analysis")
    log.info("=" * 60)

    vi_csv    = os.path.join(RES_INDICES, "vegetation_indices_timeseries.csv")
    grace_csv = os.path.join(PROC_GRACE,  "grace_tws_alqassim_timeseries.csv")
    tmp_csv   = os.path.join(PROC_CRUTS,  "cruts_tmp_alqassim_timeseries.csv")
    pre_csv   = os.path.join(PROC_CRUTS,  "cruts_pre_alqassim_timeseries.csv")

    with _Timer("VI linear trends + STL decomposition + ACF"):
        from analysis.trend_correlation import all_trends
        try:
            all_trends(vi_csv, RES_ANALYSIS)
        except Exception as exc:
            log.error("Trend analysis failed: %s", exc)

    with _Timer("Pearson & Spearman correlation matrix"):
        from analysis.trend_correlation import correlation_matrix
        try:
            correlation_matrix(vi_csv, tmp_csv, pre_csv, grace_csv, RES_ANALYSIS)
        except Exception as exc:
            log.error("Correlation matrix failed: %s", exc)

    log.info("Stage 5 complete.")


# ---------------------------------------------------------------------------
# Stage 6 — Validation & model comparison
# ---------------------------------------------------------------------------

def stage6_validation() -> None:
    log.info("=" * 60)
    log.info("STAGE 6 — Validation & Model Comparison")
    log.info("=" * 60)

    with _Timer("Model comparison table"):
        from analysis.validation import model_comparison_table
        try:
            model_comparison_table(RES_SRGAN, RES_ANALYSIS)
        except Exception as exc:
            log.error("Model comparison failed: %s", exc)

    # PlanetScope scene validation (if available)
    from glob import glob
    ps_refs = sorted(glob(os.path.join(PROC_PLANETSCOPE, "*.tif")))
    sr_scenes = sorted(glob(os.path.join(RES_SRGAN, "*_srgan_sr.tif")))

    if ps_refs and sr_scenes:
        with _Timer("PlanetScope scene-level validation"):
            from analysis.validation import validate_against_planetscope
            try:
                validate_against_planetscope(sr_scenes[0], ps_refs[0], RES_ANALYSIS)
            except Exception as exc:
                log.error("PlanetScope validation failed: %s", exc)
    else:
        log.info("No PlanetScope reference or SR scenes found — "
                 "skipping scene-level validation.")

    log.info("Stage 6 complete.")


# ---------------------------------------------------------------------------
# Stage 7 — Figures
# ---------------------------------------------------------------------------

def stage7_figures() -> None:
    log.info("=" * 60)
    log.info("STAGE 7 — Figure Generation")
    log.info("=" * 60)

    with _Timer("All paper figures"):
        from visualisation.plot_results import plot_all
        try:
            plot_all(RES_ANALYSIS, RES_INDICES, FIGURES_DIR)
        except Exception as exc:
            log.error("Figure generation failed: %s", exc)

    log.info("Stage 7 complete.")
    log.info("Figures saved to: %s", FIGURES_DIR)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

STAGE_MAP = {
    0: ("Environment check",        stage0_env_check),
    1: ("Preprocessing",            stage1_preprocessing),
    2: ("Super-resolution models",  stage2_super_resolution),
    3: ("Vegetation & water indices", stage3_indices),
    4: ("Spatiotemporal analysis",  stage4_spatiotemporal),
    5: ("Trend & correlation",      stage5_trend_correlation),
    6: ("Validation",               stage6_validation),
    7: ("Figure generation",        stage7_figures),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline for GAN-based super-resolution "
            "agricultural monitoring (Al-Qassim, Saudi Arabia)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  Run full pipeline:
    python run_pipeline.py

  Run only preprocessing and SR training:
    python run_pipeline.py --stages 1 2

  Run analysis + figures with pre-trained models:
    python run_pipeline.py --stages 3 4 5 6 7 --skip_training

  Run only figure generation:
    python run_pipeline.py --stages 7
        """,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=list(STAGE_MAP.keys()),
        default=list(STAGE_MAP.keys()),
        help="Stage numbers to run (0–7). Runs all by default.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip SRCNN/SRGAN training; use existing checkpoint files.",
    )
    parser.add_argument(
        "--skip_env_check",
        action="store_true",
        help="Skip environment check (Stage 0).",
    )
    args = parser.parse_args()

    stages = sorted(set(args.stages))
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  SR Agricultural Monitoring Pipeline                 ║")
    log.info("║  Al-Qassim, Saudi Arabia                             ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("Stages to run: %s", stages)
    if args.skip_training:
        log.info("Training skipped — will use pre-trained checkpoints.")

    pipeline_start = time.time()

    for stage_num in stages:
        name, fn = STAGE_MAP[stage_num]

        if stage_num == 0 and args.skip_env_check:
            log.info("Skipping Stage 0 (--skip_env_check).")
            continue

        log.info("")
        try:
            if stage_num == 2:
                fn(skip_training=args.skip_training)
            elif stage_num == 0:
                ok = fn()
                if not ok and not args.skip_env_check:
                    log.error("Aborting pipeline — fix environment issues first.")
                    sys.exit(1)
            else:
                fn()
        except KeyboardInterrupt:
            log.warning("Pipeline interrupted by user at Stage %d.", stage_num)
            sys.exit(0)
        except Exception as exc:
            log.exception("Unhandled error in Stage %d (%s): %s", stage_num, name, exc)
            log.warning("Continuing to next stage …")

    elapsed = time.time() - pipeline_start
    log.info("")
    log.info("Pipeline complete. Total elapsed time: %.1f s (%.1f min).",
             elapsed, elapsed / 60)
    log.info("Results  : %s", os.path.join(ROOT, "results"))
    log.info("Figures  : %s", FIGURES_DIR)
    log.info("Log file : %s", os.path.join(ROOT, "pipeline.log"))


if __name__ == "__main__":
    main()
