"""
config.py
=========
Central configuration for the GAN-based Super-Resolution Agricultural
Monitoring project (Al-Qassim, Saudi Arabia).

All paths, hyperparameters, and dataset constants are defined here.
Import this module in every script to guarantee consistency.
"""

import os

# ---------------------------------------------------------------------------
# Project root (resolved relative to this file so the repo is portable)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR        = os.path.join(ROOT_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
RESULTS_DIR     = os.path.join(ROOT_DIR, "results")
FIGURES_DIR     = os.path.join(ROOT_DIR, "figures")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

# Raw sub-directories (one per data source)
RAW_LANDSAT     = os.path.join(RAW_DIR, "landsat")
RAW_GRACE       = os.path.join(RAW_DIR, "grace")
RAW_PLANETSCOPE = os.path.join(RAW_DIR, "planetscope")
RAW_CRUTS       = os.path.join(RAW_DIR, "cru_ts")
RAW_PME         = os.path.join(RAW_DIR, "pme_station")

# Processed sub-directories
PROC_LANDSAT    = os.path.join(PROCESSED_DIR, "landsat")
PROC_GRACE      = os.path.join(PROCESSED_DIR, "grace")
PROC_PLANETSCOPE= os.path.join(PROCESSED_DIR, "planetscope")
PROC_CRUTS      = os.path.join(PROCESSED_DIR, "cru_ts")
PROC_PME        = os.path.join(PROCESSED_DIR, "pme")

# Results sub-directories
RES_BICUBIC     = os.path.join(RESULTS_DIR, "bicubic")
RES_SRCNN       = os.path.join(RESULTS_DIR, "srcnn")
RES_SRGAN       = os.path.join(RESULTS_DIR, "srgan")
RES_INDICES     = os.path.join(RESULTS_DIR, "indices")
RES_ANALYSIS    = os.path.join(RESULTS_DIR, "analysis")

# ---------------------------------------------------------------------------
# Study area — Al-Qassim, Saudi Arabia
# ---------------------------------------------------------------------------
STUDY_AREA = {
    "name"    : "Al-Qassim",
    "country" : "Saudi Arabia",
    "epsg"    : 32638,                          # WGS 84 / UTM Zone 38N
    "bbox_wgs84": (43.0, 25.5, 45.5, 27.5),    # (min_lon, min_lat, max_lon, max_lat)
    "area_km2": 73_000,
}

# ---------------------------------------------------------------------------
# Landsat
# ---------------------------------------------------------------------------
LANDSAT = {
    "collection"   : "landsat_ot_c2_l2",
    "cloud_cover"  : 10,          # maximum cloud cover (%)
    "bands"        : {
        "blue"  : "SR_B2",
        "green" : "SR_B3",
        "red"   : "SR_B4",
        "nir"   : "SR_B5",
        "swir1" : "SR_B6",
        "swir2" : "SR_B7",
        "qa"    : "QA_PIXEL",
    },
    "scale_factor" : 0.0000275,
    "offset"       : -0.2,
    "resolution_m" : 30,
    "crs"          : "EPSG:32638",
}

# ---------------------------------------------------------------------------
# GRACE
# ---------------------------------------------------------------------------
GRACE = {
    "product"     : "GRCTellus.JPL.200204_202310.GLO.RL06.1M.MSCNv03CRI",
    "variable"    : "lwe_thickness",      # liquid water equivalent thickness (cm)
    "start_date"  : "2003-01",
    "end_date"    : "2020-06",
    "filter"      : "DDK5",
    "unit_out"    : "mm",                 # converted from cm in preprocessing
    "fill_value"  : -99999.0,
}

# ---------------------------------------------------------------------------
# PlanetScope
# ---------------------------------------------------------------------------
PLANETSCOPE = {
    "dates"       : ["2020-10", "2020-02"],
    "resolution_m": 3,
    "bands"       : ["blue", "green", "red", "nir"],
    "resampling"  : "nearest",            # for co-registration to Landsat grid
}

# ---------------------------------------------------------------------------
# CRU-TS
# ---------------------------------------------------------------------------
CRUTS = {
    "version"     : "4.05",
    "variables"   : ["tmp", "pre"],       # temperature (°C), precipitation (mm/month)
    "start_year"  : 1982,
    "end_year"    : 2020,
    "resolution_deg": 0.5,
    "doi"         : "https://doi.org/10.1038/s41597-020-0453-3",
}

# ---------------------------------------------------------------------------
# PME Met Station
# ---------------------------------------------------------------------------
PME = {
    "station"     : "Buraidah",
    "variable"    : "rainfall_mm",
    "start_year"  : 1982,
    "end_year"    : 2020,
    "gap_fill_max_months": 3,             # max consecutive missing months to interpolate
    "outlier_std" : 3.0,                  # z-score threshold for outlier removal
}

# ---------------------------------------------------------------------------
# Super-resolution patch extraction
# ---------------------------------------------------------------------------
PATCH = {
    "lr_size"    : 32,        # low-resolution patch size (pixels)
    "hr_size"    : 128,       # high-resolution patch size (pixels)
    "scale"      : 4,         # upscaling factor
    "stride"     : 16,        # stride for sliding-window extraction (50% overlap)
    "augment"    : True,      # random flips + 90° rotations
    "val_split"  : 0.15,
    "test_split" : 0.10,
    "random_seed": 42,
}

# ---------------------------------------------------------------------------
# SRCNN hyperparameters
# ---------------------------------------------------------------------------
SRCNN = {
    "filters"    : [64, 32, 1],
    "kernels"    : [9,  1,  5],
    "epochs"     : 200,
    "batch_size" : 16,
    "lr"         : 1e-4,
    "loss"       : "mse",
    "checkpoint" : os.path.join(CHECKPOINTS_DIR, "srcnn_best.h5"),
}

# ---------------------------------------------------------------------------
# SRGAN hyperparameters
# ---------------------------------------------------------------------------
SRGAN = {
    "epochs"            : 200,
    "batch_size"        : 16,
    "lr_gen"            : 1e-4,
    "lr_disc"           : 1e-4,
    "lambda_content"    : 1e-3,    # weight for adversarial loss vs. content loss
    "vgg_layer"         : "block5_conv4",
    "pretrain_epochs"   : 10,      # pixel-loss warm-up before adversarial training
    "checkpoint_gen"    : os.path.join(CHECKPOINTS_DIR, "srgan_gen_best.h5"),
    "checkpoint_disc"   : os.path.join(CHECKPOINTS_DIR, "srgan_disc_best.h5"),
    "save_interval"     : 10,      # save checkpoint every N epochs
}

# ---------------------------------------------------------------------------
# Vegetation indices
# ---------------------------------------------------------------------------
VI = {
    "savi_L"     : 0.5,    # soil brightness correction factor
    "ndvi_range" : (-1.0, 1.0),
    "savi_range" : (-1.0, 1.0),
}

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
ANALYSIS = {
    "time_periods"  : ["2003-2008", "2009-2014", "2015-2020"],
    "alpha"         : 0.05,      # significance level
    "kfold_splits"  : 5,
    "morans_queen"  : True,      # Queen contiguity for spatial weights
    "kriging_nlags" : 20,
    "fourier_min_period_months": 6,
}

# ---------------------------------------------------------------------------
# Matplotlib / figure settings
# ---------------------------------------------------------------------------
FIGURE = {
    "dpi"      : 300,
    "format"   : "pdf",          # also saves .png
    "cmap_vi"  : "RdYlGn",
    "cmap_temp": "coolwarm",
    "cmap_rain": "Blues",
    "cmap_gws" : "BrBG",
    "font_size": 11,
}

# ---------------------------------------------------------------------------
# Auto-create all output directories on import
# ---------------------------------------------------------------------------
_dirs = [
    PROCESSED_DIR, PROC_LANDSAT, PROC_GRACE, PROC_PLANETSCOPE,
    PROC_CRUTS, PROC_PME,
    RESULTS_DIR, RES_BICUBIC, RES_SRCNN, RES_SRGAN,
    RES_INDICES, RES_ANALYSIS,
    FIGURES_DIR, CHECKPOINTS_DIR,
]
for _d in _dirs:
    os.makedirs(_d, exist_ok=True)
