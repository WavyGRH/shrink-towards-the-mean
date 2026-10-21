"""Experimental protocol used in the paper. Changing any value here changes the results."""
import numpy as np

RATIOS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.60, 0.75)  # training fraction per school
N_SPLITS = 10                                       # random splits per training fraction
K_FOLDS = 5                                         # cross-validation folds inside the training set
SPLIT_SEED_BASE = 10_000                            # split s uses default_rng(SPLIT_SEED_BASE + s)
CV_SEED_BASE = 20_000                               # its CV folds use default_rng(CV_SEED_BASE + s)
CONT_COLS = (3, 4)                                  # the two numeric School features, standardised
FISTA_ITERS = 500                                   # fixed proximal-gradient budget
MU_B = 1e-6                                         # tiny ridge on the shared mean (partial pooling)

METHODS = ("pooled ridge", "independent ridge", "l2,1 MTL", "trace-norm MTL",
           "mean-reg MTL", "mean+low-rank MTL")

GRIDS = {
    "pooled ridge": np.logspace(-6, 2, 17),
    "independent ridge": np.logspace(-6, 2, 17),
    "l2,1 MTL": np.logspace(-1.5, 2.5, 12),
    "trace-norm MTL": np.logspace(-1.5, 2.5, 12),
    "mean-reg MTL": np.logspace(-4, 3, 15),
    "mean+low-rank MTL": np.logspace(-2.5, 3.5, 15),
}

SCHOOL_URL = "https://raw.githubusercontent.com/jiayuzhou/MALSAR/master/data/school.mat"
SCHOOL_SHA256 = "57fb3238fdd03a9aa17276e2ce18d9bbb5d128c3261e794fe6233b58b6909068"
