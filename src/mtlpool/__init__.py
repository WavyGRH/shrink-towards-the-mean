"""Pooling baselines and partial pooling for regularised multi-task learning (School benchmark)."""
from .config import GRIDS, METHODS, RATIOS
from .data import load_school, make_split, split, standardise, unzip
from .estimators import (ESTIMATORS, independent_ridge, l21_mtl, mean_lowrank, mean_regularized,
                         pooled_ridge, trace_mtl)
from .evaluation import cv_select, metrics, paired_test
from .solvers import fista, prox_l21, prox_trace

__version__ = "1.0.0"
