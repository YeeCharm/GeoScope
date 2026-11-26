"""Standalone evaluation and schema validation toolkit for GeoScope-Bench."""

from .metrics import (
    PairEvaluation,
    RequestEvaluation,
    aggregate_results,
    evaluate_pair,
    evaluate_request,
    foreground_union,
    hungarian_match_iou,
    identity_check_passes,
    mask_iou,
)

__all__ = [
    "PairEvaluation",
    "RequestEvaluation",
    "aggregate_results",
    "evaluate_pair",
    "evaluate_request",
    "foreground_union",
    "hungarian_match_iou",
    "identity_check_passes",
    "mask_iou",
]
