"""Official metric implementations for GeoScope-Bench.

Implements the evaluation protocol described in Section 3.3 and Appendix B of:
"GeoScope: What to Mask and What to Merge in Remote Sensing Reasoning Segmentation".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ImportError:  # pragma: no cover - pure-Python fallback used when scipy is absent
    _scipy_linear_sum_assignment = None


def to_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a 2D array to a boolean binary mask."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape {arr.shape}.")
    return arr.astype(bool)


def mask_iou(pred: np.ndarray, target: np.ndarray) -> Tuple[int, int, float]:
    """Return (intersection, union, iou) for two 2D binary masks."""
    p = to_bool_mask(pred)
    t = to_bool_mask(target)
    if p.shape != t.shape:
        raise ValueError(f"Mask shape mismatch: pred {p.shape} vs target {t.shape}.")
    inter = int(np.logical_and(p, t).sum())
    union = int(np.logical_or(p, t).sum())
    iou = float(inter / union) if union > 0 else 1.0
    return inter, union, iou


def foreground_union(masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Compute the pixelwise binary union of a list of masks."""
    out = np.zeros(shape, dtype=bool)
    for m in masks:
        bm = to_bool_mask(m)
        if bm.shape != shape:
            raise ValueError(f"Mask shape {bm.shape} does not match expected {shape}.")
        out |= bm
    return out


def _hungarian_minimize_pure(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Hungarian (Kuhn-Munkres) solver for rectangular cost matrices."""
    mat = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = mat.shape
    transposed = False
    if n_rows > n_cols:
        mat = mat.T
        n_rows, n_cols = n_cols, n_rows
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            row_cost = mat[i0 - 1]
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = row_cost[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = np.empty(n_rows, dtype=np.int64)
    col_ind = np.empty(n_rows, dtype=np.int64)
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            row_ind[p[j] - 1] = p[j] - 1
            col_ind[p[j] - 1] = j - 1

    if transposed:
        order = np.argsort(col_ind)
        return col_ind[order], row_ind[order]
    return row_ind, col_ind


def hungarian_match_iou(
    pred_masks: Sequence[np.ndarray],
    ref_masks: Sequence[np.ndarray],
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """One-to-one maximum-sum-IoU Hungarian assignment.

    Returns:
        matches: list of (pred_idx, ref_idx, iou)
        iou_matrix: shape (len(pred_masks), len(ref_masks))
    """
    m = len(pred_masks)
    n = len(ref_masks)
    iou_matrix = np.zeros((m, n), dtype=np.float64)
    if m == 0 or n == 0:
        return [], iou_matrix

    for i, pm in enumerate(pred_masks):
        for j, rm in enumerate(ref_masks):
            _, _, iou_matrix[i, j] = mask_iou(pm, rm)

    cost = -iou_matrix
    if _scipy_linear_sum_assignment is not None:
        row_ind, col_ind = _scipy_linear_sum_assignment(cost)
    else:
        row_ind, col_ind = _hungarian_minimize_pure(cost)

    matches = [(int(r), int(c), float(iou_matrix[r, c])) for r, c in zip(row_ind, col_ind)]
    return matches, iou_matrix


def identity_check_passes(
    pred_mask: np.ndarray,
    intended_mask: np.ndarray,
    competing_masks: Sequence[np.ndarray],
    margin_cap: float = 0.05,
) -> bool:
    """Check Appendix B Eq. (S1) target identity margin against competing references."""
    _, _, iou_pm = mask_iou(pred_mask, intended_mask)
    for comp in competing_masks:
        _, _, iou_pr = mask_iou(pred_mask, comp)
        _, _, iou_mr = mask_iou(intended_mask, comp)
        required_margin = min(margin_cap, 0.5 * (1.0 - iou_mr))
        if (iou_pm - iou_pr) + 1e-12 < required_margin:
            return False
    return True


@dataclass
class RequestEvaluation:
    request_id: str
    union_inter: int
    union_union: int
    union_iou: float
    group_inter: int
    group_union: int
    group_giou: float
    exact_count: bool
    min_matched_iou: float
    identity_ok: bool
    passed: bool
    # Maps ref_idx -> pred_idx for matched groups
    ref_to_pred: Dict[int, int]


@dataclass
class PairEvaluation:
    pair_id: str
    transformation: str
    req_a: RequestEvaluation
    req_b: RequestEvaluation
    relation_passed: bool
    pair_success: bool


def evaluate_request(
    request_id: str,
    pred_masks: Sequence[np.ndarray],
    ref_masks: Sequence[np.ndarray],
    competing_masks: Optional[Sequence[np.ndarray]] = None,
    iou_threshold: float = 0.5,
    margin_cap: float = 0.05,
) -> RequestEvaluation:
    """Evaluate predictions for a single request against its reference output groups."""
    if not ref_masks:
        raise ValueError(f"Request {request_id} has no reference masks.")
    shape = to_bool_mask(ref_masks[0]).shape
    pred_bool = [to_bool_mask(m) for m in pred_masks]
    ref_bool = [to_bool_mask(m) for m in ref_masks]

    pred_u = foreground_union(pred_bool, shape)
    ref_u = foreground_union(ref_bool, shape)
    u_inter, u_union, u_iou = mask_iou(pred_u, ref_u)

    matches, _ = hungarian_match_iou(pred_bool, ref_bool)
    m = len(pred_bool)
    n = len(ref_bool)
    denom = max(n, m, 1)
    group_giou = sum(iou for _, _, iou in matches) / denom

    matched_preds = set()
    matched_refs = set()
    ref_to_pred: Dict[int, int] = {}
    g_inter = 0
    g_union = 0
    for pi, ri, _ in matches:
        matched_preds.add(pi)
        matched_refs.add(ri)
        ref_to_pred[ri] = pi
        inter, union, _ = mask_iou(pred_bool[pi], ref_bool[ri])
        g_inter += inter
        g_union += union

    # Unmatched predicted or reference masks contribute their area to the pooled union.
    for pi, pm in enumerate(pred_bool):
        if pi not in matched_preds:
            g_union += int(pm.sum())
    for ri, rm in enumerate(ref_bool):
        if ri not in matched_refs:
            g_union += int(rm.sum())

    exact_count = m == n
    min_matched_iou = min((iou for _, _, iou in matches), default=0.0) if exact_count else 0.0

    identity_ok = True
    if exact_count:
        extra_comp = [to_bool_mask(c) for c in (competing_masks or [])]
        for pi, ri, _ in matches:
            other_refs = [ref_bool[j] for j in range(n) if j != ri] + extra_comp
            if other_refs and not identity_check_passes(
                pred_bool[pi], ref_bool[ri], other_refs, margin_cap=margin_cap
            ):
                identity_ok = False
                break
    else:
        identity_ok = False

    passed = exact_count and (min_matched_iou >= iou_threshold) and identity_ok
    return RequestEvaluation(
        request_id=request_id,
        union_inter=u_inter,
        union_union=u_union,
        union_iou=u_iou,
        group_inter=g_inter,
        group_union=g_union,
        group_giou=group_giou,
        exact_count=exact_count,
        min_matched_iou=min_matched_iou,
        identity_ok=identity_ok,
        passed=passed,
        ref_to_pred=ref_to_pred,
    )


def evaluate_pair(
    pair_id: str,
    transformation: str,
    request_a_id: str,
    request_b_id: str,
    pred_a: Sequence[np.ndarray],
    ref_a: Sequence[np.ndarray],
    pred_b: Sequence[np.ndarray],
    ref_b: Sequence[np.ndarray],
    relation_spec: Optional[Dict] = None,
    iou_threshold: float = 0.5,
    relation_iou_threshold: float = 0.7,
    extent_containment_threshold: float = 0.9,
    margin_cap: float = 0.05,
) -> PairEvaluation:
    """Evaluate a same-image paired request and its transformation-specific relation C_i."""
    spec = relation_spec or {}
    t_type = transformation.lower()

    # For Identity pairs, each endpoint's reference acts as a competing target for the other.
    comp_a = ref_b if t_type == "identity" else None
    comp_b = ref_a if t_type == "identity" else None

    eval_a = evaluate_request(
        request_a_id, pred_a, ref_a, competing_masks=comp_a,
        iou_threshold=iou_threshold, margin_cap=margin_cap,
    )
    eval_b = evaluate_request(
        request_b_id, pred_b, ref_b, competing_masks=comp_b,
        iou_threshold=iou_threshold, margin_cap=margin_cap,
    )

    pred_a_bool = [to_bool_mask(m) for m in pred_a]
    pred_b_bool = [to_bool_mask(m) for m in pred_b]
    shape = to_bool_mask(ref_a[0]).shape

    relation_passed = False
    if eval_a.passed and eval_b.passed:
        if t_type == "identity":
            # Mutual identity checks already enforce switching targets; also verify disjointness.
            ua = foreground_union(pred_a_bool, shape)
            ub = foreground_union(pred_b_bool, shape)
            _, _, cross_iou = mask_iou(ua, ub)
            relation_passed = cross_iou < 0.5
        elif t_type == "extent":
            whole_side = spec.get("whole_request", "a")
            whole_u = foreground_union(pred_a_bool if whole_side == "a" else pred_b_bool, shape)
            part_u = foreground_union(pred_b_bool if whole_side == "a" else pred_a_bool, shape)
            whole_area = int(whole_u.sum())
            part_area = int(part_u.sum())
            if part_area > 0 and part_area < whole_area:
                containment = int(np.logical_and(part_u, whole_u).sum()) / part_area
                min_cont = float(spec.get("min_containment", extent_containment_threshold))
                relation_passed = containment >= min_cont
        elif t_type == "membership":
            # Preserved target groups must align across endpoints with IoU >= 0.70.
            preserved_pairs = spec.get("preserved_group_pairs")
            if preserved_pairs:
                ok = True
                for ref_idx_a_1b, ref_idx_b_1b in preserved_pairs:
                    pa_idx = eval_a.ref_to_pred.get(int(ref_idx_a_1b) - 1)
                    pb_idx = eval_b.ref_to_pred.get(int(ref_idx_b_1b) - 1)
                    if pa_idx is None or pb_idx is None:
                        ok = False
                        break
                    _, _, cross_iou = mask_iou(pred_a_bool[pa_idx], pred_b_bool[pb_idx])
                    if cross_iou < relation_iou_threshold:
                        ok = False
                        break
                relation_passed = ok
            else:
                # Infer preserved targets from identical reference masks across endpoints.
                ok = True
                found_preserved = False
                for ra_i, rm_a in enumerate(ref_a):
                    for rb_i, rm_b in enumerate(ref_b):
                        _, _, r_iou = mask_iou(rm_a, rm_b)
                        if r_iou >= 0.99:
                            found_preserved = True
                            pa_idx = eval_a.ref_to_pred[ra_i]
                            pb_idx = eval_b.ref_to_pred[rb_i]
                            _, _, cross_iou = mask_iou(pred_a_bool[pa_idx], pred_b_bool[pb_idx])
                            if cross_iou < relation_iou_threshold:
                                ok = False
                relation_passed = ok and found_preserved
        elif t_type == "partition":
            ua = foreground_union(pred_a_bool, shape)
            ub = foreground_union(pred_b_bool, shape)
            _, _, union_cross_iou = mask_iou(ua, ub)
            relation_passed = union_cross_iou >= relation_iou_threshold
        elif t_type == "paraphrase":
            if len(pred_a_bool) == len(pred_b_bool) and len(pred_a_bool) > 0:
                matches, _ = hungarian_match_iou(pred_a_bool, pred_b_bool)
                relation_passed = all(iou >= relation_iou_threshold for _, _, iou in matches)
        else:
            raise ValueError(f"Unknown transformation type: {transformation!r}")

    return PairEvaluation(
        pair_id=pair_id,
        transformation=t_type,
        req_a=eval_a,
        req_b=eval_b,
        relation_passed=relation_passed,
        pair_success=eval_a.passed and eval_b.passed and relation_passed,
    )


def _summarize_requests(reqs: Sequence[RequestEvaluation]) -> Dict[str, float]:
    if not reqs:
        return {
            "num_requests": 0,
            "union_giou": 0.0,
            "union_ciou": 0.0,
            "group_giou": 0.0,
            "group_ciou": 0.0,
            "request_pass": 0.0,
        }
    u_giou = 100.0 * sum(r.union_iou for r in reqs) / len(reqs)
    u_inter = sum(r.union_inter for r in reqs)
    u_union = sum(r.union_union for r in reqs)
    u_ciou = 100.0 * (u_inter / u_union) if u_union > 0 else 0.0

    g_giou = 100.0 * sum(r.group_giou for r in reqs) / len(reqs)
    g_inter = sum(r.group_inter for r in reqs)
    g_union = sum(r.group_union for r in reqs)
    g_ciou = 100.0 * (g_inter / g_union) if g_union > 0 else 0.0

    req_pass = 100.0 * sum(1 for r in reqs if r.passed) / len(reqs)
    return {
        "num_requests": len(reqs),
        "union_giou": round(u_giou, 2),
        "union_ciou": round(u_ciou, 2),
        "group_giou": round(g_giou, 2),
        "group_ciou": round(g_ciou, 2),
        "request_pass": round(req_pass, 2),
    }


def aggregate_results(pair_results: Sequence[PairEvaluation]) -> Dict[str, object]:
    """Aggregate overall and per-transformation GeoScope-Bench scores (%)."""
    unique_reqs: Dict[str, RequestEvaluation] = {}
    by_type: Dict[str, List[PairEvaluation]] = {}
    for pr in pair_results:
        unique_reqs.setdefault(pr.req_a.request_id, pr.req_a)
        unique_reqs.setdefault(pr.req_b.request_id, pr.req_b)
        by_type.setdefault(pr.transformation, []).append(pr)

    overall = _summarize_requests(list(unique_reqs.values()))
    overall["num_pairs"] = len(pair_results)
    overall["pair_success"] = (
        round(100.0 * sum(1 for p in pair_results if p.pair_success) / len(pair_results), 2)
        if pair_results
        else 0.0
    )

    per_transformation: Dict[str, Dict[str, float]] = {}
    for t_name in sorted(by_type):
        subset = by_type[t_name]
        t_reqs: Dict[str, RequestEvaluation] = {}
        for pr in subset:
            t_reqs.setdefault(pr.req_a.request_id, pr.req_a)
            t_reqs.setdefault(pr.req_b.request_id, pr.req_b)
        stats = _summarize_requests(list(t_reqs.values()))
        stats["num_pairs"] = len(subset)
        stats["pair_success"] = round(
            100.0 * sum(1 for p in subset if p.pair_success) / len(subset), 2
        )
        per_transformation[t_name] = stats

    return {
        "overall": overall,
        "by_transformation": per_transformation,
    }
