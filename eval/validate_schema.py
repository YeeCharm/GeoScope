"""Schema and programmatic integrity auditor for GeoScope-Bench manifests.

Enforces the six programmatic constraints in Appendix A.2 when mask files are supplied,
and validates structural schema integrity on manifest-only checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .metrics import foreground_union

VALID_TRANSFORMATIONS = {"identity", "extent", "membership", "partition", "paraphrase"}


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return arr > 0


def _validate_request_structure(
    req: Dict,
    label: str,
    errors: List[str],
) -> None:
    for field in ("request_id", "instruction", "targets", "groups"):
        if field not in req:
            errors.append(f"{label}: missing required field '{field}'")
    if errors:
        return

    targets = req["targets"]
    groups = req["groups"]
    if not isinstance(targets, list) or not targets:
        errors.append(f"{label}: 'targets' must be a non-empty list")
        return
    if not isinstance(groups, list) or not groups:
        errors.append(f"{label}: 'groups' must be a non-empty list")
        return

    target_ids = set()
    for idx, t in enumerate(targets):
        tid = t.get("target_id")
        if not isinstance(tid, int) or tid < 1:
            errors.append(f"{label}.targets[{idx}]: invalid target_id {tid!r}")
        elif tid in target_ids:
            errors.append(f"{label}: duplicate target_id {tid}")
        else:
            target_ids.add(tid)
        if not t.get("description"):
            errors.append(f"{label}.targets[{idx}]: empty description")

    for g_idx, g in enumerate(groups):
        tids = g.get("target_ids")
        if not isinstance(tids, list) or not tids:
            errors.append(f"{label}.groups[{g_idx}]: 'target_ids' must be non-empty")
            continue
        for tid in tids:
            if tid not in target_ids:
                errors.append(
                    f"{label}.groups[{g_idx}]: references unknown target_id {tid}"
                )


def _assemble_group_masks(
    req: Dict,
    mask_root: Path,
    expected_shape: Tuple[int, int],
    label: str,
    errors: List[str],
) -> List[np.ndarray]:
    target_masks: Dict[int, np.ndarray] = {}
    for t in req["targets"]:
        mpath = mask_root / t["mask_path"]
        if not mpath.exists():
            errors.append(f"{label}: missing mask file {mpath}")
            continue
        bm = _load_mask(mpath)
        if bm.shape != expected_shape:
            errors.append(
                f"{label}: mask {mpath} shape {bm.shape} != image_size {expected_shape}"
            )
            continue
        if not bm.any():
            errors.append(f"{label}: target mask {mpath} is empty")
        target_masks[int(t["target_id"])] = bm

    group_masks: List[np.ndarray] = []
    for g in req["groups"]:
        members = [target_masks[tid] for tid in g["target_ids"] if tid in target_masks]
        if len(members) != len(g["target_ids"]):
            continue
        gm = foreground_union(members, expected_shape)
        if not gm.any():
            errors.append(f"{label}: assembled output group {g.get('group_id')} is empty")
        group_masks.append(gm)
    return group_masks


def validate_manifest(
    manifest_path: Path,
    mask_root: Path | None = None,
    check_masks: bool = False,
) -> Tuple[int, List[str]]:
    errors: List[str] = []
    pair_ids = set()
    count = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            count += 1
            rec = json.loads(line)
            pid = rec.get("pair_id", f"line:{lineno}")
            if pid in pair_ids:
                errors.append(f"{pid}: duplicate pair_id")
            pair_ids.add(pid)

            t_type = str(rec.get("transformation", "")).lower()
            if t_type not in VALID_TRANSFORMATIONS:
                errors.append(f"{pid}: invalid transformation {t_type!r}")

            img_size = rec.get("image_size")
            if (
                not isinstance(img_size, list)
                or len(img_size) != 2
                or any(not isinstance(v, int) or v <= 0 for v in img_size)
            ):
                errors.append(f"{pid}: 'image_size' must be [height, width] of positive ints")
                continue

            for key in ("request_a", "request_b"):
                if key not in rec:
                    errors.append(f"{pid}: missing '{key}'")
                else:
                    _validate_request_structure(rec[key], f"{pid}.{key}", errors)

            if check_masks and mask_root is not None and "request_a" in rec and "request_b" in rec:
                shape = (int(img_size[0]), int(img_size[1]))
                ga = _assemble_group_masks(rec["request_a"], mask_root, shape, f"{pid}.a", errors)
                gb = _assemble_group_masks(rec["request_b"], mask_root, shape, f"{pid}.b", errors)
                if len(ga) == len(rec["request_a"]["groups"]) and len(gb) == len(rec["request_b"]["groups"]):
                    ua = foreground_union(ga, shape)
                    ub = foreground_union(gb, shape)
                    if t_type == "identity" and np.logical_and(ua, ub).any():
                        errors.append(f"{pid}: Identity pair targets are not disjoint")
                    elif t_type == "partition" and not np.array_equal(ua, ub):
                        errors.append(f"{pid}: Partition pair foreground unions are not pixel-identical")
                    elif t_type == "extent":
                        spec = rec.get("relation_spec", {})
                        whole = ua if spec.get("whole_request", "a") == "a" else ub
                        part = ub if spec.get("whole_request", "a") == "a" else ua
                        if not (np.logical_and(part, whole).sum() == part.sum() and part.sum() < whole.sum()):
                            errors.append(f"{pid}: Extent part mask is not a proper subset of whole mask")

    return count, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate GeoScope-Bench JSONL manifests.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to benchmark JSONL manifest.")
    parser.add_argument("--mask-root", type=Path, default=None, help="Root directory for reference mask files.")
    parser.add_argument("--check-masks", action="store_true", help="Load masks and enforce all 6 Appendix A.2 checks.")
    args = parser.parse_args(argv)

    count, errors = validate_manifest(
        args.manifest,
        mask_root=args.mask_root or args.manifest.parent,
        check_masks=args.check_masks,
    )
    if errors:
        print(f"[FAIL] {args.manifest} ({count} pairs, {len(errors)} errors):")
        for err in errors[:30]:
            print(f"  - {err}")
        return 1
    mode = "schema + mask integrity" if args.check_masks else "schema"
    print(f"[OK] {args.manifest}: {count} pairs passed {mode} validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
