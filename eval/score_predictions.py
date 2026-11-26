"""CLI scorer for GeoScope-Bench predictions (plus built-in synthetic self-test)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .metrics import aggregate_results, evaluate_pair, foreground_union


def _load_mask_item(item, root: Path, shape: Tuple[int, int]) -> np.ndarray:
    """Load a binary mask from a PNG path, uncompressed RLE dict, or 2D list."""
    if isinstance(item, str):
        with Image.open(root / item) as im:
            arr = np.asarray(im.convert("L")) > 0
        if arr.shape != shape:
            raise ValueError(f"Mask {item} has shape {arr.shape}, expected {shape}.")
        return arr
    if isinstance(item, dict) and "counts" in item and "size" in item:
        h, w = int(item["size"][0]), int(item["size"][1])
        flat = np.zeros(h * w, dtype=bool)
        pos = 0
        val = False
        for run in item["counts"]:
            if val:
                flat[pos : pos + int(run)] = True
            pos += int(run)
            val = not val
        return flat.reshape((h, w), order="F")
    arr = np.asarray(item, dtype=bool)
    if arr.shape != shape:
        raise ValueError(f"Inline mask has shape {arr.shape}, expected {shape}.")
    return arr


def _load_reference_groups(req: Dict, root: Path, shape: Tuple[int, int]) -> List[np.ndarray]:
    target_map = {
        int(t["target_id"]): _load_mask_item(t["mask_path"], root, shape)
        for t in req["targets"]
    }
    return [
        foreground_union([target_map[int(tid)] for tid in g["target_ids"]], shape)
        for g in req["groups"]
    ]


def run_self_test() -> Dict[str, object]:
    """Run an end-to-end verification over all 5 transformations using synthetic masks."""
    shape = (32, 32)
    box_a = np.zeros(shape, dtype=bool)
    box_a[2:12, 2:12] = True
    box_a_part = np.zeros(shape, dtype=bool)
    box_a_part[4:9, 4:9] = True
    box_b = np.zeros(shape, dtype=bool)
    box_b[18:28, 18:28] = True
    union_ab = box_a | box_b

    # 1. Identity (pass)
    r1 = evaluate_pair("self-id", "identity", "id-a", "id-b", [box_a], [box_a], [box_b], [box_b])
    assert r1.pair_success, "Identity passing case failed"

    # 2. Extent (pass)
    r2 = evaluate_pair(
        "self-ext", "extent", "ext-a", "ext-b",
        [box_a], [box_a], [box_a_part], [box_a_part],
        relation_spec={"whole_request": "a", "min_containment": 0.9},
    )
    assert r2.pair_success, "Extent passing case failed"

    # 3. Membership (pass) + duplicate-localization failure check
    r3_pass = evaluate_pair(
        "self-mem-ok", "membership", "mem-a", "mem-b",
        [box_a], [box_a], [box_a, box_b], [box_a, box_b],
        relation_spec={"preserved_group_pairs": [[1, 1]]},
    )
    assert r3_pass.pair_success, "Membership passing case failed"

    r3_dup = evaluate_pair(
        "self-mem-dup", "membership", "mem2-a", "mem2-b",
        [box_a], [box_a], [box_a, box_a], [box_a, box_b],
        relation_spec={"preserved_group_pairs": [[1, 1]]},
    )
    assert not r3_dup.pair_success, "Duplicate localization should fail Membership Pair Success"

    # 4. Partition (pass) + wrong-grouping check
    r4_pass = evaluate_pair(
        "self-part-ok", "partition", "part-a", "part-b",
        [union_ab], [union_ab], [box_a, box_b], [box_a, box_b],
    )
    assert r4_pass.pair_success, "Partition passing case failed"

    r4_wrong = evaluate_pair(
        "self-part-bad", "partition", "part2-a", "part2-b",
        [union_ab], [union_ab], [union_ab], [box_a, box_b],
    )
    assert r4_wrong.req_b.union_iou == 1.0 and r4_wrong.req_b.group_giou < 1.0
    assert not r4_wrong.pair_success, "Wrong grouping should fail Partition Pair Success"

    # 5. Paraphrase (pass)
    r5 = evaluate_pair(
        "self-para", "paraphrase", "para-a", "para-b",
        [box_a], [box_a], [box_a], [box_a],
    )
    assert r5.pair_success, "Paraphrase passing case failed"

    summary = aggregate_results([r1, r2, r3_pass, r4_pass, r5])
    assert summary["overall"]["pair_success"] == 100.0
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score GeoScope-Bench predictions.")
    parser.add_argument("--manifest", type=Path, help="Path to GeoScope-Bench JSONL manifest.")
    parser.add_argument("--predictions", type=Path, help="Path to predictions JSONL keyed by request_id.")
    parser.add_argument("--mask-root", type=Path, default=None, help="Root directory for reference masks.")
    parser.add_argument("--pred-root", type=Path, default=None, help="Root directory for predicted masks.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Group IoU cutoff (default: 0.5).")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save JSON summary.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in synthetic metric verification.")
    args = parser.parse_args(argv)

    if args.self_test:
        summary = run_self_test()
        print("[OK] Synthetic self-test passed across all 5 transformations.")
        print(json.dumps(summary, indent=2))
        return 0

    if not args.manifest or not args.predictions:
        parser.error("--manifest and --predictions are required unless --self-test is set.")

    ref_root = args.mask_root or args.manifest.parent
    pred_root = args.pred_root or args.predictions.parent

    pred_by_req: Dict[str, List] = {}
    with args.predictions.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            pred_by_req[str(item["request_id"])] = item.get("masks", [])

    pair_results = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            shape = (int(rec["image_size"][0]), int(rec["image_size"][1]))
            req_a = rec["request_a"]
            req_b = rec["request_b"]
            ref_a = _load_reference_groups(req_a, ref_root, shape)
            ref_b = _load_reference_groups(req_b, ref_root, shape)
            pred_a = [
                _load_mask_item(m, pred_root, shape)
                for m in pred_by_req.get(req_a["request_id"], [])
            ]
            pred_b = [
                _load_mask_item(m, pred_root, shape)
                for m in pred_by_req.get(req_b["request_id"], [])
            ]
            pair_results.append(
                evaluate_pair(
                    pair_id=rec["pair_id"],
                    transformation=rec["transformation"],
                    request_a_id=req_a["request_id"],
                    request_b_id=req_b["request_id"],
                    pred_a=pred_a,
                    ref_a=ref_a,
                    pred_b=pred_b,
                    ref_b=ref_b,
                    relation_spec=rec.get("relation_spec"),
                    iou_threshold=args.iou_threshold,
                )
            )

    summary = aggregate_results(pair_results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
