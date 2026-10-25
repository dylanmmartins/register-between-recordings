import argparse
import os
import sys

import numpy as np

from regbrec import core as rc

IMAGE_CHOICES = {"mean": 0, "corr": 1, "maxproj": 2}


def read_points(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    mv = np.column_stack([d["moving_x"], d["moving_y"]])
    fx = np.column_stack([d["fixed_x"], d["fixed_y"]])
    return mv, fx


def main(argv=None):
    ap = argparse.ArgumentParser(prog="regbrec", description="Match suite2p ROIs between two recordings of the same field of view.")
    ap.add_argument("day1", nargs="?", help="day-1 (fixed) suite2p plane0 / suite2p / experiment folder")
    ap.add_argument("day2", nargs="?", help="day-2 (moving) folder")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--auto", action="store_true", help="estimate the affine automatically")
    g.add_argument("--points", help="control-point CSV (moving_x, moving_y, fixed_x, fixed_y)")
    g.add_argument("--transform", help="transform JSON saved by the GUI / this script")
    g.add_argument("--session", help="session JSON saved by the GUI (uses its transform and curated matches)")
    ap.add_argument("--auto-image", choices=IMAGE_CHOICES, default="mean", help="image used by --auto")
    ap.add_argument("--proc", action="store_true", help="use the processed image (e.g. meanImgE) for --auto")
    ap.add_argument("--no-refine", action="store_true", help="skip ROI-centroid refinement after --auto")
    ap.add_argument("--threshold", type=float, default=rc.DEFAULT_OVERLAP_THOLD,
                    help="overlap threshold, fraction of the day-1 ROI (default 0.6)")
    ap.add_argument("--out", help="output path (default: <day1 plane dir>/<day1>__<day2>_reg.npz)")
    ap.add_argument("--save-transform", help="also write the transform to this JSON file")
    a = ap.parse_args(argv)

    if a.session:
        reg = rc.Registration.load_session(a.session)
        print(f"loaded session with {len(reg.pairs)} curated matches (not re-detected)")
    else:
        if not (a.day1 and a.day2):
            ap.error("give DAY1 and DAY2 folders (or --session)")
        reg = rc.Registration(a.day1, a.day2)
        if a.auto:
            A, info = reg.auto_transform(mask=IMAGE_CHOICES[a.auto_image], proc=int(a.proc),
                                         refine=not a.no_refine)
            print(f"auto transform ({info['source']}): image corr {info['corr_before']:.3f} -> "
                  f"{info['corr_after']:.3f}", end="")
            if "n_centroid_pairs" in info:
                print(f"; {info['n_centroid_pairs']} centroid pairs, median residual "
                      f"{info['median_centroid_residual']:.2f} px", end="")
            print()
        elif a.points:
            mv, fx = read_points(a.points)
            reg.set_control_points(mv, fx)
            res = reg.control_point_residuals()
            print(f"affine from {len(mv)} control points, residual mean {res.mean():.2f} px, max {res.max():.2f} px")
        elif a.transform:
            reg.load_transform(a.transform)
        else:
            print("warning: no transform given; day-2 ROIs are compared in their own coordinates")
        reg.detect(a.threshold)

    if reg.A is not None:
        print("A (day 2 -> day 1, [x, y, 1]):")
        print(np.array2string(reg.A, precision=5, suppress_small=True))
    if a.save_transform and reg.A is not None:
        reg.save_transform(a.save_transform)

    res = reg.result()
    n1, n2 = len(reg.day1.iscell_raw), len(reg.day2.iscell_raw)
    nm = len(res["rois_idcs"])
    print(f"day 1: {n1} cells | day 2: {n2} cells | matched: {nm} "
          f"({100 * nm / max(n1, 1):.1f}% of day 1, {100 * nm / max(n2, 1):.1f}% of day 2)")
    print(f"day-1 only: {(res['status'] == rc.STATUS_DAY1_ONLY).sum()} | "
          f"day-2 only (new IDs {n1}..{len(res['unified_id']) - 1}): {(res['status'] == rc.STATUS_DAY2_ONLY).sum()}")
    if len(res["day2_lost_in_transform_raw"]):
        print(f"{len(res['day2_lost_in_transform_raw'])} day-2 ROIs had no pixels left after warping "
              f"(they are kept as day-2-only cells)")
    if nm:
        p = res["p_overlaps"]
        print(f"overlap of matches: median {np.median(p):.3f}, range {p.min():.3f}-{p.max():.3f}")

    out = a.out
    if out is None:
        out = os.path.join(reg.day1.plane_dir, f"{rc.session_label(reg.day1.plane_dir)}__"
                           f"{rc.session_label(reg.day2.plane_dir)}_reg.npz")
    npz, csvp = rc.save_result(res, out)
    print(f"saved {npz}\n      {csvp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
