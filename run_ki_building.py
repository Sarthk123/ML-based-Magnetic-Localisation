"""Run Mag-Match on the KI Building dataset (Jung, Oh, Myung 2015 / [27]).

The dataset records two consecutive laps of one floor of the building
(robot-to-robot registration scenario from Sec. IV-A-3 of the Mag-Match
paper).  The two laps are separated where the SLAM trajectory returns to
the origin around step ~1290.

Files:
  mag_log.txt        - idx, raw_x, raw_y, raw_z, cal_x, cal_y, cal_z
  robotSLAMPose.txt  - idx, x, y, z, roll, pitch, yaw
  robotOdomPose.txt  - idx, x, y, z, roll, pitch, yaw

The SLAM-corrected poses are used as measurement locations (per the paper:
"the SLAM-corrected points were used, as the odometry points were
significantly askew").

Outputs the position/rotation registration error against ground truth and
compares to Table I of the paper:  0.1784 m +/- 0.0836,  0.2688 deg +/- 0.1076.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np

from mag_match import (
    MagMatchConfig,
    extract_features,
    match_descriptors,
    modified_msac,
    rotation_angle_deg,
)


DATASET_PATH = ""


# ---------------------------------------------------------------------------
# 1. Data loading & lap splitting
# ---------------------------------------------------------------------------


def load_ki_building(folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (positions, mag_calibrated, slam_yaw) - one row per timestep.

    positions  : (N, 3) - x, y, z (z is identically 0 for this dataset)
    mag_cal    : (N, 3) - calibrated magnetometer readings (cols 4-6 of mag_log)
    yaw        : (N,)   - SLAM yaw, used to rotate body-frame mag readings
                          into the world frame.
    """
    mag = np.loadtxt(os.path.join(folder, "mag_log.txt"))
    slam = np.loadtxt(os.path.join(folder, "robotSLAMPose.txt"))
    assert mag.shape[0] == slam.shape[0], "step counts must match"
    pos = slam[:, 1:4].copy()
    mag_cal = mag[:, 4:7].copy()
    yaw = slam[:, 6].copy()
    return pos, mag_cal, yaw


def split_laps(pos: np.ndarray, return_dist: float = 1.0,
               min_first_lap_steps: int = 500) -> Tuple[slice, slice]:
    """Split the trajectory at the moment it last visits the origin region.

    The robot starts at (0, 0), explores, returns to ~(0, 0), then leaves
    again for the second lap.  We split at the *last* timestep of the
    returning cluster - the next step is the first one of lap 2.
    """
    d = np.linalg.norm(pos[:, :2], axis=1)
    near_start = np.where((d < return_dist) & (np.arange(len(d)) > min_first_lap_steps))[0]
    if near_start.size == 0:
        raise RuntimeError("could not detect a return to the origin")
    # Find the last step of the first contiguous return cluster.
    end_first = near_start[0]
    for k in near_start[1:]:
        if k == end_first + 1:
            end_first = k
        else:
            break
    return slice(0, end_first + 1), slice(end_first + 1, pos.shape[0])


# ---------------------------------------------------------------------------
# 2. Body-frame to world-frame for magnetometer readings.
# ---------------------------------------------------------------------------


def body_to_world(mag_body: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate body-frame magnetometer readings into the SLAM world frame.

    The robot moves on a flat floor (roll = pitch = 0) so only yaw matters:
        B_world = R_z(yaw) B_body.
    """
    B = np.empty_like(mag_body)
    c, s = np.cos(yaw), np.sin(yaw)
    B[:, 0] = c * mag_body[:, 0] - s * mag_body[:, 1]
    B[:, 1] = s * mag_body[:, 0] + c * mag_body[:, 1]
    B[:, 2] = mag_body[:, 2]
    return B


# ---------------------------------------------------------------------------
# 3. End-to-end registration test.
# ---------------------------------------------------------------------------


def apply_se3(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ R.T + t


def run_ki(synthetic_rotation_deg: float = 0.0,
           synthetic_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
           seed: int = 0,
           verbose: bool = True) -> dict:
    """Run map-to-map registration between the two laps.

    If `synthetic_rotation_deg` / `synthetic_translation` are non-zero, the
    target lap (lap 2) is additionally transformed by R_z(angle), t to
    produce a non-trivial registration problem.  The combined ground-truth
    base->target transform is then (R_synth, t_synth); the error is the
    deviation between Mag-Match's recovered T and that target.

    With both set to zero the test recovers the identity (this is the
    paper's setup: both laps share the SLAM-corrected world frame, so the
    algorithm should report a near-identity transform).
    """
    pos, mag_body, yaw = load_ki_building(DATASET_PATH)
    B_world = body_to_world(mag_body, yaw)

    # Centre and offset the field so the typical magnitude is in a friendly
    # range for the descriptor's component bins (the absolute geomagnetic DC
    # component is uninformative for matching).
    field_dc = B_world.mean(axis=0)
    B_world = B_world - field_dc

    lap1, lap2 = split_laps(pos)
    if verbose:
        print(f"Lap 1: {lap1.stop - lap1.start} samples (steps {lap1.start}..{lap1.stop - 1})")
        print(f"Lap 2: {lap2.stop - lap2.start} samples (steps {lap2.start}..{lap2.stop - 1})")

    X_base = pos[lap1].copy()
    B_base = B_world[lap1].copy()
    X_target = pos[lap2].copy()
    B_target = B_world[lap2].copy()

    # Optional synthetic transform applied to the target lap so we can
    # validate against a *known* ground truth even when the two laps are
    # natively in the same frame.
    angle = np.deg2rad(synthetic_rotation_deg)
    R_synth = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                        [np.sin(angle),  np.cos(angle), 0.0],
                        [0.0,            0.0,           1.0]])
    t_synth = np.asarray(synthetic_translation, dtype=float)
    if synthetic_rotation_deg != 0.0 or np.any(t_synth != 0.0):
        # Express target measurements in the *target* map frame: apply the
        # inverse so that x_world = R_synth x_target + t_synth.
        X_target = (X_target - t_synth) @ R_synth
        B_target = B_target @ R_synth   # vectors only rotate
    R_gt_b2t = R_synth.T
    t_gt_b2t = -R_synth.T @ t_synth

    # ---- Mag-Match config ----
    # KI Building is a ~45x56m floor with z = 0.  Because all measurements
    # share z, the GP is essentially 2D - we still use the 3D divergence-free
    # kernel, but we keep the inducing/query grid thin in z to stay close to
    # the data plane (one z-layer with a small thickness so the kernel is
    # well-conditioned).
    field_scale = float(np.quantile(np.linalg.norm(B_base, axis=1), 0.9))
    field_std = float(np.std(B_base))
    sigma_n = max(field_std * 0.05, 1.0)

    # Length scale 1.5 m chosen by parameter sweep on this dataset:
    # smaller scales over-fit and blow up NaN in the GP, while larger scales
    # produce too few discriminative keypoints (the corridor field is
    # quasi-constant over a few metres).
    ell = 1.5
    cfg = MagMatchConfig(
        sigma_f=field_scale,
        ell=ell,
        sigma_n=sigma_n,
        inducing_grid=int(np.ceil(50 / ell) + 2),
        query_grid=int(np.ceil(50 / ell) + 8),
        variance_quantile=0.4,
        min_doh_factor=2.0,
        support_radius=ell * 4.0,
        component_range=field_scale * 2.0,
        component_bin=field_scale * 0.2,
        nms_radius=ell * 1.0,
        max_keypoints=200,
    )

    # Build flat (single-z) inducing and query grids manually so we don't
    # waste capacity on a thick z dimension.
    margin = 2.0 * ell
    base_lo, base_hi = X_base.min(axis=0) - margin, X_base.max(axis=0) + margin
    target_lo, target_hi = X_target.min(axis=0) - margin, X_target.max(axis=0) + margin
    # z range: thin slab of one ell so the 3D divergence-free kernel is not
    # singular but the prior assumes minimal z-variation.
    base_lo[2], base_hi[2] = -ell * 0.5, ell * 0.5
    target_lo[2], target_hi[2] = -ell * 0.5, ell * 0.5

    # Each map queries on its own measurement footprint.  Lap 2 only retraces
    # part of the floor, so querying the target outside its trajectory just
    # wastes candidates on high-variance regions where no keypoints survive.
    base_q_lo = X_base.min(axis=0) + 0.5 * ell
    base_q_hi = X_base.max(axis=0) - 0.5 * ell
    base_q_lo[2], base_q_hi[2] = 0.0, 0.0
    target_q_lo = X_target.min(axis=0) + 0.5 * ell
    target_q_hi = X_target.max(axis=0) - 0.5 * ell
    target_q_lo[2], target_q_hi[2] = 0.0, 0.0

    if verbose:
        print(f"Field 90th-pctile norm = {field_scale:.1f}, std = {field_std:.1f}, "
              f"sigma_n = {sigma_n:.1f}")
        print(f"Base extents: {X_base.min(axis=0).round(2)} -> {X_base.max(axis=0).round(2)}")
        print(f"Target extents: {X_target.min(axis=0).round(2)} -> {X_target.max(axis=0).round(2)}")

    # The MagMatchConfig grid sizing is for an isotropic NxNxN volume; for
    # this 2D dataset we patch _make_grid to squash z to a single layer
    # (preserves the 3D divergence-free kernel without wasting capacity on
    # the empty z dimension).
    return _run_2d_pipeline(X_base, B_base, base_lo, base_hi, base_q_lo, base_q_hi,
                            X_target, B_target, target_lo, target_hi, target_q_lo, target_q_hi,
                            R_gt_b2t, t_gt_b2t, cfg, seed=seed, verbose=verbose)


def _run_2d_pipeline(X_base, B_base, base_lo, base_hi, base_q_lo, base_q_hi,
                     X_target, B_target, target_lo, target_hi, target_q_lo, target_q_hi,
                     R_gt_b2t, t_gt_b2t, cfg, seed=0, verbose=True):
    """Build features for both laps with single-z-layer grids and register."""
    # Patch the inducing/query grids so they are flat (single z layer).
    from mag_match import _make_grid as _mg, RecursiveSparseGP, DivergenceFreeKernel
    import mag_match as mm

    def flat_grid(lo, hi, n):
        # Force a single z layer at the centre of [lo[2], hi[2]].
        zc = 0.5 * (lo[2] + hi[2])
        ax = [np.linspace(lo[i], hi[i], n) for i in range(2)]
        XX, YY = np.meshgrid(*ax, indexing="ij")
        ZZ = np.full_like(XX, zc)
        return np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # Monkey-patch _make_grid for the duration of this call so flat (2D)
    # inducing/query grids are used.
    original_make_grid = mm._make_grid
    mm._make_grid = flat_grid
    try:
        t0 = time.time()
        kps_base, gp_base = extract_features(
            X_base, B_base, base_lo, base_hi, cfg,
            query_lo=base_q_lo, query_hi=base_q_hi)
        t1 = time.time()
        kps_target, gp_target = extract_features(
            X_target, B_target, target_lo, target_hi, cfg,
            query_lo=target_q_lo, query_hi=target_q_hi)
        t2 = time.time()
    finally:
        mm._make_grid = original_make_grid

    if verbose:
        print(f"Base map: {len(kps_base)} keypoints in {t1 - t0:.1f}s")
        print(f"Target map: {len(kps_target)} keypoints in {t2 - t1:.1f}s")

    # NN matcher with no Lowe ratio test - HOV descriptors over a corridor
    # building cluster more tightly than in the synthetic case so the ratio
    # rejects too many true matches.  We let MSAC do the geometric filtering.
    matches = match_descriptors(kps_base, kps_target, ratio=1.0)
    if verbose:
        print(f"Putative NN matches: {matches.shape[0]}")

    inlier_dist = 2.0       # 2 m positional tolerance (the keypoint grid spacing)
    cross_thresh = 0.4
    R_est, t_est, inlier_mask, fitness = modified_msac(
        kps_base, kps_target, matches,
        inlier_dist=inlier_dist, cross_thresh=cross_thresh,
        iters=20000, random_state=seed)

    rot_err = rotation_angle_deg(R_est @ R_gt_b2t.T)
    trans_err = float(np.linalg.norm(t_est - t_gt_b2t))
    n_inliers = int(inlier_mask.sum())

    if verbose:
        print()
        print("=== Registration result ===")
        print(f"  Rotation error    : {rot_err:.4f} deg")
        print(f"  Translation error : {trans_err:.4f} m")
        print(f"  Inliers           : {n_inliers}/{matches.shape[0]}")
        print(f"  Fitness (mean ||cross||) : {fitness:.4f}")
        print()
        print("Reference (paper Table I, KI Building map-to-map):")
        print("  0.1784 m  +/-  0.0836,    0.2688 deg  +/-  0.1076")

    return dict(
        R_est=R_est, t_est=t_est,
        R_gt_b2t=R_gt_b2t, t_gt_b2t=t_gt_b2t,
        rotation_error_deg=rot_err,
        translation_error_m=trans_err,
        n_inliers=n_inliers,
        n_matches=int(matches.shape[0]),
        n_keypoints_base=len(kps_base),
        n_keypoints_target=len(kps_target),
        fitness=fitness,
    )


def main():
    parser = argparse.ArgumentParser(description="Mag-Match on KI Building dataset")
    parser.add_argument("--rotation", type=float, default=0.0,
                        help="synthetic rotation (deg) applied to lap 2")
    parser.add_argument("--translation", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="synthetic translation [tx ty tz] applied to lap 2")
    parser.add_argument("--trials", type=int, default=1,
                        help="number of MSAC random trials (paper Table I uses repeated MSAC iterations)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print(" Mag-Match - KI Building dataset")
    print("=" * 60)

    if args.trials <= 1:
        run_ki(synthetic_rotation_deg=args.rotation,
               synthetic_translation=tuple(args.translation),
               seed=args.seed)
        return

    rot_errors, trans_errors = [], []
    n_inliers, n_kps = [], []
    for k in range(args.trials):
        print(f"\n--- trial {k + 1}/{args.trials} (seed={args.seed + k}) ---")
        out = run_ki(synthetic_rotation_deg=args.rotation,
                     synthetic_translation=tuple(args.translation),
                     seed=args.seed + k, verbose=True)
        if np.isfinite(out["fitness"]):
            rot_errors.append(out["rotation_error_deg"])
            trans_errors.append(out["translation_error_m"])
            n_inliers.append(out["n_inliers"])
            n_kps.append(0.5 * (out["n_keypoints_base"] + out["n_keypoints_target"]))
        else:
            print("  -> registration failed")

    if rot_errors:
        rot = np.array(rot_errors)
        tr = np.array(trans_errors)
        print("\n" + "=" * 60)
        print(f"Summary over {len(rot)} successful trials "
              f"({args.trials - len(rot)} failures)")
        print(f"  Translation RMSE = {np.sqrt(np.mean(tr ** 2)):.4f} m "
              f"(median {np.median(tr):.4f}, std {tr.std():.4f})")
        print(f"  Rotation    RMSE = {np.sqrt(np.mean(rot ** 2)):.4f} deg "
              f"(median {np.median(rot):.4f}, std {rot.std():.4f})")
        print(f"  Avg keypoints/map: {np.mean(n_kps):.1f}, "
              f"avg inliers: {np.mean(n_inliers):.1f}")
        print()
        print("Paper Table I (KI Building map-to-map):")
        print("  Translation: 0.1784 m  +/-  0.0836")
        print("  Rotation:    0.2688 deg +/- 0.1076")


if __name__ == "__main__":
    main()
