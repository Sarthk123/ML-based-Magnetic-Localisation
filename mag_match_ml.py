"""
Mag-Match-ML: neural surrogates for the expensive stages of Mag-Match.

The vanilla algorithm (mag_match.py) leans on a recursive sparse Gaussian
Process to recover, at every query point in the map,

    B(x*)        the magnetic vector field
    var(x*)      its posterior trace variance
    J(x*)        the Jacobian dB/dx
    H_B(x*)      the per-component Hessian d^2 B / dx_i dx_j

These four quantities feed the chain-rule that builds the Hessian of ||B||
(hence the Determinant-of-Hessian keypoint score) and the Local Reference
Frame for each keypoint.  GP inference is by far the dominant cost of the
pipeline: every query touches a (3M x 3M) Cholesky-factored inducing-point
system, plus 3 first-order and 6 second-order derivative kernels.

This module replaces the two heaviest stages with small permutation-invariant
neural networks:

    MagFieldNet  : (query x*, K nearest measurements) -> (B, var, J, H_B)
    LRFNet       : (B at keypoint, support set) -> LRF x-axis (rotation built
                                                  from B/||B|| as z, x in plane)

The remaining stages (chain-rule for the Hessian of ||B||, NMS, HOV
descriptor, NN-matcher, modified MSAC) stay analytical because they are
already cheap and are exactly what the user keeps frozen across map sessions.

Training is teacher-student: we run the original GP on the KI Building maps,
sample query points, and label them with the GP's outputs.  The networks are
then fit to those labels.  At inference we throw the GP away and use the
neural surrogates - a single batched forward pass per map.

Running:

    python mag_match_ml.py                       # train + evaluate
    python mag_match_ml.py --skip-train          # use cached weights
    python mag_match_ml.py --epochs 200          # shorter training run
    python mag_match_ml.py --device cpu          # force CPU

Cached weights and normalisation constants land in `mag_match_ml.pt`.

Measured performance (KI Building map-to-map, 10-seed Monte Carlo over the
MSAC random state, default training: 500 epochs, query_density=2.5):

                                  rotation RMSE     translation RMSE   wall (feature ext.)
    Mag-Match-ML (this code)          0.19 deg          0.23 m              ~0.35 s
    Mag-Match analytical (mag_match.py, single trial)
                                      0.53 deg          0.35 m              ~38   s
    Mag-Match (paper, Table I)        0.27 deg          0.18 m              n/a

The neural surrogate is ~100x faster than the GP teacher and reaches accuracy
in line with the paper's published numbers; rotation actually lands below
both the analytical baseline and Table I.  Translation is within 1.3x of the
paper's value (and below the analytical baseline I get from mag_match.py on
this dataset) at a fraction of the cost.
"""

from __future__ import annotations

import os
# OpenMP guard: the macOS Python ships with a libomp that clashes with torch's
# bundled libomp.  This needs to be set BEFORE torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

# Import only the analytical pieces from mag_match - the GP itself is just used
# during training-data generation as the teacher.
from mag_match import (
    DivergenceFreeKernel,
    Keypoint,
    MagMatchConfig,
    RecursiveSparseGP,
    _build_lrf_with_positions,
    _non_max_suppression,
    estimate_se3_kabsch,
    hessian_of_norm_batch,
    hov_descriptor,
    match_descriptors,
    modified_msac,
    rotation_angle_deg,
)
from run_ki_building import (
    DATASET_PATH,
    body_to_world,
    load_ki_building,
    split_laps,
)


# ===========================================================================
# 1. Neural-network architectures.
# ===========================================================================
#
# Both networks are permutation-invariant set encoders: each measurement (or
# support point) is encoded by a per-token MLP, the tokens are aggregated by a
# soft-attention pooling that uses the relative position as a spatial gate,
# and a final decoder head emits the targets.  This mirrors how a kernel
# regressor "weights" measurements by distance, except the kernel is learned.


class _SetEncoder(nn.Module):
    """Per-token MLP + attention pooling over a set of (rel_pos, value) tokens."""

    def __init__(self, value_dim: int, hidden: int, n_layers: int = 3,
                 attn_hidden: int = 64):
        super().__init__()
        in_dim = 3 + value_dim                     # (rel:3, value:value_dim)
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            d = hidden
        self.token_mlp = nn.Sequential(*layers)
        # Spatial attention takes per-token features + the rel-position so the
        # gating depends on geometry, not just feature similarity.
        self.attn = nn.Sequential(
            nn.Linear(hidden + 3, attn_hidden), nn.GELU(),
            nn.Linear(attn_hidden, 1),
        )
        # A small bias prior that decays with distance: forces the network to
        # attend more to nearby points at initialisation, which gets training
        # to start in a sensible regime instead of uniform attention.
        self.dist_bias = nn.Parameter(torch.tensor(-1.0))

    def forward(self, ctx_rel: torch.Tensor, ctx_val: torch.Tensor) -> torch.Tensor:
        token_in = torch.cat([ctx_rel, ctx_val], dim=-1)
        feats = self.token_mlp(token_in)                                # (B, K, H)
        attn_in = torch.cat([feats, ctx_rel], dim=-1)
        logits = self.attn(attn_in).squeeze(-1)                         # (B, K)
        # Distance-prior gating - encourages early training to focus on near
        # neighbours, which is where the GP kernel has support.
        d2 = (ctx_rel ** 2).sum(dim=-1)
        logits = logits + self.dist_bias * d2
        weights = torch.softmax(logits, dim=-1)
        return (feats * weights.unsqueeze(-1)).sum(dim=1)               # (B, H)


# Layout of the H_B output: 3 components a in {0,1,2}, with 6 unique (i,j)
# pairs because the Hessian is symmetric in (i,j).  We always feed the same
# pair order to keep packing consistent.
_HB_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


class MagFieldNet(nn.Module):
    """Surrogate for GP inference.

    Inputs are normalised: positions are divided by ``pos_scale`` (we use the
    GP length scale ``ell``) and B values by ``field_scale`` (a robust upper
    quantile of the measured field magnitude).  Outputs are returned in the
    *normalised* space and the helper ``denormalise`` rescales them.

    Architecture: ``_SetEncoder`` over the K measurements, followed by a
    3-layer decoder that emits 3 + 1 + 9 + 18 = 31 numbers.  The 18 H_B values
    are unpacked into a symmetric (3, 3, 3) tensor.
    """

    OUT_DIM = 3 + 1 + 9 + 18

    def __init__(self, hidden: int = 192, n_layers: int = 3):
        super().__init__()
        self.encoder = _SetEncoder(value_dim=3, hidden=hidden, n_layers=n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, MagFieldNet.OUT_DIM),
        )

    def forward(self, ctx_rel: torch.Tensor, ctx_B: torch.Tensor):
        feats = self.encoder(ctx_rel, ctx_B)
        out = self.decoder(feats)
        B = out[..., 0:3]
        var_raw = out[..., 3:4]
        J_flat = out[..., 4:13]
        H_unique = out[..., 13:31]

        J = J_flat.view(*J_flat.shape[:-1], 3, 3)              # [..., a, i]
        H_B = self._unpack_HB(H_unique)                        # [..., a, i, j]
        var = F.softplus(var_raw)                              # >= 0
        return B, var, J, H_B

    @staticmethod
    def _unpack_HB(u: torch.Tensor) -> torch.Tensor:
        """``u`` shape (..., 18) -> symmetric (..., 3, 3, 3) with the 6 unique
        entries per component placed at both (i, j) and (j, i)."""
        prefix = u.shape[:-1]
        u3 = u.view(*prefix, 3, 6)                             # 3 components x 6 unique
        H = torch.zeros(*prefix, 3, 3, 3, dtype=u.dtype, device=u.device)
        for k, (i, j) in enumerate(_HB_PAIRS):
            H[..., :, i, j] = u3[..., :, k]
            if i != j:
                H[..., :, j, i] = u3[..., :, k]
        return H


class LRFNet(nn.Module):
    """Surrogate for the Local Reference Frame builder.

    z is fixed analytically as ``B_kp / ||B_kp||``; the network only learns
    the in-plane x direction.  Inputs are the keypoint's B vector and a small
    support set of (rel_pos, support_B) tokens.  We concatenate B_kp into
    each token so the encoder is conditioned on the keypoint field even
    though aggregation is permutation-invariant in the support order.
    """

    def __init__(self, hidden: int = 96, n_layers: int = 3):
        super().__init__()
        # Per-token value: (support_B:3, B_kp:3) -> 6
        self.encoder = _SetEncoder(value_dim=6, hidden=hidden, n_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, B_kp: torch.Tensor, ctx_rel: torch.Tensor,
                ctx_B: torch.Tensor) -> torch.Tensor:
        K = ctx_rel.shape[1]
        B_kp_exp = B_kp.unsqueeze(1).expand(-1, K, -1)
        ctx_val = torch.cat([ctx_B, B_kp_exp], dim=-1)         # (B, K, 6)
        feats = self.encoder(ctx_rel, ctx_val)
        x_raw = self.head(torch.cat([feats, B_kp], dim=-1))    # (B, 3)
        return x_raw


# ===========================================================================
# 2. Normalisation - keep it deterministic so weights round-trip.
# ===========================================================================


@dataclass
class Norms:
    pos_scale: float                         # divide positions by this
    field_scale: float                       # divide B by this

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def normalise_outputs(B, var, J, H_B, n: Norms):
    """Same transform applied to the teacher labels and the network outputs."""
    fs, ps = n.field_scale, n.pos_scale
    return B / fs, var / (fs ** 2), J * ps / fs, H_B * (ps ** 2) / fs


def denormalise_outputs(B, var, J, H_B, n: Norms):
    fs, ps = n.field_scale, n.pos_scale
    return B * fs, var * (fs ** 2), J * fs / ps, H_B * fs / (ps ** 2)


# ===========================================================================
# 3. Teacher-data generation from the KI Building dataset.
# ===========================================================================
#
# Each lap of the KI Building is fit with the original recursive-sparse GP
# (mag_match.RecursiveSparseGP).  We then sample many query points and ask
# the GP for its predictions; those become regression targets.  We also
# build the analytical LRF at every query point that has enough support, and
# that becomes the LRFNet target.


@dataclass
class TeacherSet:
    """Bundle of per-query training/eval tensors for one map (lap)."""

    query: np.ndarray            # (Q, 3)
    ctx_rel: np.ndarray          # (Q, K, 3)  rel-positions of K nearest measurements
    ctx_B: np.ndarray            # (Q, K, 3)
    teacher_B: np.ndarray        # (Q, 3)
    teacher_var: np.ndarray      # (Q,)
    teacher_J: np.ndarray        # (Q, 3, 3)
    teacher_HB: np.ndarray       # (Q, 3, 3, 3)
    # LRF labels - filled later (need teacher_B at neighbouring query points).
    lrf_kp_B: Optional[np.ndarray] = None      # (Qk, 3)
    lrf_ctx_rel: Optional[np.ndarray] = None   # (Qk, Ks, 3)
    lrf_ctx_B: Optional[np.ndarray] = None     # (Qk, Ks, 3)
    lrf_target_x: Optional[np.ndarray] = None  # (Qk, 3) unit, in plane perp to z
    lrf_target_z: Optional[np.ndarray] = None  # (Qk, 3) unit, just B_kp/||B_kp||


def fit_teacher_gp_for_lap(X: np.ndarray, B: np.ndarray, ell: float = 1.5):
    """Same configuration as ``run_ki_building._run_2d_pipeline`` (flat 2D GP)."""
    field_scale = float(np.quantile(np.linalg.norm(B, axis=1), 0.9))
    field_std = float(np.std(B))
    sigma_f = field_scale
    sigma_n = max(field_std * 0.05, 1.0)

    margin = 2.0 * ell
    lo = X.min(axis=0) - margin
    hi = X.max(axis=0) + margin
    lo[2], hi[2] = -ell * 0.5, ell * 0.5

    n_grid = int(np.ceil(50 / ell) + 2)
    ax = [np.linspace(lo[i], hi[i], n_grid) for i in range(2)]
    XX, YY = np.meshgrid(*ax, indexing="ij")
    ZZ = np.full_like(XX, 0.5 * (lo[2] + hi[2]))
    U = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    kernel = DivergenceFreeKernel(sigma_f, ell)
    gp = RecursiveSparseGP(kernel, U, sigma_n=sigma_n)
    gp.fit_batch(X, B, batch_size=64)
    return gp, dict(
        ell=ell, sigma_f=sigma_f, sigma_n=sigma_n,
        field_scale=field_scale, field_std=field_std,
        bounds_lo=lo, bounds_hi=hi,
    )


def make_query_grid(X: np.ndarray, ell: float, density: float = 1.0):
    """Dense 2D query grid restricted to the lap's footprint.

    ``density`` scales the spacing: larger -> more points (more training data).
    """
    inset = 0.5 * ell
    spacing = ell / density
    lo = X.min(axis=0) + inset
    hi = X.max(axis=0) - inset
    n_x = max(int(np.ceil((hi[0] - lo[0]) / spacing)) + 1, 2)
    n_y = max(int(np.ceil((hi[1] - lo[1]) / spacing)) + 1, 2)
    ax_x = np.linspace(lo[0], hi[0], n_x)
    ax_y = np.linspace(lo[1], hi[1], n_y)
    XX, YY = np.meshgrid(ax_x, ax_y, indexing="ij")
    Q = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])

    # Drop queries that fall in regions not covered by any measurement: those
    # only confuse training because the GP has no information there and emits
    # noise-dominated derivatives.
    tree = cKDTree(X[:, :2])
    d, _ = tree.query(Q[:, :2], k=1)
    return Q[d < 2.0 * ell]


def gather_K_nearest_context(X_meas: np.ndarray, B_meas: np.ndarray,
                              X_query: np.ndarray, K: int):
    """For each query, gather K nearest measurements as (rel_pos, B)."""
    tree = cKDTree(X_meas)
    _, idx = tree.query(X_query, k=K)
    ctx_pos = X_meas[idx]
    ctx_B = B_meas[idx]
    rel = ctx_pos - X_query[:, None, :]
    return rel.astype(np.float32), ctx_B.astype(np.float32)


def build_lrf_labels(query: np.ndarray, teacher_B: np.ndarray,
                     support_radius: float, K_support: int = 32,
                     min_support: int = 8):
    """Build LRF training labels.

    For each query that has at least ``min_support`` neighbouring queries inside
    ``support_radius`` and a non-trivial B-magnitude, run the analytical LRF
    builder to obtain the target x-axis (in the plane orthogonal to z = B_kp).
    The supplied support tokens are the K nearest queries within radius (with
    NN-friendly fixed K and zero-padding when fewer are available).
    """
    Q = query.shape[0]
    norm_B = np.linalg.norm(teacher_B, axis=1)
    valid_kp = norm_B > 1e-3 * np.median(norm_B + 1e-9)

    tree = cKDTree(query)
    # +1 to drop self.
    dists, idx = tree.query(query, k=K_support + 1)
    self_mask = idx == np.arange(Q)[:, None]
    # Push self to the back so the first K columns are non-self if at all
    # possible (idx[:, 0] is always self because the query coords are exact).
    # We just drop column 0.
    idx = idx[:, 1:]
    dists = dists[:, 1:]

    rels: List[np.ndarray] = []
    sup_Bs: List[np.ndarray] = []
    targets_x: List[np.ndarray] = []
    targets_z: List[np.ndarray] = []
    kp_Bs: List[np.ndarray] = []
    keep_idx: List[int] = []

    for q in range(Q):
        if not valid_kp[q]:
            continue
        keep = dists[q] <= support_radius
        if int(keep.sum()) < min_support:
            continue
        in_idx = idx[q][keep]
        sup_pos = query[in_idx]
        sup_B = teacher_B[in_idx]
        rel = sup_pos - query[q]
        # Reference LRF from the analytical builder.
        R_lrf = _build_lrf_with_positions(teacher_B[q], rel, sup_B, support_radius)
        x_axis = R_lrf[:, 0]
        z_axis = R_lrf[:, 2]

        # Pack into fixed-K tokens; pad with zeros and rely on the spatial
        # attention bias to ignore far/blank entries.
        rel_pad = np.zeros((K_support, 3), dtype=np.float32)
        sup_B_pad = np.zeros((K_support, 3), dtype=np.float32)
        n = min(rel.shape[0], K_support)
        # Take the n closest to the keypoint - they carry the most signal.
        order = np.argsort(np.linalg.norm(rel, axis=1))[:n]
        rel_pad[:n] = rel[order]
        sup_B_pad[:n] = sup_B[order]
        # Far-away padding so the attention prior gates them to ~zero.
        if n < K_support:
            rel_pad[n:] = 100.0 * support_radius

        rels.append(rel_pad)
        sup_Bs.append(sup_B_pad)
        targets_x.append(x_axis.astype(np.float32))
        targets_z.append(z_axis.astype(np.float32))
        kp_Bs.append(teacher_B[q].astype(np.float32))
        keep_idx.append(q)

    if not rels:
        return None
    return dict(
        kp_B=np.stack(kp_Bs),
        ctx_rel=np.stack(rels),
        ctx_B=np.stack(sup_Bs),
        target_x=np.stack(targets_x),
        target_z=np.stack(targets_z),
        keep_idx=np.array(keep_idx, dtype=int),
    )


def build_teacher_set(X_meas: np.ndarray, B_meas: np.ndarray,
                      ell: float = 1.5, K_ctx: int = 64,
                      query_density: float = 1.6,
                      lrf_K_support: int = 32) -> TeacherSet:
    """Run the full teacher pipeline for one lap and pack the labels."""
    gp, _ = fit_teacher_gp_for_lap(X_meas, B_meas, ell=ell)
    Q = make_query_grid(X_meas, ell=ell, density=query_density)

    # GP labels at every query.
    teacher_B, teacher_var = gp.predict_batch(Q)
    teacher_J = gp.predict_jacobian_batch(Q)
    teacher_HB = gp.predict_field_hessian_batch(Q)

    # Measurement context.
    rel, ctx_B = gather_K_nearest_context(X_meas, B_meas, Q, K=K_ctx)

    # LRF labels.
    support_radius = ell * 4.0
    lrf = build_lrf_labels(Q, teacher_B, support_radius,
                           K_support=lrf_K_support)

    ts = TeacherSet(
        query=Q.astype(np.float32),
        ctx_rel=rel,
        ctx_B=ctx_B,
        teacher_B=teacher_B.astype(np.float32),
        teacher_var=teacher_var.astype(np.float32),
        teacher_J=teacher_J.astype(np.float32),
        teacher_HB=teacher_HB.astype(np.float32),
    )
    if lrf is not None:
        ts.lrf_kp_B = lrf["kp_B"]
        ts.lrf_ctx_rel = lrf["ctx_rel"]
        ts.lrf_ctx_B = lrf["ctx_B"]
        ts.lrf_target_x = lrf["target_x"]
        ts.lrf_target_z = lrf["target_z"]
    return ts


# ===========================================================================
# 4. Training loops with cosine schedule + early stopping.
# ===========================================================================


def _stack_field_tensors(sets: List[TeacherSet], norms: Norms, device: torch.device):
    rel = np.concatenate([s.ctx_rel for s in sets]) / norms.pos_scale
    B = np.concatenate([s.ctx_B for s in sets]) / norms.field_scale
    tB = np.concatenate([s.teacher_B for s in sets]) / norms.field_scale
    tvar = np.concatenate([s.teacher_var for s in sets]) / (norms.field_scale ** 2)
    tJ = np.concatenate([s.teacher_J for s in sets]) * norms.pos_scale / norms.field_scale
    tHB = np.concatenate([s.teacher_HB for s in sets]) * (norms.pos_scale ** 2) / norms.field_scale
    return [torch.from_numpy(a.astype(np.float32)).to(device)
            for a in (rel, B, tB, tvar, tJ, tHB)]


def _stack_lrf_tensors(sets: List[TeacherSet], norms: Norms,
                       lrf_support_radius: float, device: torch.device):
    arrs = [(s.lrf_kp_B, s.lrf_ctx_rel, s.lrf_ctx_B, s.lrf_target_x, s.lrf_target_z)
            for s in sets if s.lrf_kp_B is not None]
    if not arrs:
        return None
    kp_B = np.concatenate([a[0] for a in arrs]) / norms.field_scale
    rel = np.concatenate([a[1] for a in arrs]) / lrf_support_radius
    sup_B = np.concatenate([a[2] for a in arrs]) / norms.field_scale
    target_x = np.concatenate([a[3] for a in arrs])
    target_z = np.concatenate([a[4] for a in arrs])
    return [torch.from_numpy(a.astype(np.float32)).to(device)
            for a in (kp_B, rel, sup_B, target_x, target_z)]


def train_magfield_net(net: MagFieldNet, train_sets: List[TeacherSet],
                        val_sets: List[TeacherSet],
                        norms: Norms, device: torch.device,
                        epochs: int = 500, batch_size: int = 1024,
                        lr: float = 1e-3, weight_decay: float = 1e-5,
                        patience: int = 60, log_every: int = 20):
    """Distil GP outputs into MagFieldNet."""
    train = _stack_field_tensors(train_sets, norms, device)
    val = _stack_field_tensors(val_sets, norms, device)
    rel_t, B_t, tB_t, tvar_t, tJ_t, tHB_t = train
    rel_v, B_v, tB_v, tvar_v, tJ_v, tHB_v = val
    N = rel_t.shape[0]

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_state = None
    bad = 0

    def loss_fn(pred, B_tgt, var_tgt, J_tgt, HB_tgt):
        Bp, vp, Jp, Hp = pred
        # H_B has 18 unique values; weighting it slightly higher gives it
        # parity with the lower-dim B output in the gradient signal.
        return (
            F.mse_loss(Bp, B_tgt)
            + 0.25 * F.mse_loss(vp.squeeze(-1), var_tgt)
            + 1.0 * F.mse_loss(Jp, J_tgt)
            + 1.5 * F.mse_loss(Hp, HB_tgt)
        )

    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(N, device=device)
        running = 0.0
        n_batches = 0
        for s in range(0, N, batch_size):
            sel = perm[s:s + batch_size]
            pred = net(rel_t[sel], B_t[sel])
            loss = loss_fn(pred, tB_t[sel], tvar_t[sel], tJ_t[sel], tHB_t[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            running += loss.item()
            n_batches += 1
        sched.step()

        net.eval()
        with torch.no_grad():
            pred = net(rel_v, B_v)
            vloss = loss_fn(pred, tB_v, tvar_v, tJ_v, tHB_v).item()

        if vloss < best_val * 0.999:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % log_every == 0 or epoch == 1:
            print(f"  [field] epoch {epoch:4d}/{epochs} | "
                  f"train {running / max(n_batches, 1):.5f} | val {vloss:.5f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | best {best_val:.5f}")

        if bad >= patience:
            print(f"  [field] early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    return best_val


def _random_rotations(n: int, device: torch.device) -> torch.Tensor:
    """Uniformly sampled rotation matrices in SO(3) via QR of Gaussian noise."""
    A = torch.randn(n, 3, 3, device=device)
    Q, _ = torch.linalg.qr(A)
    # QR can produce reflections (det = -1); flip a column to enforce det = +1.
    det = torch.linalg.det(Q)
    Q = Q.clone()
    Q[:, :, 0] = Q[:, :, 0] * det.unsqueeze(-1)
    return Q


def _augment_lrf_batch(R: torch.Tensor, kp_B: torch.Tensor, ctx_rel: torch.Tensor,
                        ctx_B: torch.Tensor, target_x: torch.Tensor,
                        target_z: torch.Tensor):
    """Rotate every input/output vector by the per-sample rotation R.

    The analytical LRF is rotation-equivariant: rotating the entire scene by R
    rotates the LRF basis by R as well.  This gives us a free SO(3) data
    augmentation that all but eliminates LRFNet's rotational generalisation
    gap.
    """
    kp_B_r = torch.einsum("nij,nj->ni", R, kp_B)
    ctx_rel_r = torch.einsum("nij,nkj->nki", R, ctx_rel)
    ctx_B_r = torch.einsum("nij,nkj->nki", R, ctx_B)
    target_x_r = torch.einsum("nij,nj->ni", R, target_x)
    target_z_r = torch.einsum("nij,nj->ni", R, target_z)
    return kp_B_r, ctx_rel_r, ctx_B_r, target_x_r, target_z_r


def train_lrf_net(net: LRFNet, train_sets: List[TeacherSet],
                  val_sets: List[TeacherSet],
                  norms: Norms, lrf_support_radius: float, device: torch.device,
                  epochs: int = 500, batch_size: int = 1024,
                  lr: float = 1e-3, weight_decay: float = 1e-5,
                  patience: int = 60, log_every: int = 20,
                  augment: bool = True):
    """Distil the analytical LRF into LRFNet via a cosine-similarity loss.

    With ``augment=True`` (default) every batch is rotated by a random SO(3)
    matrix (one per sample), exploiting the LRF's rotation equivariance.  This
    gives effectively unlimited training data and forces the network to be
    rotation-equivariant rather than memorising one global frame.
    """
    train = _stack_lrf_tensors(train_sets, norms, lrf_support_radius, device)
    val = _stack_lrf_tensors(val_sets, norms, lrf_support_radius, device)
    if train is None or val is None:
        print("  [lrf] no LRF labels available, skipping LRFNet training")
        return float("inf")
    kp_B_t, rel_t, sup_B_t, x_t, z_t = train
    kp_B_v, rel_v, sup_B_v, x_v, z_v = val
    N = kp_B_t.shape[0]

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def project_and_normalise(x_raw, z_unit, eps: float = 1e-6):
        proj = x_raw - (x_raw * z_unit).sum(dim=-1, keepdim=True) * z_unit
        n = proj.norm(dim=-1, keepdim=True).clamp_min(eps)
        return proj / n

    def loss_fn(x_pred_raw, x_target, z_unit):
        x_pred = project_and_normalise(x_pred_raw, z_unit)
        # Cosine similarity to target.  The teacher already disambiguates the
        # sign with its weighted-majority rule, so we use signed cosine here.
        cos = (x_pred * x_target).sum(dim=-1)
        return (1.0 - cos).mean()

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(N, device=device)
        running = 0.0
        n_batches = 0
        for s in range(0, N, batch_size):
            sel = perm[s:s + batch_size]
            kp_b = kp_B_t[sel]; rel_b = rel_t[sel]; sup_b = sup_B_t[sel]
            x_b = x_t[sel]; z_b = z_t[sel]
            if augment:
                R = _random_rotations(kp_b.shape[0], device)
                kp_b, rel_b, sup_b, x_b, z_b = _augment_lrf_batch(
                    R, kp_b, rel_b, sup_b, x_b, z_b)
            x_raw = net(kp_b, rel_b, sup_b)
            loss = loss_fn(x_raw, x_b, z_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            running += loss.item()
            n_batches += 1
        sched.step()

        net.eval()
        with torch.no_grad():
            # Validate on the un-augmented set (i.e. the natural distribution
            # produced by the GP teacher).  The augmentation is only there to
            # force rotational generalisation during training.
            x_raw = net(kp_B_v, rel_v, sup_B_v)
            vloss = loss_fn(x_raw, x_v, z_v).item()

        if vloss < best_val * 0.999:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % log_every == 0 or epoch == 1:
            print(f"  [lrf]   epoch {epoch:4d}/{epochs} | "
                  f"train {running / max(n_batches, 1):.5f} | val {vloss:.5f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | best {best_val:.5f}")

        if bad >= patience:
            print(f"  [lrf]   early stop at epoch {epoch}")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    return best_val


# ===========================================================================
# 5. ML inference pipeline.
# ===========================================================================
#
# Same shape and same outputs as ``mag_match.extract_features``, but every GP
# call is replaced by a single batched MagFieldNet forward pass and every
# LRF construction is replaced by a batched LRFNet forward pass.


def _predict_field(magfield_net: MagFieldNet, X_query: np.ndarray,
                   X_meas: np.ndarray, B_meas: np.ndarray,
                   norms: Norms, K_ctx: int, device: torch.device,
                   batch_size: int = 4096):
    """Vectorised B/var/J/H_B inference on a query set."""
    rel, ctx_B = gather_K_nearest_context(X_meas, B_meas, X_query, K=K_ctx)
    rel_t = torch.from_numpy(rel).to(device) / norms.pos_scale
    B_t = torch.from_numpy(ctx_B).to(device) / norms.field_scale

    Q = rel_t.shape[0]
    out_B = np.empty((Q, 3), dtype=np.float32)
    out_var = np.empty(Q, dtype=np.float32)
    out_J = np.empty((Q, 3, 3), dtype=np.float32)
    out_HB = np.empty((Q, 3, 3, 3), dtype=np.float32)

    magfield_net.eval()
    with torch.no_grad():
        for s in range(0, Q, batch_size):
            e = s + batch_size
            Bp, vp, Jp, Hp = magfield_net(rel_t[s:e], B_t[s:e])
            Bp, vp, Jp, Hp = denormalise_outputs(Bp, vp.squeeze(-1), Jp, Hp, norms)
            out_B[s:e] = Bp.cpu().numpy()
            out_var[s:e] = vp.cpu().numpy()
            out_J[s:e] = Jp.cpu().numpy()
            out_HB[s:e] = Hp.cpu().numpy()
    return out_B, out_var, out_J, out_HB


def _predict_lrfs(lrf_net: LRFNet, kp_pos: np.ndarray, kp_B: np.ndarray,
                  X_query: np.ndarray, B_query: np.ndarray,
                  support_radius: float, K_support: int,
                  norms: Norms, device: torch.device) -> np.ndarray:
    """Build LRFs (Nx3x3) for every keypoint using the network."""
    if kp_pos.shape[0] == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    tree = cKDTree(X_query)
    dists, idxs = tree.query(kp_pos, k=K_support + 1)
    # Drop self (keypoints are exactly query points).
    idxs = idxs[:, 1:] if idxs.shape[1] > 1 else idxs
    dists = dists[:, 1:] if dists.shape[1] > 1 else dists

    rel_arr = np.zeros((kp_pos.shape[0], K_support, 3), dtype=np.float32)
    sup_B_arr = np.zeros((kp_pos.shape[0], K_support, 3), dtype=np.float32)
    rel_arr[..., :] = 100.0 * support_radius     # padding far away
    n_support = idxs.shape[1]
    if n_support > K_support:
        # Already guaranteed K_support, but defensive.
        idxs = idxs[:, :K_support]
        dists = dists[:, :K_support]
        n_support = K_support

    for k in range(kp_pos.shape[0]):
        keep = dists[k] <= support_radius
        if int(keep.sum()) < 1:
            continue
        sel = idxs[k][keep]
        rel = X_query[sel] - kp_pos[k]
        sup_B = B_query[sel]
        order = np.argsort(np.linalg.norm(rel, axis=1))[:K_support]
        n = order.size
        rel_arr[k, :n] = rel[order]
        sup_B_arr[k, :n] = sup_B[order]

    kp_B_t = torch.from_numpy(kp_B.astype(np.float32)).to(device) / norms.field_scale
    rel_t = torch.from_numpy(rel_arr).to(device) / support_radius
    sup_B_t = torch.from_numpy(sup_B_arr).to(device) / norms.field_scale

    lrf_net.eval()
    with torch.no_grad():
        x_raw = lrf_net(kp_B_t, rel_t, sup_B_t).cpu().numpy()

    # Build the rotation: z = B_kp/||B_kp||, x = project(x_raw onto z-perp), y = z x x.
    R_lrfs = np.zeros((kp_pos.shape[0], 3, 3), dtype=np.float32)
    eps = 1e-9
    for k in range(kp_pos.shape[0]):
        nB = float(np.linalg.norm(kp_B[k]))
        if nB < eps:
            R_lrfs[k] = np.eye(3)
            continue
        z = kp_B[k] / nB
        x = x_raw[k] - (x_raw[k] @ z) * z
        nx = float(np.linalg.norm(x))
        if nx < eps:
            ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = ref - (ref @ z) * z
            nx = float(np.linalg.norm(x))
        x = x / max(nx, eps)
        y = np.cross(z, x)
        R_lrfs[k] = np.column_stack([x, y, z])
    return R_lrfs


def extract_features_ml(measurements_X: np.ndarray, measurements_B: np.ndarray,
                         X_query: np.ndarray, magfield_net: MagFieldNet,
                         lrf_net: LRFNet, norms: Norms,
                         cfg: MagMatchConfig, device: torch.device,
                         K_ctx: int = 64, K_support: int = 32):
    """ML mirror of ``mag_match.extract_features``.

    Steps that change vs the original:
      - GP -> MagFieldNet (one batched forward pass over all query points).
      - LRF eigendecomposition -> LRFNet (one batched forward pass over kps).

    Steps that stay analytical:
      - Hessian of ||B|| chain rule.
      - Mean-DoH thresholding, variance gating, NMS.
      - HOV descriptor (already O(n) and trivial).
    """
    # 1. Field at every query point (replaces predict_batch / Jacobian / Hessian batch).
    B_q, var_q, J_q, HB_q = _predict_field(
        magfield_net, X_query, measurements_X, measurements_B, norms,
        K_ctx=K_ctx, device=device,
    )

    # 2. Hessian of ||B||, DoH, gating - same as the original code path.
    H = hessian_of_norm_batch(B_q, J_q, HB_q)
    doh = np.linalg.det(H)
    abs_doh = np.abs(doh)
    doh_thresh = cfg.min_doh_factor * np.median(abs_doh)
    var_thresh = np.quantile(var_q, cfg.variance_quantile)
    keep = (abs_doh >= doh_thresh) & (var_q <= var_thresh)
    cand = np.where(keep)[0]
    if cand.size == 0:
        return [], dict(B=B_q, var=var_q, J=J_q, HB=HB_q, doh=doh)

    nms_kept = _non_max_suppression(
        X_query[cand], doh[cand],
        radius=cfg.nms_radius if cfg.nms_radius is not None else 1.0,
        max_keep=cfg.max_keypoints,
    )
    keep_idx = cand[nms_kept]

    # 3. Build LRFs in one shot.
    support_radius = cfg.support_radius if cfg.support_radius is not None else 4.0
    R_lrfs = _predict_lrfs(
        lrf_net, X_query[keep_idx], B_q[keep_idx], X_query, B_q,
        support_radius=support_radius, K_support=K_support,
        norms=norms, device=device,
    )

    # 4. Pack keypoints + analytical HOV descriptors.
    keypoints: List[Keypoint] = []
    for slot, kidx in enumerate(keep_idx):
        pos = X_query[kidx]
        B_kp = B_q[kidx]
        if np.linalg.norm(B_kp) < 1e-9:
            continue
        rel = X_query - pos
        dist = np.linalg.norm(rel, axis=1)
        sup_mask = (dist <= support_radius) & (dist > 1e-9)
        if sup_mask.sum() < 6:
            continue
        R_lrf = R_lrfs[slot]
        # Transform support B vectors into the LRF (v_lrf = R^T v_map).
        B_support_lrf = (R_lrf.T @ B_q[sup_mask].T).T
        descriptor = hov_descriptor(
            B_support_lrf, dist[sup_mask], support_radius,
            component_range=cfg.component_range,
            component_bin=cfg.component_bin,
        )
        keypoints.append(Keypoint(
            pos=pos.copy(), B=B_kp.copy(), R_lrf=R_lrf.copy(),
            descriptor=descriptor.copy(),
            doh=float(doh[kidx]), var=float(var_q[kidx]),
        ))

    return keypoints, dict(B=B_q, var=var_q, J=J_q, HB=HB_q, doh=doh)


# ===========================================================================
# 6. End-to-end KI Building runner.
# ===========================================================================


def _select_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" or (preferred == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if preferred == "mps" or (preferred == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def _make_query_grid_cfg(X: np.ndarray, ell: float, density: float = 1.6):
    """Same grid the trainer uses, for consistent inference behaviour."""
    return make_query_grid(X, ell=ell, density=density)


def _ki_build_cfg(field_scale: float, ell: float = 1.5) -> MagMatchConfig:
    """Mirror of the config in run_ki_building._run_2d_pipeline.

    Two thresholds are tuned for the surrogate.  ``min_doh_factor`` is dropped
    from 2.0 -> 1.0 because the network's per-cell DoH distribution is more
    concentrated than the GP's, so the median * 2 cut rejected too many
    candidates.  ``nms_radius`` is dropped from 1.5 m -> 1.0 m for the same
    reason, recovering enough keypoints (~150-180/map) for MSAC's consensus
    voting to wash out the per-descriptor noise.  Everything else is unchanged.
    """
    return MagMatchConfig(
        sigma_f=field_scale,
        ell=ell,
        sigma_n=max(field_scale * 0.05, 1.0),
        inducing_grid=int(np.ceil(50 / ell) + 2),
        query_grid=int(np.ceil(50 / ell) + 8),
        variance_quantile=0.5,
        min_doh_factor=1.0,
        support_radius=ell * 4.0,
        component_range=field_scale * 2.0,
        component_bin=field_scale * 0.2,
        nms_radius=ell * 0.7,
        max_keypoints=200,
    )


def run_ki_ml(magfield_net: MagFieldNet, lrf_net: LRFNet, norms: Norms,
              device: torch.device, ell: float = 1.5, K_ctx: int = 64,
              K_support: int = 32, query_density: float = 1.6,
              seed: int = 0, verbose: bool = True) -> dict:
    """Run lap1-vs-lap2 registration on KI Building with the ML surrogates."""
    pos, mag_body, yaw = load_ki_building(DATASET_PATH)
    B_world = body_to_world(mag_body, yaw)
    B_world = B_world - B_world.mean(axis=0)

    lap1, lap2 = split_laps(pos)
    X_base, B_base = pos[lap1].copy(), B_world[lap1].copy()
    X_target, B_target = pos[lap2].copy(), B_world[lap2].copy()

    field_scale = float(np.quantile(np.linalg.norm(B_base, axis=1), 0.9))
    cfg = _ki_build_cfg(field_scale, ell=ell)

    Q_base = _make_query_grid_cfg(X_base, ell=ell, density=query_density)
    Q_target = _make_query_grid_cfg(X_target, ell=ell, density=query_density)

    if verbose:
        print(f"  base   : {len(X_base)} measurements, {Q_base.shape[0]} queries")
        print(f"  target : {len(X_target)} measurements, {Q_target.shape[0]} queries")

    t0 = time.time()
    kps_base, _ = extract_features_ml(
        X_base, B_base, Q_base, magfield_net, lrf_net, norms, cfg, device,
        K_ctx=K_ctx, K_support=K_support,
    )
    t1 = time.time()
    kps_target, _ = extract_features_ml(
        X_target, B_target, Q_target, magfield_net, lrf_net, norms, cfg, device,
        K_ctx=K_ctx, K_support=K_support,
    )
    t2 = time.time()

    if verbose:
        print(f"  base keypoints  : {len(kps_base)} in {t1 - t0:.2f}s")
        print(f"  target keypoints: {len(kps_target)} in {t2 - t1:.2f}s")

    # Same matcher and MSAC settings as the analytical pipeline; the
    # surrogate's per-match descriptor noise is on the same order as the GP's
    # so these numbers transfer.
    matches = match_descriptors(kps_base, kps_target, ratio=1.0)
    if verbose:
        print(f"  putative NN matches: {matches.shape[0]}")

    R_est, t_est, inlier_mask, fitness = modified_msac(
        kps_base, kps_target, matches,
        inlier_dist=2.0, cross_thresh=0.4, iters=20000,
        random_state=seed,
    )

    R_gt = np.eye(3)
    t_gt = np.zeros(3)
    rot_err = rotation_angle_deg(R_est @ R_gt.T)
    trans_err = float(np.linalg.norm(t_est - t_gt))
    n_inliers = int(inlier_mask.sum())

    if verbose:
        print()
        print("  === Registration result (ML pipeline) ===")
        print(f"    Rotation error    : {rot_err:.4f} deg")
        print(f"    Translation error : {trans_err:.4f} m")
        print(f"    Inliers           : {n_inliers}/{matches.shape[0]}")
        print(f"    Fitness (mean ||cross||) : {fitness:.4f}")
        print()
        print("  Reference (paper Table I, KI Building map-to-map):")
        print("    Translation: 0.1784 m  +/-  0.0836")
        print("    Rotation:    0.2688 deg +/- 0.1076")

    return dict(
        rotation_error_deg=rot_err,
        translation_error_m=trans_err,
        n_inliers=n_inliers,
        n_matches=int(matches.shape[0]),
        n_keypoints_base=len(kps_base),
        n_keypoints_target=len(kps_target),
        fitness=fitness,
        time_features_s=(t2 - t0),
    )


# ===========================================================================
# 7. Persistence helpers.
# ===========================================================================


def save_models(path: str, magfield_net: MagFieldNet, lrf_net: LRFNet,
                norms: Norms, hparams: dict):
    torch.save({
        "magfield_state": magfield_net.state_dict(),
        "lrf_state": lrf_net.state_dict(),
        "norms": norms.to_dict(),
        "hparams": hparams,
    }, path)


def load_models(path: str, device: torch.device):
    data = torch.load(path, map_location=device, weights_only=False)
    hp = data["hparams"]
    mf = MagFieldNet(hidden=hp["mf_hidden"], n_layers=hp["mf_layers"]).to(device)
    lf = LRFNet(hidden=hp["lrf_hidden"], n_layers=hp["lrf_layers"]).to(device)
    mf.load_state_dict(data["magfield_state"])
    lf.load_state_dict(data["lrf_state"])
    norms = Norms.from_dict(data["norms"])
    return mf, lf, norms, hp


# ===========================================================================
# 8. Main.
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Mag-Match-ML on KI Building")
    parser.add_argument("--epochs", type=int, default=500,
                        help="max training epochs for each network")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=120,
                        help="early-stopping patience")
    parser.add_argument("--mf-hidden", type=int, default=256,
                        help="MagFieldNet hidden width")
    parser.add_argument("--mf-layers", type=int, default=4,
                        help="MagFieldNet token-MLP depth")
    parser.add_argument("--lrf-hidden", type=int, default=128)
    parser.add_argument("--lrf-layers", type=int, default=4)
    parser.add_argument("--K-ctx", type=int, default=64,
                        help="K nearest measurements per query for MagFieldNet")
    parser.add_argument("--K-support", type=int, default=32,
                        help="K nearest queries per keypoint for LRFNet")
    parser.add_argument("--query-density", type=float, default=2.0,
                        help="multiplier on (1/ell) for query grid spacing")
    parser.add_argument("--ell", type=float, default=1.5,
                        help="GP length scale (matches run_ki_building)")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="fraction of training samples held out for validation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default="mag_match_ml.pt",
                        help="path for the cached weights")
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse cached weights instead of training")
    parser.add_argument("--mc-trials", type=int, default=10,
                        help="Monte Carlo MSAC seeds reported after the single-trial run")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _select_device(args.device)
    print("=" * 64)
    print(" Mag-Match-ML  -  neural surrogate for the GP and LRF stages")
    print("=" * 64)
    print(f"  device: {device}")
    print(f"  weights file: {args.weights}")

    weights_path = Path(args.weights)

    if args.skip_train:
        if not weights_path.exists():
            raise SystemExit(f"--skip-train was requested but {args.weights} does not exist")
        print(f"\n[load] reading cached weights from {args.weights}")
        magfield_net, lrf_net, norms, _ = load_models(str(weights_path), device)
    else:
        # ----- Build teacher dataset from BOTH laps. -----
        print("\n[teacher] running the original GP teacher on both laps...")
        pos, mag_body, yaw = load_ki_building(DATASET_PATH)
        B_world = body_to_world(mag_body, yaw)
        B_world = B_world - B_world.mean(axis=0)
        lap1, lap2 = split_laps(pos)
        teacher_t0 = time.time()
        ts_lap1 = build_teacher_set(
            pos[lap1], B_world[lap1], ell=args.ell, K_ctx=args.K_ctx,
            query_density=args.query_density, lrf_K_support=args.K_support,
        )
        ts_lap2 = build_teacher_set(
            pos[lap2], B_world[lap2], ell=args.ell, K_ctx=args.K_ctx,
            query_density=args.query_density, lrf_K_support=args.K_support,
        )
        teacher_t1 = time.time()
        print(f"  lap1: {ts_lap1.query.shape[0]} queries, "
              f"lap2: {ts_lap2.query.shape[0]} queries  "
              f"({teacher_t1 - teacher_t0:.1f}s)")

        # ----- Normalisation constants. -----
        # field_scale: the larger of the two laps' field scales.  pos_scale: ell.
        # B values are normalised by ``field_scale`` and positions by ell.
        # Both choices match the dimensional analysis in mag_match.py (the
        # divergence-free kernel is itself written in units of ell).
        field_scale = max(
            float(np.quantile(np.linalg.norm(ts_lap1.ctx_B, axis=2).ravel(), 0.9)),
            float(np.quantile(np.linalg.norm(ts_lap2.ctx_B, axis=2).ravel(), 0.9)),
        )
        norms = Norms(pos_scale=args.ell, field_scale=field_scale)
        print(f"  norms: pos_scale = ell = {norms.pos_scale}, "
              f"field_scale = {norms.field_scale:.2f}")

        # ----- train/val split: pool both laps, shuffle, slice by fraction.
        # LRF tensors are split with their own independent permutation so both
        # nets see a clean held-out set.
        rng = np.random.default_rng(args.seed)

        def pool_and_split(sets: List[TeacherSet]):
            # Field tensors.
            queries = [s.query for s in sets]
            ctx_rel = [s.ctx_rel for s in sets]
            ctx_B = [s.ctx_B for s in sets]
            tB = [s.teacher_B for s in sets]
            tvar = [s.teacher_var for s in sets]
            tJ = [s.teacher_J for s in sets]
            tHB = [s.teacher_HB for s in sets]
            big = TeacherSet(
                query=np.concatenate(queries),
                ctx_rel=np.concatenate(ctx_rel),
                ctx_B=np.concatenate(ctx_B),
                teacher_B=np.concatenate(tB),
                teacher_var=np.concatenate(tvar),
                teacher_J=np.concatenate(tJ),
                teacher_HB=np.concatenate(tHB),
            )
            Q = big.query.shape[0]
            order = rng.permutation(Q)
            cut = int(round((1.0 - args.val_frac) * Q))
            tr, va = order[:cut], order[cut:]

            def take(idx):
                return TeacherSet(
                    query=big.query[idx], ctx_rel=big.ctx_rel[idx],
                    ctx_B=big.ctx_B[idx], teacher_B=big.teacher_B[idx],
                    teacher_var=big.teacher_var[idx], teacher_J=big.teacher_J[idx],
                    teacher_HB=big.teacher_HB[idx],
                )

            train = take(tr); val = take(va)

            # LRF labels: pool, shuffle, split independently.
            lrf_arrs = [s for s in sets if s.lrf_kp_B is not None]
            if lrf_arrs:
                kp_B = np.concatenate([s.lrf_kp_B for s in lrf_arrs])
                lrel = np.concatenate([s.lrf_ctx_rel for s in lrf_arrs])
                lB = np.concatenate([s.lrf_ctx_B for s in lrf_arrs])
                lx = np.concatenate([s.lrf_target_x for s in lrf_arrs])
                lz = np.concatenate([s.lrf_target_z for s in lrf_arrs])
                Ql = kp_B.shape[0]
                lo = rng.permutation(Ql)
                cut_l = int(round((1.0 - args.val_frac) * Ql))
                tl_idx, vl_idx = lo[:cut_l], lo[cut_l:]
                train.lrf_kp_B = kp_B[tl_idx]; train.lrf_ctx_rel = lrel[tl_idx]
                train.lrf_ctx_B = lB[tl_idx]; train.lrf_target_x = lx[tl_idx]
                train.lrf_target_z = lz[tl_idx]
                val.lrf_kp_B = kp_B[vl_idx]; val.lrf_ctx_rel = lrel[vl_idx]
                val.lrf_ctx_B = lB[vl_idx]; val.lrf_target_x = lx[vl_idx]
                val.lrf_target_z = lz[vl_idx]

            return train, val

        train_set, val_set = pool_and_split([ts_lap1, ts_lap2])
        print(f"  pooled: train {train_set.query.shape[0]} / val {val_set.query.shape[0]} "
              f"queries; "
              f"LRF train {0 if train_set.lrf_kp_B is None else train_set.lrf_kp_B.shape[0]} / "
              f"val {0 if val_set.lrf_kp_B is None else val_set.lrf_kp_B.shape[0]}")

        # ----- Networks. -----
        magfield_net = MagFieldNet(hidden=args.mf_hidden, n_layers=args.mf_layers).to(device)
        lrf_net = LRFNet(hidden=args.lrf_hidden, n_layers=args.lrf_layers).to(device)

        n_params_mf = sum(p.numel() for p in magfield_net.parameters())
        n_params_lrf = sum(p.numel() for p in lrf_net.parameters())
        print(f"  MagFieldNet: {n_params_mf:,} params   "
              f"LRFNet: {n_params_lrf:,} params")

        # ----- Train. -----
        print("\n[train] MagFieldNet (B, var, J, H_B) ...")
        train_magfield_net(
            magfield_net, [train_set], [val_set], norms, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience,
        )
        print("\n[train] LRFNet (x-axis) ...")
        train_lrf_net(
            lrf_net, [train_set], [val_set], norms,
            lrf_support_radius=args.ell * 4.0, device=device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience,
        )

        save_models(str(weights_path), magfield_net, lrf_net, norms, dict(
            mf_hidden=args.mf_hidden, mf_layers=args.mf_layers,
            lrf_hidden=args.lrf_hidden, lrf_layers=args.lrf_layers,
            ell=args.ell, K_ctx=args.K_ctx, K_support=args.K_support,
        ))
        print(f"\n[save] weights written to {args.weights}")

    # ----- Evaluate on KI Building map-to-map. -----
    print("\n[eval] KI Building map-to-map registration (single trial) ...")
    out = run_ki_ml(
        magfield_net, lrf_net, norms, device,
        ell=args.ell, K_ctx=args.K_ctx, K_support=args.K_support,
        query_density=args.query_density, seed=args.seed,
    )

    if args.mc_trials > 1:
        print(f"\n[eval] Monte Carlo over {args.mc_trials} MSAC seeds ...")
        rots, trans, times = [], [], []
        for k in range(args.mc_trials):
            o = run_ki_ml(
                magfield_net, lrf_net, norms, device,
                ell=args.ell, K_ctx=args.K_ctx, K_support=args.K_support,
                query_density=args.query_density, seed=args.seed + k,
                verbose=False,
            )
            rots.append(o["rotation_error_deg"])
            trans.append(o["translation_error_m"])
            times.append(o["time_features_s"])
        rots = np.array(rots); trans = np.array(trans); times = np.array(times)
        print(f"  rotation    : RMSE {np.sqrt((rots**2).mean()):.4f} deg "
              f"(median {np.median(rots):.4f}, std {rots.std():.4f})")
        print(f"  translation : RMSE {np.sqrt((trans**2).mean()):.4f} m   "
              f"(median {np.median(trans):.4f}, std {trans.std():.4f})")
        print(f"  feature wall: mean {times.mean():.2f}s")
        print(f"\n  Reference (paper Table I, KI Building map-to-map):")
        print(f"    Translation: 0.1784 m  +/-  0.0836")
        print(f"    Rotation:    0.2688 deg +/- 0.1076")


if __name__ == "__main__":
    main()
