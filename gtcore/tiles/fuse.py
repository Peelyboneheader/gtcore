"""Posterior seed positions of the hierarchical bent-tile model (plan stage 5).

Model (one random-effects least-squares problem, not a post-hoc blend)::

    x_i = s_i + e_i,           e_i ~ N(0, Sigma_i)      localization error
    s_i = m_i(theta) + d_i,    d_i ~ N(0, Sigma_s)      seed off the ideal sheet
    Sigma_s = slack^2 I

``x_i`` is the detected centre, ``s_i`` the true seed, ``m_i(theta)`` the
bent-tile model's seed ``i`` (:mod:`gtcore.tiles.deform`, 9 parameters).
Eliminating ``s_i`` gives the weighted bent-tile fit
(``fit_deformable(..., seed_cov=Sigma)``, weights ``(Sigma_i + Sigma_s)^-1``);
given its estimate ``theta_hat``, the conditional (posterior) seed is

    s_hat_i = (Sigma_i^-1 + Sigma_s^-1)^-1 (Sigma_i^-1 x_i + Sigma_s^-1 m_i)
            = x_i + K_i (m_i - x_i),       K_i = Sigma_i (Sigma_i + Sigma_s)^-1
    Cov     = (Sigma_i^-1 + Sigma_s^-1)^-1 = (I - K_i) Sigma_i

(the Kalman form is used: it needs no inverse of ``Sigma_i``, which may be
near-singular in-plane).  This is the best linear unbiased predictor of the
random effect in Henderson's mixed-model sense with ``theta`` plugged in.
The default covariance is the conditional one above, which does NOT carry
the uncertainty of ``theta_hat`` (fitted to the same seeds);
``cov_mode="pev"`` returns the linearised prediction-error covariance that
does (``s_hat - s = (I - K + K G L) e - K (I - G L) d`` with ``G`` the model
Jacobian and ``L`` the Gauss-Newton sensitivity of the fit).  Calibration
of both is checked with the NEES test, not assumed
(docs/localization-notes.md, stage 5: the conditional covariance is
over-confident on thick slices, the PEV tracks the input covariance).

Only 3 of a full tile's 12 seed coordinates are redundant against the 9
model parameters (plus the axis and curvature terms), so the gain is
bounded and must be measured; on an isotropic thin-slice scan the pull is
small in every direction, on a thick-slice scan it acts mainly along z.

Tentative tiles (cover pass, triplet completion) and degraded tiles (loose
tier / count completion) use a larger slack (``slack_tentative_mm``): the
grouping itself is less certain, so the model is trusted less.  Seeds not
owned by a fused tile pass through unchanged.

Pure numpy: ~50 us per seed including the per-tile Python overhead
(0.6 ms for a 12-seed scan), ~0.2 ms per seed with ``cov_mode="pev"``.
"""
from __future__ import annotations

from itertools import permutations
from typing import Optional

import numpy as np

from . import deform as _deform
from .deform import SLACK_MM, seed_whitening
from .model import RigidTile

__all__ = ["posterior_seed_positions", "slab_covariance_stand_in",
           "SLACK_MM", "SLACK_TENTATIVE_MM"]

SLACK_TENTATIVE_MM = 1.0


def slab_covariance_stand_in(affine, n, noise_frac=0.1):
    """Analytic stand-in per-seed covariance, ``(n, 3, 3)`` RAS mm^2.

    Diagonal over the VOXEL axes: ``s_a^2 / 12`` (uniform position inside
    one voxel/slab of size ``s_a``) plus ``(noise_frac * s_a)^2``, rotated
    into RAS by the direction cosines -- i.e. ``M diag(c) M^T`` with
    ``M = affine[:3, :3]`` and ``c = 1/12 + noise_frac^2``.  Used by the
    stage-5 tests and validation until the grey-level centroid covariance
    of stage 2 (``gtcore.seeds.refine``) is available; it is the same for
    every seed of a scan.
    """
    M = np.asarray(affine, dtype=float)[:3, :3]
    c = 1.0 / 12.0 + float(noise_frac) ** 2
    cov = c * (M @ M.T)
    return np.repeat(cov[None, :, :], int(n), axis=0)


def _poses(result):
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return list(result)
    if hasattr(result, "all_tiles"):
        return list(result.all_tiles)
    return list(getattr(result, "tiles", []))


def _match(fit, x):
    """Model points ``m_i`` matched to observed seeds ``x`` (fit order).

    Uses the correspondence stored on the fit (``fit.assignment``: observed
    ``i`` <-> canonical model seed ``assignment[i]``) when it reproduces
    the fit's own residuals, which holds whenever the fit was computed from
    these centres in this order (every fit in :mod:`gtcore.tiles.auto`).
    Otherwise -- e.g. a fit computed elsewhere -- the minimum total squared
    distance assignment over all permutations (k <= 4 seeds, <= 24
    candidates: the Hungarian optimum by enumeration)."""
    model = np.asarray(fit.seed_points(), dtype=float)
    k = x.shape[0]
    asg = tuple(int(a) for a in getattr(fit, "assignment", ()))[:k]
    res = getattr(fit, "residuals_mm", None)
    if (len(asg) == k and len(set(asg)) == k
            and all(0 <= a < len(model) for a in asg)):
        m = model[list(asg)]
        if res is None or len(res) < k or np.allclose(
                np.linalg.norm(x - m, axis=1), np.asarray(res)[:k],
                atol=1e-6, rtol=0.0):
            return m, "assignment"
    best, best_cost = None, np.inf
    for perm in permutations(range(len(model)), k):
        cost = float(((x - model[list(perm)]) ** 2).sum())
        if cost < best_cost:
            best, best_cost = perm, cost
    return model[list(best)], "hungarian"


def _fit_sensitivity(fit, n_obs, W):
    """Linearised model Jacobian ``G = dm/dtheta`` (3k x p) at the fitted
    parameters and the Gauss-Newton sensitivity ``L = dtheta/dx`` (p x 3k)
    of the bent-tile fit to the ``n_obs`` observed seeds, with position
    weights ``W`` (k, 3, 3) (identity for an unweighted fit) and the
    curvature prior; the axis term is left out (it is absent on coarse
    weighted fits, and leaving it out can only overstate the parameter
    uncertainty).  ``theta = (rotation vector about the fitted R, t,
    kappa1, kappa2, psi)``; 2-seed (rigid) fits use the 6 pose parameters.
    """
    R = np.asarray(fit.pose.R, dtype=float)
    pr = fit.params
    uv = RigidTile(fit.pose.kind).seed_uv[list(fit.assignment)[:n_obs]]

    def local(k1, k2, psi):
        return _deform._sheet_and_tangent(uv, k1, k2, psi)[0]

    p0 = local(pr.kappa1, pr.kappa2, pr.psi)
    k = n_obs
    cols = []
    for j in range(3):                       # d(R rod(w) p)/dw at w = 0
        e = np.zeros(3)
        e[j] = 1.0
        cols.append((np.cross(e, p0) @ R.T).ravel())
    for j in range(3):                       # dt
        e = np.zeros((k, 3))
        e[:, j] = 1.0
        cols.append(e.ravel())
    bend = len(fit.residuals_mm) > 2
    if bend:
        h = 1e-6
        base = np.array([pr.kappa1, pr.kappa2, pr.psi])
        for j in range(3):
            dp = np.zeros(3)
            dp[j] = h
            d = (local(*(base + dp)) - local(*(base - dp))) / (2.0 * h)
            cols.append((d @ R.T).ravel())
    G = np.stack(cols, axis=1)               # (3k, p)
    Wb = np.zeros((3 * k, 3 * k))
    for i in range(k):
        Wb[3 * i:3 * i + 3, 3 * i:3 * i + 3] = W[i]
    WtW = Wb.T @ Wb
    JtJ = G.T @ WtW @ G
    if bend:
        wb = _deform._W_BEND_MM_MM
        JtJ[6, 6] += wb ** 2
        JtJ[7, 7] += wb ** 2
    L = np.linalg.pinv(JtJ, rcond=1e-10) @ G.T @ WtW
    return G, L


def _pev_cov(Sigma, K, slack_mm, G, L):
    """Per-seed blocks of the prediction-error covariance of
    ``s_hat = x + K (m(theta_hat) - x)`` with ``theta_hat`` re-estimated
    from the same ``x`` (linearised): ``s_hat - s = (I - K + K G L) e
    - K (I - G L) d``."""
    k = Sigma.shape[0]
    n = 3 * k
    Kb = np.zeros((n, n))
    Sb = np.zeros((n, n))
    for i in range(k):
        Kb[3 * i:3 * i + 3, 3 * i:3 * i + 3] = K[i]
        Sb[3 * i:3 * i + 3, 3 * i:3 * i + 3] = Sigma[i]
    GL = G @ L
    I = np.eye(n)
    Ae = I - Kb + Kb @ GL
    Ad = Kb @ (I - GL)
    C = Ae @ Sb @ Ae.T + float(slack_mm) ** 2 * (Ad @ Ad.T)
    C = 0.5 * (C + C.T)
    return np.stack([C[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(k)])


def _posterior(x, Sigma, m, slack_mm):
    """Vectorised conditional mean / covariance (module docstring)."""
    S = Sigma + float(slack_mm) ** 2 * np.eye(3)[None, :, :]
    K = np.transpose(np.linalg.solve(S, Sigma), (0, 2, 1))   # Sigma S^-1
    s_hat = x + np.einsum("kij,kj->ki", K, m - x)
    P = Sigma - K @ Sigma
    P = 0.5 * (P + np.transpose(P, (0, 2, 1)))
    return s_hat, P, K


def posterior_seed_positions(result, centers_ras, cov_ras,
                             slack_mm: float = SLACK_MM,
                             slack_tentative_mm: float = SLACK_TENTATIVE_MM,
                             tile_slack: Optional[dict] = None,
                             cov_mode: str = "conditional"):
    """Posterior seed centres and covariances given the fitted tiles.

    Parameters
    ----------
    result : AutoFitResult / TileFitResult / list of TilePose
        Selected tiles; every tile (supported and tentative) carrying a
        bent-tile fit (``pose.deform``) is fused.  For the hierarchical
        model the fit must come from the SAME covariances
        (``fit_tiles_auto(..., seed_cov=cov_ras)``); with an unweighted fit
        the result is still the conditional mean given that fit.
    centers_ras : (N, 3)
        Detected (raw) seed centres, indexed like ``pose.seed_indices``.
    cov_ras : (N, 3, 3)
        Their localization covariances ``Sigma_i`` (mm^2).
    slack_mm, slack_tentative_mm : float
        ``Sigma_s = slack^2 I`` for supported tiles and for tentative or
        degraded ones.
    tile_slack : dict, optional
        ``{tile_id: slack_mm}`` overrides (sensitivity studies).
    cov_mode : ``"conditional"`` (default) or ``"pev"``
        ``"conditional"`` returns ``(Sigma_i^-1 + Sigma_s^-1)^-1``, exact
        given ``theta_hat``.  ``"pev"`` returns the linearised prediction-
        error covariance that also carries the uncertainty of
        ``theta_hat`` re-estimated from the same seeds (Henderson's PEV of
        the predicted random effect): with no redundancy (a triplet, a
        half tile) it equals the prior, with 3 redundant DOF it lies
        between the two.  The posterior MEAN is the same in both modes.

    Returns
    -------
    (centers_post (N, 3), cov_post (N, 3, 3), info)
        ``info`` holds per-seed arrays ``tile_of`` (-1 = passed through),
        ``slack_used`` (nan if passed through), ``model_ras``, ``shift_ras``,
        ``shift_mm``, ``var_ratio`` (trace posterior / trace prior), the
        per-tile summaries ``tiles`` and the counts ``n_fused`` /
        ``n_passthrough``.
    """
    x_all = np.asarray(centers_ras, dtype=float).reshape(-1, 3)
    n = x_all.shape[0]
    cov = np.asarray(cov_ras, dtype=float)
    if cov.shape != (n, 3, 3):
        raise ValueError("cov_ras must be (%d, 3, 3), got %r" % (n, cov.shape))
    if cov_mode not in ("conditional", "pev"):
        raise ValueError("cov_mode must be 'conditional' or 'pev'")
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    post = x_all.copy()
    post_cov = cov.copy()
    tile_of = np.full(n, -1, dtype=int)
    slack_used = np.full(n, np.nan)
    model_ras = np.full((n, 3), np.nan)
    tiles = []
    for pose in _poses(result):
        idx = [int(i) for i in pose.seed_indices]
        entry = dict(tile_id=int(pose.tile_id), kind=pose.kind,
                     confidence=getattr(pose, "confidence", "supported"),
                     degraded=bool(getattr(pose, "degraded", False)),
                     seed_indices=idx,
                     inferred_seed=getattr(pose, "inferred_seed_ras", None)
                     is not None)
        fit = getattr(pose, "deform", None)
        if fit is None or not idx:
            entry["skipped"] = "no bent-tile fit"
            tiles.append(entry)
            continue
        clash = [i for i in idx if tile_of[i] >= 0]
        if clash:
            entry["skipped"] = "seeds %s already fused by another tile" % clash
            tiles.append(entry)
            continue
        lenient = entry["confidence"] == "tentative" or entry["degraded"]
        s = float(slack_tentative_mm if lenient else slack_mm)
        if tile_slack and int(pose.tile_id) in tile_slack:
            s = float(tile_slack[int(pose.tile_id)])
        x = x_all[idx]
        m, how = _match(fit, x)
        s_hat, P, K = _posterior(x, cov[idx], m, s)
        if cov_mode == "pev":
            if getattr(fit, "weighted_residuals_mm", None) is not None:
                W = seed_whitening(cov[idx], slack_mm)[0]
            else:
                W = np.repeat(np.eye(3)[None], len(idx), axis=0)
            if how == "assignment":
                G, L = _fit_sensitivity(fit, len(idx), W)
                P = _pev_cov(cov[idx], K, s, G, L)
            else:
                P = cov[idx].copy()      # no trusted correspondence: no gain
        post[idx] = s_hat
        post_cov[idx] = P
        tile_of[idx] = int(pose.tile_id)
        slack_used[idx] = s
        model_ras[idx] = m
        sh = np.linalg.norm(s_hat - x, axis=1)
        entry.update(
            slack_mm=s, match=how, n_seeds=len(idx),
            mean_shift_mm=float(sh.mean()), max_shift_mm=float(sh.max()),
            rms_mm=float(fit.rms_mm),
            wrms_mm=float(getattr(fit, "wrms_mm", fit.rms_mm)),
            chi_rms=getattr(fit, "chi_rms", None),
            bending_energy=float(fit.bending_energy))
        tiles.append(entry)
    shift = post - x_all
    tr0 = np.trace(cov, axis1=1, axis2=2)
    tr1 = np.trace(post_cov, axis1=1, axis2=2)
    var_ratio = np.where(tr0 > 0, tr1 / np.where(tr0 > 0, tr0, 1.0), 1.0)
    info = dict(slack_mm=float(slack_mm), cov_mode=cov_mode,
                slack_tentative_mm=float(slack_tentative_mm),
                tile_of=tile_of, slack_used=slack_used, model_ras=model_ras,
                shift_ras=shift, shift_mm=np.linalg.norm(shift, axis=1),
                var_ratio=var_ratio, tiles=tiles,
                n_fused=int((tile_of >= 0).sum()),
                n_passthrough=int((tile_of < 0).sum()))
    return post, post_cov, info
