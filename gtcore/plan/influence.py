"""Influence matrix (section 3 B). Owner: A2 Dose, branch ``plan/influence``.

Must implement ``build_influence(candidates, target, rx_cgy, sk_per_seed_u,
m_opt, oars, oar_limits, engine, kernel, rng_seed) -> InfluenceMatrix``:

- ``D[c, m]`` = total-decay dose [cGy] at nominal S_K at target point ``m``
  from candidate ``c``'s seeds via ``gtcore.dose.engine.dose_at_points``
  (tabulated kernel by default), chunked, float32; the same per OAR set;
- target subsampling to ``m_opt`` points via ``TargetSet.subsample``
  (stratified by area weight), ``target_index`` recorded;
- the section 3 B gate as a unit test: for random selections of 1-8
  candidates, V100 from the row-sum agrees with V100 from
  ``compute_dose_grid`` (exact kernel, 1 mm, sampled at the same points)
  within 0.5 percentage points and D90 within 1 % of rx.

Never edit ``gtcore/dose/engine.py``.
"""
