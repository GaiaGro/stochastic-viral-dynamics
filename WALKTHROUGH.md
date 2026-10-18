# Walkthrough

A guided tour of the repository for the two authors, to read and run before the
paper is written. The quick parts take about twenty minutes; nothing here needs a
GPU. Every command is run from the repository root.

## 1. Set up

```bash
git clone git@github.com:VanniLeonardo/stochastic-viral-dynamics.git
cd stochastic-viral-dynamics
make install          # needs uv (https://docs.astral.sh/uv/); creates .venv
make test             # 81 tests, about 20 s; the first call compiles the simulator (+10 s)
make figures          # rebuilds figures/fig2 ... fig8 from results/, no simulation, about 1 min
```

Without uv:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt -e .
make test PY=.venv/bin/python
```

If `make test` is green and `make figures` fills `figures/`, the environment is
complete. Open the PNGs in `figures/` now; they are the paper's Figures 2 to 8.

## 2. Map of the repository

| path | what it is |
|---|---|
| `viraldyn/model.py` | the model: state, 15 reaction channels, stoichiometry, propensities, ODE `dx/dt = S a(x)`, R0 formulas, exact suppression efficacy, system-size scaling, parameter presets |
| `viraldyn/ssa.py` | the exact simulator (Gillespie direct method, numba-compiled), records on a time grid, tracks lineage extinction times, early stops |
| `viraldyn/theory.py` | branching-process and birth-death formulas: establishment probabilities, Kendall's extinction formula, lineage extinction along a path |
| `viraldyn/ensembles.py`, `stats.py`, `plotting.py` | seeded process-pool runner with provenance, Wilson intervals, shared figure style |
| `scripts/exp_*.py` | one script per experiment; each docstring states the design and the questions it answers; each writes CSVs plus a `.meta.json` |
| `scripts/fig*.py` | one script per figure; they read only `results/`, never simulate |
| `results/` | the committed numbers; every figure and every number in the paper comes from here |
| `tests/` | correctness tests, including agreement with the course notebook's own simulator |
| `PLAN.md` | the roadmap; each work package has a status entry with the final numbers and a "Review" entry with what the review changed |
| `paper/notebook_corrections.md` | eleven things the course notebook got wrong, each with the corrected statement and the results file that proves it |
| `notebooks/original_course_submission.ipynb` | the notebook, kept for provenance only |

## 3. The model on one screen

State `x = (T, Iw, Im, Vw, Vm, E)`: target cells, cells infected by the wild type
or the mutant, free virions of each strain, effector (CTL) cells. Fifteen channels:
infection by each strain (a virion is consumed), virion production by each
infected-cell type (a wild-type cell emits a mutant virion with probability `mu`),
target-cell production and death, virion clearance, infected-cell death, killing
of infected cells by effectors, effector proliferation at rate `q (Iw + Im) E` and
effector death. The drug multiplies wild-type infection by `1 - eps_b`.

Two derived quantities organise everything:

- `R0 = p (1 - mu) / delta * beta T* / (c + beta T*)`, with `T* = lambda / d_T` the
  uninfected equilibrium. The last factor is the probability that a virion infects
  before it is cleared; because `beta` is also in its denominator, the efficacy at
  which `R0 = 1` is `eps* = 0.798`, not `1 - 1/R0 = 0.79` (`suppression_efficacy`).
- `I* = d_E / q`: the infected-cell load above which effectors grow, and therefore
  also the chronic set point of infected cells whenever the response is active.
  A set point of one or two cells is a continuous-variable artefact; this single
  fact explains three of the notebook's results.

Presets (`viraldyn/model.py`, bottom):

| preset | T* | R0 wild type / mutant | I* | used for |
|---|---|---|---|---|
| `establishment` | 1000 | 33 / 32 | 1 (q = 0.1) | Figures 4 and 5; Figure 8 with q lowered to 0.002 (I* = 50) |
| `acute` | 1000 | 7.4 / 7.4 | 2 | Figure 2b |
| `therapy` | 1000 | 4.8 / 3.9 | 2 | the notebook's; shown only as not viable (`preset_viability.csv`) |
| `resistance` | 1000 | 4.8 / 3.9 | 50 | Figures 2a, 6, 7 |
| `chronic_large` | 1e5 | 4.8 / 3.9 | 5000 | Figure 3 (resistance scaled by 100, same dynamics as densities) |
| `chronic` | 1e6 | 33 / 26 | 1000 | the notebook's; not viable from its own equilibrium (`chronic_preset_check.csv`), kept for provenance and simulator tests |

Every experiment starts at the uninfected equilibrium `T0 = T*` of its preset.

## 4. Reading order

1. `PLAN.md`, sections 0 and 3: the framing, then each work package's status and
   review entries. This is the paper's skeleton with numbers attached.
2. `paper/notebook_corrections.md`: what we are correcting and where the proof is.
3. The docstring at the top of each `scripts/exp_*.py`: the design decisions.
4. The CSVs, with the table in section 6 as a key.

## 5. Figures, scripts and the files behind them

| figure | script | reads | produced by |
|---|---|---|---|
| 2 deterministic regimes | `fig2_deterministic.py` | nothing (ODE only) | |
| 3 simulator against ODE, phase planes, fluctuation size | `fig3_validation.py` | `baseline_ensemble.npz`, `fluctuations_phase.npz`, `fluctuations.csv` | `exp_baseline.py`, `exp_fluctuations.py` |
| 4 establishment | `fig4_establishment.py` | `establishment.csv` | `exp_establishment.py` |
| 5 immune activation failure | `fig5_immune_activation.py` | `immune_activation.csv`, `immune_activation_trajectories.npz` | `exp_immune_activation.py` |
| 6 competitive release | `fig6_competitive_release.py` | `competitive_release_ensemble.npz` | `exp_resistance.py` |
| 7 resistance emergence | `fig7_resistance.py` | `resistance.csv`, `resistance_system_size.csv` | `exp_resistance.py` |
| 8 regime map | `fig8_regime_map.py` | `regime_map.csv` | `exp_regime_map.py` |
| appendix: the notebook's regime map | `fig9_notebook_regime.py` | `regime_map_notebook_q.csv` | `exp_regime_map.py` |

## 6. Inspecting the numbers without running anything

A CSV is one row per condition. To look at one:

```bash
column -s, -t < results/resistance.csv | less -S        # scroll with arrows, q to quit
.venv/bin/python -c "import csv; [print(r['t_treat'], r['eps'], r['p_emerge'], r['p_theory_cond']) for r in csv.DictReader(open('results/resistance.csv'))]"
```

What each file answers, and its key columns:

| file | question | key columns |
|---|---|---|
| `establishment.csv` | P(establish) vs inoculum and R0, against branching theory | `V0`, `E0`, `R0_wt`, `p_hat`, `wilson_low/high`, `p_theory` |
| `establishment_threshold.csv` | does the 100-infection threshold matter | `p_reach_25` ... `p_reach_400` (identical) |
| `immune_activation.csv` | P(activation fails) vs E0, d_E, q; full model and k = 0 variant | `sweep`, `variant`, `p_fail`, `p_fail_early`, `p_theory`, `p_theory_per_path`, `n_*` fates |
| `immune_collapse.csv` | what happens after a collapsed response; the peak-effector gap | `early_*`, `collapse_*`, `n_fail_peakE_*` |
| `resistance.csv` | P(resistance emerges) per treatment start and efficacy | `p_emerge`, `wilson_*`, `p_theory`, `p_theory_cond`, `n_cure`, `n_wt_survived_window`, `p_emerge_threshold_*` |
| `resistance_runs.csv` | one row per realisation behind the grid | `pre`, `emerged`, `mut_infections`, `t_wt_ext`, `p_theory_cond` |
| `resistance_system_size.csv` | the same cell at Omega 1 to 300 | `omega`, `p_emerge`, `p_theory_cond`, `n_cure` |
| `resistance_preset_sensitivity.csv` | does the choice of k and q matter | `k`, `q`, `p_emerge`, `n_cure`, `n` (0 = not viable) |
| `resistance_horizon.csv` | efficacy 0.5 followed to day 360 | `horizon`, `p_emerge` |
| `preset_viability.csv` | notebook's therapy preset vs ours | `ode_trough_load`, `p_alive_day100` |
| `regime_map.csv` | P(establish), P(persist at day 200) over beta and k | `bw`, `k`, `R_eff`, `p_est`, `p_persist`, `p_est_theory`, `ode_class` |
| `regime_map_notebook_q.csv` | the same map with the notebook's q (nothing persists) | as above |
| `regime_map_horizon.csv` | boundary cells followed to day 1000 | `p_persist_200`, `p_persist_1000` |
| `baseline_validation.csv` | does the simulator track the ODE at large populations | `max_abs_log10_dev_median`, `coverage_5_95` |
| `fluctuations.csv` | how fluctuations around the chronic fixed point shrink with system size, against linear-noise theory | `omega`, `species`, `cv_sim`, `cv_lna` |
| `chronic_preset_check.csv` | the notebook's chronic preset from T0 = 1000 and from T* | `R_eff_at_start`, `n_alive` |

Each `*.meta.json` next to a CSV records the git commit, whether the working tree
was clean, the seeds, the parameters and the runtime.

## 7. Reproducing

Everything is seeded, so a rerun gives the same CSV byte for byte; only the
`.meta.json` files change (timestamp, commit, runtime).

| command | writes | time on 16 cores |
|---|---|---|
| `make exp-establishment` | `establishment*.csv` | 5 s |
| `make exp-immune` | `immune_activation.*` | 15 s |
| `make exp-immune-collapse` | `immune_collapse.csv` (needs the previous one) | 5 s |
| `make exp-baseline` | `baseline_*`, `chronic_preset_check.csv` | 80 s |
| `make exp-fluctuations` | `fluctuations.csv`, `fluctuations_phase.npz` | 2 min |
| `make exp-regime-map` | `regime_map*.csv` | 9 min |
| `make exp-resistance` | `resistance*.csv`, `preset_viability.csv`, Figure 6 ensemble | 42 min |

After any rerun:

```bash
git status --short results/     # should list only *.meta.json
git diff --stat results/        # confirms no CSV changed
```

## 8. Poking at the model

```python
from viraldyn import PRESETS, simulate, solve_ode, r0_wild_type, suppression_efficacy
import numpy as np

p = PRESETS["resistance"]
print(r0_wild_type(p), suppression_efficacy(p))          # 4.76, 0.798

t = np.linspace(0, 150, 1501)
ode = solve_ode(p, [1000, 10, 0, 0, 0, 10], t)           # columns T, Iw, Im, Vw, Vm, E
run = simulate(p, [1000, 10, 0, 0, 0, 10], t, seed_value=1)
print(ode[-1].round(1)); print(run.x_end)                 # deterministic vs one exact realisation at day 150
print(run.t_ext)                                           # first extinction time of wild-type, mutant, effector lineages

from viraldyn.theory import establishment_probability, immune_failure_probability
pe = PRESETS["establishment"]
print(establishment_probability(pe.replace(k=0), [1000, 0, 0, 1, 0, 0]))   # 0.808 = pi (1 - 1/R0)
print(immune_failure_probability(pe, [1000, 0, 0, 1, 0, 1]))                # 0.045, the old "4.9 percent"
```

Run it with `.venv/bin/python -i` and change numbers. `simulate` takes
`stop_on_extinction`, `stop_cum_infections` and `stop_above` for early stops; the
grid you pass is where the state is recorded.

## 9. A review checklist

- Pick three numbers in a `PLAN.md` status entry and find them in the CSV. If one
  does not match, that is a finding; two were wrong before the review.
- For each figure, confirm the script reads only `results/` (search for `simulate`
  in `scripts/fig*.py`: there should be none).
- Read the docstring of `exp_resistance.py` and ask whether each constant at the
  top is justified in `PLAN.md` or by a sensitivity CSV. The same for
  `exp_regime_map.py` and `exp_immune_activation.py`.
- Known open points, to be stated rather than hidden in the paper: the regime map
  is a day-200 statement (persistence drops by 0.035 on average by day 1000); at
  efficacy 0.5 the mutant replaces the wild type over months, so the 120-day numbers
  are a lower bound and the two theory variants bracket the 360-day value; the
  lineage theories use the deterministic environment of the wild type alone and run
  a few percent low where the realised load grows faster than the ODE.

## 10. Rules we kept

- No number in the paper without a CSV in `results/` and the script that wrote it.
- Every realisation has its own seed, recorded with the results.
- Figures use medians and 5th to 95th percentile bands, never means of trajectories.
- Wild type is blue, mutant orange, effectors aqua, target cells gray, deterministic
  black; ordered levels use the blue ramp (`viraldyn/plotting.py`).
- CI runs the tests and rebuilds every figure from the committed CSVs on each push.
