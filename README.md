# stochastic-viral-dynamics

Exact stochastic simulation of a two-strain within-host viral model with a CTL
response, compared with its deterministic ODE limit and with branching-process
predictions. Work in progress towards a short paper; the roadmap is in
[`PLAN.md`](PLAN.md).

Gaia Grossi and Leonardo Vanni. The project grew out of a Stochastic Processes
course project at Bocconi University (January 2026); the original notebook is kept
for provenance in [`notebooks/`](notebooks/).

## Start here

```bash
make install     # uv venv + requirements      (or: pip install -r requirements-lock.txt -e .)
make test        # unit tests, about 1 min
make bench       # time the compiled simulator in each regime
make figures     # rebuild every figure from the committed CSVs in results/
```

## What is here

| path | content |
|---|---|
| `viraldyn/model.py` | state, 15 reaction channels, stoichiometry, propensities, ODE limit, R0 formulas, parameter presets |
| `viraldyn/ssa.py` | numba-compiled Gillespie direct method with fixed-grid recording, lineage extinction times, early stopping |
| `viraldyn/theory.py` | branching-process extinction and establishment probabilities (two-type generating functions, closed form for one virion) |
| `viraldyn/ensembles.py`, `stats.py`, `plotting.py` | seeded process-pool runner with provenance metadata, Wilson intervals, shared figure style |
| `tests/` | stoichiometry and ODE-network consistency, R0 against the next-generation matrix, exact distributions, agreement with the original pure-Python simulator |
| `scripts/exp_*.py` | one script per experiment; each writes CSVs plus a `.meta.json` with the git commit, seeds and timing (`make exp-<name>`) |
| `scripts/fig*.py` | one script per figure, reading only the committed results (`make figures`) |
| `results/` | committed CSVs that every figure and every number in the paper is read from |
| `paper/` | LaTeX sources |

## Model

State `x = (T, Iw, Im, Vw, Vm, E)`: target cells, cells infected by the wild type
or the mutant, free virions of each strain, effector CTLs. The deterministic model
is `dx/dt = S a(x)` with the same stoichiometry `S` and propensities `a` used by
the stochastic simulation, which is enforced by a test.

```python
from viraldyn import PRESETS, simulate, solve_ode
import numpy as np

p = PRESETS["acute"]
t = np.linspace(0, 60, 1201)
run = simulate(p, x0=[1000, 0, 0, 50, 0, 2], t_grid=t, seed_value=1)
ode = solve_ode(p, [1000, 0, 0, 50, 0, 2], t)
run.X          # (1201, 6) state on the grid
run.t_ext      # first extinction time of the wild-type, mutant and CTL lineages
```

## Licence

MIT. See `CITATION.cff` for how to cite.
