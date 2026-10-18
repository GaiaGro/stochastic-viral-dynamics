"""Fluctuations around the chronic fixed point against system size.

The resistance preset is scaled by Omega (lambda, 1/beta, 1/q, 1/k and the
initial state), which leaves the deterministic dynamics unchanged as densities,
and N realisations are run from the uninfected equilibrium to day 300. Over the
stationary window (days 200 to 300) the pooled mean and standard deviation of
target cells, infected cells, wild-type virions and effectors give a coefficient
of variation per species and size, which is compared with the linear-noise
approximation around the deterministic fixed point (van Kampen): the covariance
solves the Lyapunov equation J C + C J^T + D = 0 and the coefficient of variation
falls like Omega^(-1/2). One realisation at Omega = 1 and one at Omega = 100 are
stored with the deterministic orbit for the phase plane of Figure 3.
Writes results/fluctuations.csv, results/fluctuations_phase.npz and .meta.json.
"""

import csv
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

from viraldyn import PRESETS, scale_system_size, simulate, solve_ode
from viraldyn.ensembles import provenance, run_seeds, write_meta
from viraldyn.model import E, IW, T, VW
from viraldyn.theory import fixed_point, linear_noise_covariance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fluctuations.csv"
PHASE = ROOT / "results" / "fluctuations_phase.npz"
BASE = PRESETS["resistance"]
X0 = [1000, 10, 0, 0, 0, 10]
OMEGA = (1, 3, 10, 30, 100)
N = 100
T_END = 300.0
GRID = np.linspace(0.0, T_END, 601)  # 0.5-day samples
WINDOW = (200.0, 300.0)
SPECIES = {"T": T, "Iw": IW, "Vw": VW, "E": E}
PHASE_OMEGA = (1, 100)
PHASE_GRID = np.linspace(0.0, T_END, 3001)
PHASE_SEED = 7


def scaled(omega):
    return scale_system_size(BASE, omega), [int(v * omega) for v in X0]


def one(seed, p, x0):
    r = simulate(p, x0, GRID, seed_value=seed)
    return r.X[:, list(SPECIES.values())].astype(np.int64), r.extinct


def coefficient_of_variation(samples):
    """Pooled over realisations and times: standard deviation over mean."""
    samples = np.asarray(samples, dtype=float)
    return float(samples.std() / samples.mean()), float(samples.mean()), float(samples.std())


def main():
    started = time.time()
    one(0, *scaled(1))
    rows = []
    seed = 6_000_000
    in_window = (GRID >= WINDOW[0]) & (GRID <= WINDOW[1])
    for omega in OMEGA:
        p, x0 = scaled(omega)
        out = run_seeds(partial(one, p=p, x0=x0), range(seed, seed + N))
        runs = np.stack([o[0] for o in out])  # (N, time, 4)
        n_extinct = sum(o[1] for o in out)
        x_star = fixed_point(p, x0)
        C = linear_noise_covariance(p, x_star)
        for col, (name, idx) in enumerate(SPECIES.items()):
            cv, mean, std = coefficient_of_variation(runs[:, in_window, col])
            rows.append({
                "omega": omega, "species": name, "n": N, "n_extinct": n_extinct,
                "x_star": float(x_star[idx]), "mean_sim": mean, "std_sim": std, "cv_sim": cv,
                "cv_lna": float(np.sqrt(C[idx, idx]) / x_star[idx]), "seed_start": seed,
            })
        seed += N
        print(f"omega {omega} done, {time.time() - started:.0f} s", flush=True)
    phase = {}
    for omega in PHASE_OMEGA:
        p, x0 = scaled(omega)
        r = simulate(p, x0, PHASE_GRID, seed_value=PHASE_SEED)
        phase[f"X_{omega}"] = r.X.astype(np.int64)
        phase[f"ode_{omega}"] = solve_ode(p, x0, PHASE_GRID)
    np.savez_compressed(PHASE, t=PHASE_GRID, omegas=np.array(PHASE_OMEGA), seed=PHASE_SEED, **phase)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_meta(OUT.with_suffix(".meta.json"), provenance(started, script=Path(__file__).name, preset="resistance", omegas=OMEGA, n=N, window=WINDOW, argv=sys.argv))
    print(f"{len(rows)} rows, {time.time() - started:.0f} s")
    for name in SPECIES:
        sel = [r for r in rows if r["species"] == name]
        slope = np.polyfit(np.log([r["omega"] for r in sel]), np.log([r["cv_sim"] for r in sel]), 1)[0]
        print(f"  {name:2s}: CV at omega 1 = {sel[0]['cv_sim']:.3f} (LNA {sel[0]['cv_lna']:.3f}), at omega 100 = {sel[-1]['cv_sim']:.4f} (LNA {sel[-1]['cv_lna']:.4f}); fitted slope {slope:+.3f} (LNA: -0.5)")


if __name__ == "__main__":
    main()
