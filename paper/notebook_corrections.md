# Corrections to the course notebook, for the manuscript

Each entry: what the notebook (`notebooks/original_course_submission.ipynb`) says or
does, what is wrong, what the paper says instead, and where the evidence lives.
Numbers quoted in the paper must come from the files named here.

## 1. "Scenario (B): Establishment of Chronic Infection"

The heading of the acute scenario is a copy of the chronic one. Scenario B is the
acute regime: the infection is cleared. Figure 2 of the paper labels it
"Acute infection: clearance" (`scripts/fig2_deterministic.py`).

## 2. "90% efficacy is typical of Single Drug HIV Therapy (cART)"

cART is combination therapy, so the sentence contradicts itself, and the model has
no pharmacology to support a clinical claim. The paper treats efficacy as a model
parameter and sweeps it across the suppression threshold. That threshold is not
1 - 1/R0 = 0.79, as the notebook's section 4.3 implies, because beta also sits in the
virion-consumption denominator of R0; solving R0(eps) = 1 exactly gives
eps* = 0.798 (`viraldyn.model.suppression_efficacy`). The grid value 0.8 therefore
lies 1 percent below criticality (R0 = 0.99), which the paper states
(`scripts/exp_resistance.py`, `results/resistance.csv`).

## 3. "We assume the mutant exists already before the arrival of the drug"

The initial state of every notebook simulation has no mutant; mutants arise by
mutation during wild-type production (channel 4). The paper keeps that mechanism and
uses "pre-existing" only for a mutant lineage present at the moment treatment starts,
which is a random event whose probability depends on when treatment starts: about
0.3 at day 5, just after the primary peak, and 0.11 to 0.18 from day 10 on in the
resistance preset (`results/resistance.csv`, column `n_pre_existing`).

## 4. "Probability of Infection Establishment: 4.9 percent established"

With those parameters R0 = 33 and a single virion establishes with probability
pi (1 - 1/R0) = 0.81; the deterministic solution itself clears the infection by
day 2 through effector expansion. The 95 percent labelled "stochastic extinction"
were cleared by the immune response like the ODE; the 5 percent labelled
"established" are realisations in which the single initial effector cell died
before dividing. The paper presents this as stochastic failure of immune
activation, with Kendall's formula as the theory: P(failure | established) =
0.051 [0.041, 0.063], theory 0.045 (`results/immune_activation.csv`, sweep `E0`,
variant `full`, E0 = 1, d_E = 0.1). Figure 5.

## 5. "Mutant survived 3 times (15.0%)" from 20 realisations

Twenty realisations give an interval of roughly 3 to 38 percent. Replaced by the
1000-realisation grid of `results/resistance.csv`. In addition the realisations
used the chronic preset, whose mutant had been redefined without a fitness cost
(item 8), and most mutant lineages died before treatment started, so the number
measured pre-treatment extinction rather than a treatment bottleneck.

## 6. The regime map was deterministic and its boundary a vertical line

The notebook classified the (beta_w, k) plane with the ODE only and drew the
R0 = 1 line without effectors. Figure 8 gives establishment and persistence
probabilities from exact realisations; the initial effectors shift the threshold
by the factor 1 + k E0 / delta, and persistence has a boundary the deterministic
model lacks (`results/regime_map.csv`).

## 7. Deterministic chronic states of one or two infected cells

The immune activation threshold d_E / q is also the chronic set point of infected
cells. In the therapy preset that is 2 cells and in the establishment preset 1 cell.
Such states exist only for continuous variables: the therapy preset's deterministic
trajectory passes through 3e-8 virions at day 17 and recovers, while 0 of 1000
exact realisations survive to day 100 (`results/preset_viability.csv`); in the
establishment preset with q = 0.1 not one of 125,000 realisations in the regime map
persists to day 200 while the ODE calls 131 cells chronic
(`results/regime_map_notebook_q.csv`). The notebook's deterministic
competitive-release figure rests on such a trajectory. The paper reports this as a
result and uses presets with set points of 50 (resistance, regime map) or 1000
(chronic) infected cells for the stochastic experiments.

## 8. Notebook state: three definitions of `params_chronic`, two of `ode_system`

Cells 10, 15 and 35 each define `params_chronic`; the last sets bm = 2e-5, equal
to bw, so the mutant has no fitness cost from cell 35 on. The thresholds table that
prints R0(Mut) = 30.00 used that definition; with the fitness cost stated in the
text (bm = 1e-5) the value is 25.7. Two `ode_system` definitions differ in whether
they clip negative states. The repository has one preset per name
(`viraldyn/model.py`), one right-hand side dx/dt = S a(x) with documented clipping,
and a test that it equals the six written equations (`tests/test_model.py`).

## 9. Reproducibility

The notebook seeded numpy once at the top, so results depended on execution order.
Every realisation in the repository carries its own seed, recorded with the results.

## 10. The "chronic" scenario starts a thousand times below its own equilibrium

The chronic preset has lambda / d_T = 1e6 target cells at the uninfected
equilibrium, but every notebook run starts from T0 = 1000, where the wild type's
reproduction number is 0.2. The damped oscillations the notebook shows are the
transient of target-cell repopulation (T has reached only 1.3e5 by day 60). Started
from its equilibrium the deterministic infection peaks on day 1 at 1.4e6 and is
cleared, and 0 of 200 exact realisations survive to day 60; from T0 = 1000 all 200
survive (`results/chronic_preset_check.csv`). The paper's chronic scenario is the
resistance preset, started at its equilibrium T* = 1000, and the large-population
check of Figure 3 uses that preset with every population scaled by 100
(`chronic_large`), which keeps its deterministic dynamics exactly
(`scripts/exp_baseline.py`, `results/baseline_validation.csv`).

## 11. The stochastic therapy ensemble showed means

Means of trajectories that are zero in some realisations and oscillating in others
are not representative. Figure 6 shows medians and 5th to 95th percentile bands and
the fraction of realisations in which each strain is present
(`results/competitive_release_ensemble.npz`).
