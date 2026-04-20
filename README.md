# TIME Simulations

Simulation code accompanying a paper currently under review. These simulations illustrate how real-world noise sources — circadian heterogeneity, nonadherence, and drug half-life — can mask chronotherapeutic effects in clinical trials, using hypertension as a motivating example.

## Background

Chronotherapy proposes that the time of day a drug is taken affects its efficacy, because many drug targets are under circadian control. In practice, detecting AM vs. PM dosing differences is difficult: patients have different circadian phases (DLMO heterogeneity), don't always adhere to their assigned dosing time, and drugs with long half-lives average out time-of-day sensitivity. These simulations attempt to quantify the effects of these factors on observed effect size (and needed statistical power).

Sample sizes and nonadherence rates are drawn from the TIME study:
- AM arm: N = 9,849
- PM arm: N = 9,537
- AM timing nonadherence: 22.5%
- PM timing nonadherence: 39%

> Mackenzie IS, Rogers A, Poulter NR, et al. Cardiovascular outcomes in adults with hypertension with evening versus morning dosing of usual antihypertensives in the UK (TIME study): a prospective, randomised, open-label, blinded-endpoint clinical trial. *Lancet*. 2022;400(10361):1417–1425. https://doi.org/10.1016/S0140-6736(22)01786-X

## Requirements

```
numpy
matplotlib
scipy
```

Install with:
```bash
pip install numpy matplotlib scipy
```

## Usage

Both scripts write all output to an `output/` directory, which is created automatically.

```bash
python main.py
python power_analysis.py
```

## Scripts

### `main.py`

Generates figures illustrating how each noise source changes the observable AM vs. PM efficacy difference. All figures use a normalized efficacy scale (AM = 0.8, PM = 1.0 under perfect conditions).

**Adherence simulations** — three scenarios using the TIME study group sizes:

| Figure | Description |
|---|---|
| `Perfect adherence.png` | Baseline: all subjects dose at their assigned time |
| `Nonadherent to time.png` | 22.5% of AM and 39% of PM subjects switch dosing time partway through the trial |
| `Nonadherent to time and drug.png` | Same switching, plus a 7% efficacy penalty for evening dosing behavior |

Nonadherent subjects are modeled as switching at a random point within the first 25% of the trial; their efficacy is a time-weighted blend of the AM and PM values.

**Circadian efficacy curves** — two hypothetical drug profiles are modeled as sinusoidal functions of wall-clock time, with amplitude and phase fit to an evening-optimal and a midday-optimal pattern. Figures show the curves, AM/PM sampling windows, and efficacy bars for early vs. late chronotypes (±2 hours from population mean DLMO).

| Figure | Description |
|---|---|
| `Evening optimal efficacy curve.png` | Sinusoidal efficacy curve peaking in the evening |
| `Two hypothetical efficacy curves.png` | Evening-optimal and midday-optimal curves overlaid |
| `Homogeneous DLMO Distribution.png` | DLMO distribution for a typical day-worker population (SD = 1.22 h) |
| `Disrupted DLMO Distribution.png` | DLMO distribution for a circadian-disrupted population (SD = 8 h) |
| `Efficacy Homogeneous *.png` | AM/PM efficacy bars sampled from the homogeneous population |
| `Efficacy Disrupted *.png` | AM/PM efficacy bars sampled from the disrupted population |
| `Evening_Optimal_Bar_Plot.png` | Efficacy by chronotype (early/late) for the evening-optimal curve |
| `Midday_Optimal_Bar_Plot.png` | Efficacy by chronotype (early/late) for the midday-optimal curve |

**Prior dosing history** — illustrates how a subject's dosing history before trial enrollment could possibly affect their observed efficacy in the PM arm, as a function of the fraction of pre-trial AM dosing.

| Figure | Description |
|---|---|
| `Prior dosing history.png` | Three scenarios: 0%, 50%, and 75% prior AM dosing history |

**Pharmacokinetic half-life** — a 3-compartment ODE model (drug → oscillatory target → interaction) solved with RK4 integration. Doses are modeled as instantaneous pulses at each hour of the day. The interaction term (drug × target integrated over time) serves as a proxy for efficacy.

| Figure | Description |
|---|---|
| `Effect of half-life on time of day efficacy.png` | Relative efficacy as a function of dosing hour for half-lives of 1.5 h and 12 h |

The short half-life drug (1.5 h) shows a pronounced time-of-day efficacy pattern; the long half-life drug (12 h) shows a flatter profile because residual drug from the prior dose blunts circadian sensitivity.

---

### `power_analysis.py`

Quantifies how each noise source independently affects statistical power, anchored to the original TIME study sample size.

**Calibration assumption:** The TIME study was designed with N ≈ 9,849 per arm at 80% power (α = 0.05, two-sided). Back-calculating from this N and the 20% baseline efficacy gap (Δ = 0.200) implies an outcome standard deviation of σ₀ ≈ 5.01, representing large between-subject variability in the clinical endpoint. Each noise source is then evaluated by how much it shrinks the observable group-mean difference (Δ_observed); required N scales as (Δ₀ / Δ_observed)².

**Scenarios (each evaluated independently):**

| Scenario | Observed Δ | Power at N = 9,849 | N for 80% power |
|---|---|---|---|
| Baseline (perfect adherence) | 20.0% | 80% | 9,850 (1.0×) |
| DLMO heterogeneity (SD = 1.22 h) | 15.6% | 59% | 16,191 (1.6×) |
| Nonadherence to time | 9.2% | 25% | 46,063 (4.7×) |
| Nonadherence + drug loss | 6.5% | 15% | 92,308 (9.4×) |

DLMO heterogeneity uses the evening-optimal sinusoidal efficacy curve from `main.py`: individual circadian phase offsets (drawn from a Normal distribution with SD = 1.22 h, based on a typical day-worker population) shift each subject's effective dosing time, pulling the two group means closer together.

**Output figures:**

| Figure | Description |
|---|---|
| `power_at_original_N.png` | Statistical power of each scenario at the original N = 9,849 |
| `n_needed_per_scenario.png` | N per arm required to restore 80% power under each noise source |

