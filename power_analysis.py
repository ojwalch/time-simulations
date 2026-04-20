"""
Power analysis anchored to the original trial N (9849 AM / 9537 PM).
Each noise source is evaluated INDEPENDENTLY relative to the baseline.

Key calibration:
  The original trial was designed at 80% power with N=9849 per arm.
  This implies a baseline Cohen's d = sqrt(2*(z_alpha+z_beta)^2 / N) = 0.040.
  With the true AM/PM efficacy gap Δ=0.200, the implied outcome SD is σ_0 = 0.200/0.040 = 5.01
  (representing large between-subject variability in the clinical outcome).

  Each noise source primarily SHRINKS the observed group-mean difference (Δ_observed).
  Because σ barely changes across scenarios, N scales as (Δ_original / Δ_observed)^2.

Scenarios (one at a time, each independent of the others):
  1. Baseline             - flat efficacy, perfect adherence
  2. DLMO heterogeneity   - typical workers (SD=1.22 h)
  3. Nonadherence to time - 22.5% AM / 39% PM switching, no DLMO shift
  4. Nonadherence + drug  - adds 7% efficacy penalty for evening dosing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# ── Original trial parameters ────────────────────────────────────────────────
N_ORIGINAL    = 9849          # original AM arm size
ALPHA         = 0.05
TARGET_POWER  = 0.80
Z_ALPHA_HALF  = norm.ppf(1 - ALPHA / 2)   # 1.960
Z_BETA        = norm.ppf(TARGET_POWER)     # 0.842

# Baseline efficacy gap (AM=0.8, PM=1.0 → Δ=0.200)
AM_EFFICACY = 0.8
PM_EFFICACY = 1.0
DELTA_0     = PM_EFFICACY - AM_EFFICACY    # 0.200

# Implied baseline outcome SD (back-calculated from N_ORIGINAL → 80% power)
D_ORIGINAL = np.sqrt(2 * (Z_ALPHA_HALF + Z_BETA)**2 / N_ORIGINAL)
SIGMA_0    = DELTA_0 / D_ORIGINAL  # ≈ 5.01 (e.g. SBP variability in clinical units)

# ── Circadian/trial parameters ───────────────────────────────────────────────
AM_TIME  = 8.0
PM_TIME  = 22.0
MORNING_NONADHERENCE = 0.225
EVENING_NONADHERENCE = 0.39
DRUG_LOSS = 0.07

# Evening-optimal circadian curve (from main.py)
AMPLITUDE      = 0.0857
VERTICAL_SHIFT = 0.9
PHASE_SHIFT    = 15.5

def efficacy_curve(x):
    return AMPLITUDE * np.sin(2 * np.pi / 24 * (x - PHASE_SHIFT)) + VERTICAL_SHIFT


# ── Simulation functions (each independent / no baseline noise layer) ────────

def sim_baseline(n):
    """Perfect adherence, flat efficacy — the original trial assumption."""
    return (np.full(n, AM_EFFICACY),
            np.full(n, PM_EFFICACY))


def sim_dlmo(n, dlmo_sd):
    """DLMO heterogeneity only: circadian phase spread shifts effective dosing time."""
    am = efficacy_curve(AM_TIME + np.random.normal(0, dlmo_sd, n))
    pm = efficacy_curve(PM_TIME + np.random.normal(0, dlmo_sd, n))
    return am, pm


def sim_nonadherence(n):
    """Nonadherence to timing only (flat AM/PM efficacy, no DLMO shift)."""
    change_pts = 0.25 * np.random.rand(n)

    am_switch = np.random.rand(n) < MORNING_NONADHERENCE
    am_blend  = AM_EFFICACY * change_pts + PM_EFFICACY * (1 - change_pts)
    am = np.where(am_switch, am_blend, AM_EFFICACY)

    pm_switch = np.random.rand(n) < EVENING_NONADHERENCE
    pm_blend  = PM_EFFICACY * change_pts + AM_EFFICACY * (1 - change_pts)
    pm = np.where(pm_switch, pm_blend, PM_EFFICACY)

    return am, pm


def sim_nonadherence_and_drug(n):
    """Nonadherence + 7% evening-dosing penalty (matches main.py logic)."""
    change_pts = 0.25 * np.random.rand(n)

    am_switch = np.random.rand(n) < MORNING_NONADHERENCE
    am_blend  = AM_EFFICACY * change_pts + PM_EFFICACY * (1 - change_pts) - DRUG_LOSS
    am = np.where(am_switch, am_blend, AM_EFFICACY)

    pm_switch  = np.random.rand(n) < EVENING_NONADHERENCE
    pm_blend   = PM_EFFICACY * change_pts + AM_EFFICACY * (1 - change_pts)  # switchers avoid penalty
    pm_adhere  = PM_EFFICACY - DRUG_LOSS                                     # adherent PM still dose in evening
    pm = np.where(pm_switch, pm_blend, pm_adhere)

    return am, pm


# ── Analytical power & N formulas ────────────────────────────────────────────

def analytical_power(delta_obs, n=N_ORIGINAL):
    """Power of t-test with n per arm, given observed mean gap delta_obs and σ_0."""
    d = delta_obs / SIGMA_0
    return float(norm.cdf(np.sqrt(n / 2) * d - Z_ALPHA_HALF))


def n_for_80pct(delta_obs):
    """N per arm needed for 80% power given an observed mean gap delta_obs."""
    d = delta_obs / SIGMA_0
    if d <= 0:
        return np.inf
    return int(np.ceil(2 * (Z_ALPHA_HALF + Z_BETA)**2 / d**2))


# ── Estimate observed Δ for each scenario ────────────────────────────────────

N_EST = 200_000  # large n for accurate mean estimation

scenarios = [
    ("Baseline\n(perfect adherence)",          sim_baseline,              {}),
    ("DLMO heterogeneity\n(typical, SD=1.22h)", sim_dlmo,                 {"dlmo_sd": 1.22}),
    ("Nonadherence\nto time",                   sim_nonadherence,          {}),
    ("Nonadherence\n+ drug loss",               sim_nonadherence_and_drug, {}),
]

print(f"Baseline:  Δ_0 = {DELTA_0:.3f},  σ_0 = {SIGMA_0:.3f},  d_0 = {D_ORIGINAL:.4f}")
print(f"Original N = {N_ORIGINAL:,}  → designed for exactly 80% power\n")
print(f"{'Scenario':<42} {'Δ_obs':>7} {'% of Δ_0':>9} {'Power@N_orig':>13} {'N for 80%':>11} {'×original':>10}")
print("─" * 95)

results = []
for label, fn, kwargs in scenarios:
    am, pm = fn(N_EST, **kwargs)
    delta   = float(np.mean(pm) - np.mean(am))
    pct     = delta / DELTA_0
    power   = analytical_power(delta)
    n_req   = n_for_80pct(delta)
    mult    = n_req / N_ORIGINAL
    results.append((label, delta, pct, power, n_req, mult))
    print(f"{label.replace(chr(10),' '):<42} {delta:>7.4f} {pct:>8.1%}  {power:>12.1%}  {n_req:>10,}   {mult:>8.1f}×")


# ── Figure 1: Power at original N ────────────────────────────────────────────
colors = ['#2e7d32', '#f9a825', '#1565c0', '#6a1b9a']

labels  = [r[0] for r in results]
powers  = [r[3] * 100 for r in results]
n_reqs  = [r[4] for r in results]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(results)), powers, color=colors, edgecolor='white', linewidth=0.8)
ax.axhline(80, color='black', linestyle='--', linewidth=1.4, label='80% target')
for i, (bar, p) in enumerate(zip(bars, powers)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{p:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xticks(range(len(results)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 105)
ax.set_ylabel("Statistical power at original N (9,849 per arm)", fontsize=12)
ax.set_title("How each noise source degrades power — holding N at original trial size", fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("output/power_at_original_N.png", dpi=150)
plt.close()
print("\nSaved: output/power_at_original_N.png")


# ── Figure 2: N needed to restore 80% power ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(results)), [r / 1000 for r in n_reqs],
              color=colors, edgecolor='white', linewidth=0.8)
ax.axhline(N_ORIGINAL / 1000, color='black', linestyle='--', linewidth=1.4,
           label=f'Original N = {N_ORIGINAL/1000:.1f}k')
for i, (bar, n, r) in enumerate(zip(bars, n_reqs, results)):
    mult = r[5]
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(n_reqs) / 1000 * 0.015,
            f'{n/1000:.0f}k\n({mult:.1f}×)', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(results)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("N per arm (thousands) for 80% power", fontsize=12)
ax.set_title("Sample size needed to restore 80% power for each noise source (independent)", fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("output/n_needed_per_scenario.png", dpi=150)
plt.close()
print("Saved: output/n_needed_per_scenario.png")
