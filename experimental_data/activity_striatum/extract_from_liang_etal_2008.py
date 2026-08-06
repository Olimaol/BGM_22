"""
!!! THE MODEL DOES NOT USE THE OUTPUT OF THIS SCRIPT. !!!

The microcircuit uses dSPN 25.0 Hz / iSPN 33.0 Hz, read straight off Table 1 of the
paper for the parkinsonian Off state. This script is the mixture deconvolution that
was considered and rejected; it is kept because it is the evidence for that
decision and it supplies the sensitivity bound on dSPN (22.37 vs 25.0 Hz).
See ./README.md for why.

data extracted from paper https://doi.org/10.1523/JNEUROSCI.1176-08.2008
extraction here: https://docs.google.com/spreadsheets/d/1FYXBhNQJZx-MvGt7IpQZi1EVFxsDKHEMx73uF5hI4pE/edit?usp=sharing (local: ./Liang_etal_2008_extraction - MSN.csv)

Compute estimated means and standard deviations for true D1 and D2 groups
from summary statistics of putative D1 / putative D2 groups using the
mixture bookkeeping and the mild assumptions described by the user:

Assumptions implemented here:
1) True counts: total cells N = 136 -> D1 = D2 = 68.
2) Putative D1 (88) = all true D1 (68) + D2 that increased (20).
   Putative D2 (48) = D2 that decreased (48).
3) Baseline (off) mean and sd for D2 are the same whether those D2 later
   increase or decrease: mu_off_D2_inc = mu_off_D2_dec = mu_off_P2.
   sigma_off_D2_inc = sigma_off_D2_dec = sigma_off_P2_off.
4) The on-off change (delta_inc) for D2 that increase equals the on-off change
   for D1: delta_D2_inc = delta_D1. This implies delta_inc can be computed
   directly from putative D1: delta_inc = mu_on_P1 - mu_off_P1.
5) For variances, we assume the on-state sd for D2_increasing equals the
   measured on-state sd for putative D2 (i.e. sigma_on_D2_inc = sigma_on_P2_on).

The script solves the linear mixture equations for the component means
and uses the mixture variance formula to recover component standard
deviations where possible.

Outputs a simple summary table.

GUIDE TO THE MIXTURE VARIANCE CALCULATIONS USED BELOW
-----------------------------------------------------
We repeatedly use the standard mixture identities for a random variable X that
is a mixture of K components with weights w_i (sum_i w_i = 1), means mu_i, and
variances sigma_i^2:

1) E[X] = sum_i w_i * mu_i  (mixture mean)
2) E[X^2] = sum_i w_i * (sigma_i^2 + mu_i^2)
3) Var[X] = E[X^2] - (E[X])^2

Combining 2) and 3) gives the practical formula we use:
    Var_mix = sum_i w_i * (sigma_i^2 + mu_i^2) - (sum_i w_i * mu_i)^2

If all quantities are known except one component variance sigma_j^2, we solve
algebraically for that unknown. When the code uses counts N_* instead of
weights, the equivalence is w_i = N_i / N_total for the corresponding mixture.
"""

import math

# Given summary statistics
N_P1 = 88  # putative D1
N_P2 = 48  # putative D2
N_total = N_P1 + N_P2
N_D1 = N_D2 = N_total // 2  # equally distributed (68 each)
N_D2_inc = N_P1 - N_D1  # number of D2 that increased
N_D2_dec = N_P2  # number of D2 that decreased

# Putative group stats (off/on)
mu_off_P1 = 24.9
sigma_off_P1 = 12.5
mu_on_P1 = 39.6
sigma_on_P1 = 16.1

mu_off_P2 = 33.5
sigma_off_P2 = 12.1
mu_on_P2 = 21.1
sigma_on_P2 = 10.3

# 1) Compute delta_inc (assumption: delta_D1 == delta_D2_inc)
delta_inc = mu_on_P1 - mu_off_P1

# 2) Solve for mu_off_D1 using the OFF-state mixture equation for putative D1.
# Composition of the putative D1 group OFF:
#   - all true D1 cells (count N_D1)
#   - the D2 cells that later increase (count N_D2_inc)
# Let w_D1 = N_D1/N_P1 and w_D2inc = N_D2_inc/N_P1 be the mixture weights.
# Using E[X] for the mixture mean:
#   mu_off_P1 = w_D1 * mu_off_D1 + w_D2inc * mu_off_D2
# Rewriting in counts (multiply both sides by N_P1):
#   N_P1 * mu_off_P1 = N_D1 * mu_off_D1 + N_D2_inc * mu_off_D2
# Solve for the unknown mu_off_D1:
#   mu_off_D1 = (N_P1 * mu_off_P1 - N_D2_inc * mu_off_D2) / N_D1
mu_off_D2 = mu_off_P2  # assumption
mu_off_D1 = (N_P1 * mu_off_P1 - N_D2_inc * mu_off_D2) / N_D1

# 3) Compute on means using delta_inc
mu_on_D1 = mu_off_D1 + delta_inc
mu_on_D2_inc = mu_off_D2 + delta_inc

# 4) Overall D2 ON-state mean is a mixture of two D2 subgroups:
#    - D2_dec (measured in putative P2 ON): count N_D2_dec, mean mu_on_P2
#    - D2_inc (estimated using delta_inc): count N_D2_inc, mean mu_on_D2_inc
# Using mixture mean with weights N_D2_dec/N_D2 and N_D2_inc/N_D2:
mu_on_D2 = (N_D2_dec * mu_on_P2 + N_D2_inc * mu_on_D2_inc) / N_D2

# 5) Recover component standard deviations using the mixture variance formula.
# We use the identity:
#   Var_mix = sum_i w_i * (sigma_i^2 + mu_i^2) - (sum_i w_i * mu_i)^2
# and rearrange to solve for the unknown component variance.

var_off_P1 = sigma_off_P1**2
var_off_D2 = sigma_off_P2**2  # assumption

# OFF-state for putative P1 mixture (components: D1_off and D2_inc_off):
# Using counts instead of weights and moving -E[P1]^2 to the left, the mixture becomes
#   N_P1 * (Var_off_P1 + mu_off_P1^2)
#     = N_D1     * (sigma_off_D1^2 + mu_off_D1^2)
#     + N_D2_inc * (sigma_off_D2^2 + mu_off_D2^2)
# We know everything except sigma_off_D1^2, thus
#   sigma_off_D1^2
#     = [ N_P1 * (Var_off_P1 + mu_off_P1^2)
#         - N_D2_inc * (sigma_off_D2^2 + mu_off_D2^2) ] / N_D1
#       - mu_off_D1^2

sigma_off_D1_sq = (
    N_P1 * (var_off_P1 + mu_off_P1**2) - N_D2_inc * (var_off_D2 + mu_off_D2**2)
) / N_D1 - mu_off_D1**2

# numerical safety: if tiny negative due to rounding, clamp to zero
if sigma_off_D1_sq < 0 and sigma_off_D1_sq > -1e-8:
    sigma_off_D1_sq = 0.0

if sigma_off_D1_sq < 0:
    raise ValueError(f"Computed negative variance for D1 off: {sigma_off_D1_sq}")

sigma_off_D1 = math.sqrt(sigma_off_D1_sq)

# ON-state for putative P1 mixture to recover sigma_on_D1.
# Components (in ON): D1_on (unknown sd) and D2_inc_on (assumed sd).
# Replace each OFF quantity above with its ON counterpart and use
# mu_on_P1, mu_on_D1, mu_on_D2_inc, and the ON variances.
var_on_P1 = sigma_on_P1**2
var_on_D2 = sigma_on_P2**2  # assumption

# Algebraic rearrangement mirrors the OFF case (P1 mix: D1_on + D2_inc_on):
#   N_P1 * (Var_on_P1 + mu_on_P1^2)
#     = N_D1     * (sigma_on_D1^2 + mu_on_D1^2)
#     + N_D2_inc * (sigma_on_D2^2 + mu_on_D2_inc^2)
# => sigma_on_D1^2
#     = [ N_P1 * (Var_on_P1 + mu_on_P1^2)
#         - N_D2_inc * (sigma_on_D2^2 + mu_on_D2_inc^2) ] / N_D1
#       - mu_on_D1^2
sigma_on_D1_sq = (
    N_P1 * (var_on_P1 + mu_on_P1**2) - N_D2_inc * (var_on_D2 + mu_on_D2_inc**2)
) / N_D1 - mu_on_D1**2

if sigma_on_D1_sq < 0 and sigma_on_D1_sq > -1e-8:
    sigma_on_D1_sq = 0.0

if sigma_on_D1_sq < 0:
    raise ValueError(f"Computed negative variance for D1 on: {sigma_on_D1_sq}")

sigma_on_D1 = math.sqrt(sigma_on_D1_sq)

# Finally compute overall D2 ON-state variance from mixture of D2_dec and D2_inc.
# Use the same mixture identity with total D2 count N_D2 = N_D2_dec + N_D2_inc.
# Known components and means:
#   - D2_dec: sigma_on_D2_dec^2 = sigma_on_P2^2, mean = mu_on_D2_dec = mu_on_P2
#   - D2_inc: sigma_on_D2_inc^2 = sigma_on_P2^2 (assumption), mean = mu_on_D2_inc
# Mixture variance (in counts form):
#   N_D2 * (Var_on_D2 + mu_on_D2^2)
#     = N_D2_dec * (sigma_on_D2^2 + mu_on_D2_dec^2)
#     + N_D2_inc * (sigma_on_D2^2 + mu_on_D2_inc^2)
# Solve for Var_on_D2 and then take sqrt for sigma_on_D2.
mu_on_D2_dec = mu_on_P2
var_on_D2 = (
    N_D2_dec * (var_on_D2 + mu_on_D2_dec**2) + N_D2_inc * (var_on_D2 + mu_on_D2_inc**2)
) / N_D2 - mu_on_D2**2

if var_on_D2 < 0 and var_on_D2 > -1e-8:
    var_on_D2 = 0.0

if var_on_D2 < 0:
    raise ValueError(f"Computed negative variance for D2 on: {var_on_D2}")

sigma_on_D2 = math.sqrt(var_on_D2)

# Off variance for overall D2 is just the measured one (assumption)
sigma_off_D2 = math.sqrt(var_off_D2)

# Print results
print("Estimated means and standard deviations for true D1 and D2 groups:\n")
print(
    f"Counts: total={N_total}, D1={N_D1}, D2={N_D2} (D2_inc={N_D2_inc}, D2_dec={N_D2_dec})\n"
)
print("{:<7s} {:>6s} {:>10s} {:>10s}".format("group", "state", "mean", "std"))
print("-" * 52)
print(f"{'D1':<7s} {'off':>6s} {mu_off_D1:10.2f} {sigma_off_D1:10.2f}")
print(f"{'D1':<7s} {'on':>6s}  {mu_on_D1:10.2f} {sigma_on_D1:10.2f}")
print(f"{'D2':<7s} {'off':>6s} {mu_off_D2:10.2f} {sigma_off_D2:10.2f}")
print(f"{'D2':<7s} {'on':>6s}  {mu_on_D2:10.2f} {sigma_on_D2:10.2f}")

# Also print intermediate / helpful values
print("\nIntermediate values:")
print(f"delta_inc (assumed equal for D1 and D2_inc) = {delta_inc:.3f}")
print(f"mu_on_D2_inc (D2 that increased) = {mu_on_D2_inc:.3f}")

# ---------------------------------------------------------------------------
# The four candidate value sets, side by side
# ---------------------------------------------------------------------------
# "putative" takes the paper's response groups at face value: increasers are dSPN,
# decreasers are iSPN. "deconvolved" is everything computed above, which additionally
# assumes a 50/50 D1:D2 split and re-assigns 20 of the 88 increasers to D2.
#
# Both rest on the same unavoidable assumption -- that the direction of the levodopa
# response identifies the receptor class -- which the paper itself flags as an
# inference, not a measurement (p. 7542). The deconvolution only adds assumptions
# on top of it; it does not remove that one.
#
# Note the OFF column: iSPN is identical in both, because mu_off_D2 = mu_off_P2 is
# an *assumption* of the deconvolution rather than a result of it. The entire OFF
# effect is the 25.0 -> 22.37 shift on dSPN.
print("\n" + "=" * 60)
print("Candidate value sets (Hz). The model uses the OFF / putative column.")
print("=" * 60)
print("{:<12s} {:>14s} {:>14s}".format("", "putative", "deconvolved"))
print("-" * 60)
print(f"{'dSPN off':<12s} {mu_off_P1:>14.2f} {mu_off_D1:>14.2f}   <- model: 25.0")
print(f"{'iSPN off':<12s} {mu_off_P2:>14.2f} {mu_off_D2:>14.2f}   <- model: 33.0")
print(f"{'dSPN on':<12s} {mu_on_P1:>14.2f} {mu_on_D1:>14.2f}")
print(f"{'iSPN on':<12s} {mu_on_P2:>14.2f} {mu_on_D2:>14.2f}")
print("-" * 60)
print(
    "The model's 25.0 / 33.0 are Table 1 of the paper (n = 140, all units incl. the\n"
    "~3% unchanged), which is the same quantity as the 'putative' column here; the\n"
    "small differences are rounding and those few extra units."
)

# End of script
