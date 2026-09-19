"""
Generates the two figures recommended for the Results section.
Run: python generate_figures.py
Outputs: fig1_crossprovider.pdf, fig2_temp_variance.pdf (vector, print-ready)

Fig 1 numbers are taken directly from Table I (three-way comparison).
Fig 2 numbers: only the two data points explicitly given (temp 0.4 and
temp 0.8, per provider) are plotted. The temp=0.0 point is NOT plotted
because no std was reported for it in the transfer notes (Groq @ 0.0 had
identical num_proposed=4 but the combined_speedup *range* was given, not
a std) -- don't invent a std to force a 3-point line. If you have the
temp=0.0 std for both providers, add it to `temps`/`gemini_std`/`groq_std`
below and the line will extend automatically.
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Figure 1: cross-provider comparative benchmark (Table I)
# ---------------------------------------------------------------------

systems = ["explain_advisor", "llm_planner", "rule_based"]

groq_vals = [1.272, 1.155, 1.096]          # point estimates only, no std
anthropic_vals = [1.218, 1.026, 1.083]
anthropic_std = [0.177, 0.105, 0.081]
gemini_vals = [1.097, 1.009, 1.143]
gemini_std = [0.079, 0.105, 0.207]

x = np.arange(len(systems))
width = 0.25

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(x - width, groq_vals, width, label="Groq (n=4, no std reported)")
ax.bar(x, anthropic_vals, width, yerr=anthropic_std, capsize=4,
       label="Anthropic (n=5)")
ax.bar(x + width, gemini_vals, width, yerr=gemini_std, capsize=4,
       label="Gemini (n=5)")

ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(systems, rotation=15)
ax.set_ylabel("Combined workload speedup")
ax.set_title("Cross-Provider Comparative Benchmark")
ax.legend()
fig.tight_layout()
fig.savefig("fig1_crossprovider.pdf")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: combined-speedup variance vs. temperature, by provider
# ---------------------------------------------------------------------

temps = [0.4, 0.8]
# Only the two std values explicitly stated in the transfer notes:
# Gemini @ 0.4 std = 0.014 ("tightest low-temperature cell")
# Gemini/Groq @ 0.8 std from Table II (1.175±0.165 / 1.170±0.170)
gemini_std_by_temp = [0.014, 0.165]
groq_std_by_temp = [None, 0.170]  # Groq @ 0.4 std not given in transfer

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(temps, gemini_std_by_temp, marker="o", label="Gemini")

# Groq: only one point available, so plot as a single marker, not a line
if groq_std_by_temp[1] is not None:
    ax.plot([0.8], [groq_std_by_temp[1]], marker="s", linestyle="none",
            label="Groq (only temp=0.8 std available)")

ax.set_xlabel("Temperature")
ax.set_ylabel("combined_speedup standard deviation")
ax.set_title("Output Variance vs. Temperature")
ax.set_xticks([0.0, 0.4, 0.8])
ax.legend()
fig.tight_layout()
fig.savefig("fig2_temp_variance.pdf")
plt.close(fig)

print("Wrote fig1_crossprovider.pdf and fig2_temp_variance.pdf")
