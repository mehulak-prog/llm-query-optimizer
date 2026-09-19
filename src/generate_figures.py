"""
Generates the two figures recommended for the Results section.
Run: python generate_figures.py
Outputs: fig1_crossprovider.pdf, fig2_temp_variance.pdf (vector, print-ready)

Fig 1 numbers are taken directly from Table I (three-way comparison).

Fig 2 numbers: Anthropic now has a full three-point sweep (temp 0.0,
0.4, 0.8 -- Table III in the paper). Gemini only has two points (0.4,
0.8) and Groq only has one (0.8), per what was actually measured in the
original Phase 5a sweep -- don't invent the missing points to force
symmetric lines across all three providers. This asymmetry is itself
part of the honest story: it's exactly why Anthropic's non-replication
of the Gemini/Groq escalating-variance pattern is reported as a
suggestive finding from a modest sample, not an established trend.
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

fig, ax = plt.subplots(figsize=(6, 4))

# Anthropic: full 3-point sweep (Table III) -- the only provider with
# all three temperature settings measured.
anthropic_temps = [0.0, 0.4, 0.8]
anthropic_std_by_temp = [0.148, 0.172, 0.112]
ax.plot(anthropic_temps, anthropic_std_by_temp, marker="^",
        label="Anthropic (n=5-7/cell, all 3 temps)")

# Gemini: only two points explicitly given in the transfer notes
# (temp=0.4 std=0.014 "tightest low-temp cell"; temp=0.8 from Table II).
gemini_temps = [0.4, 0.8]
gemini_std_by_temp = [0.014, 0.165]
ax.plot(gemini_temps, gemini_std_by_temp, marker="o",
        label="Gemini (only temp=0.4, 0.8 measured)")

# Groq: only one point available (temp=0.8, from Table II) -- plotted
# as a single marker, not a line, since there's nothing to connect it to.
ax.plot([0.8], [0.170], marker="s", linestyle="none",
        label="Groq (only temp=0.8 measured)")

ax.set_xlabel("Temperature")
ax.set_ylabel("combined_speedup standard deviation")
ax.set_title("Output Variance vs. Temperature, by Provider")
ax.set_xticks([0.0, 0.4, 0.8])
ax.legend()
fig.tight_layout()
fig.savefig("fig2_temp_variance.pdf")
plt.close(fig)

print("Wrote fig1_crossprovider.pdf and fig2_temp_variance.pdf")
