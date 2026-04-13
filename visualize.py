"""
Visualisation – Automatic Door RL Results
=========================================
Generates plots saved to ../results/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from door_environment import (
    AutomaticDoorMDP, NUM_STATES, NUM_ACTIONS,
    ACTION_NAMES, STATE_LABELS
)
from policy_iteration import PolicyIterationSolver

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Colour palette
ACTION_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#3498db"}   # green/red/blue
DOOR_COLORS   = ["#e74c3c", "#2ecc71"]                         # closed/open


def plot_convergence(history: list[dict]) -> None:
    iters     = [h["iteration"]   for h in history]
    changes   = [h["changed"]     for h in history]
    v_norms   = [h["v_norm"]      for h in history]
    sweeps    = [h["eval_sweeps"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Policy Iteration – Convergence", fontsize=14, fontweight="bold")

    axes[0].plot(iters, changes, marker="o", color="#e74c3c", linewidth=2)
    axes[0].set_title("Policy Changes per Iteration")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("# States Changed")
    axes[0].grid(alpha=0.3)

    axes[1].plot(iters, v_norms, marker="s", color="#3498db", linewidth=2)
    axes[1].set_title("Value Function Norm ‖V‖")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("‖V‖")
    axes[1].grid(alpha=0.3)

    axes[2].plot(iters, sweeps, marker="^", color="#9b59b6", linewidth=2)
    axes[2].set_title("Policy Evaluation Sweeps")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Sweeps")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "convergence.png"), dpi=150)
    plt.close()
    print("  Saved: results/convergence.png")


def plot_optimal_policy(policy: np.ndarray, V: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Optimal Policy & Value Function – Automatic Door", fontsize=14, fontweight="bold")

    # ── Policy bar chart ──────────────────────────────────────────────────────
    ax = axes[0]
    colors = [ACTION_COLORS[policy[s]] for s in range(NUM_STATES)]
    bars   = ax.barh(range(NUM_STATES), [1] * NUM_STATES, color=colors, edgecolor="white", height=0.8)
    ax.set_yticks(range(NUM_STATES))
    ax.set_yticklabels([f"S{s}: {STATE_LABELS[s]}" for s in range(NUM_STATES)], fontsize=8)
    ax.set_xticks([])
    ax.set_title("Optimal Action per State")
    ax.invert_yaxis()

    # Action text labels inside bars
    for s in range(NUM_STATES):
        ax.text(0.5, s, ACTION_NAMES[policy[s]], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    legend_patches = [
        mpatches.Patch(color=ACTION_COLORS[0], label="OPEN"),
        mpatches.Patch(color=ACTION_COLORS[1], label="CLOSE"),
        mpatches.Patch(color=ACTION_COLORS[2], label="WAIT"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")

    # ── Value function ────────────────────────────────────────────────────────
    ax2 = axes[1]
    bar_colors = [DOOR_COLORS[1] if s >= 6 else DOOR_COLORS[0] for s in range(NUM_STATES)]
    ax2.barh(range(NUM_STATES), V, color=bar_colors, edgecolor="white", height=0.8)
    ax2.set_yticks(range(NUM_STATES))
    ax2.set_yticklabels([f"S{s}" for s in range(NUM_STATES)], fontsize=8)
    ax2.set_title("State Value V(s)")
    ax2.set_xlabel("Value")
    ax2.invert_yaxis()
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.grid(axis="x", alpha=0.3)

    legend_patches2 = [
        mpatches.Patch(color=DOOR_COLORS[0], label="Door Closed"),
        mpatches.Patch(color=DOOR_COLORS[1], label="Door Open"),
    ]
    ax2.legend(handles=legend_patches2, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "optimal_policy.png"), dpi=150)
    plt.close()
    print("  Saved: results/optimal_policy.png")


def plot_q_values(solver: PolicyIterationSolver, V: np.ndarray) -> None:
    Q = solver.q_values(V)   # shape: (NUM_STATES, NUM_ACTIONS)

    fig, ax = plt.subplots(figsize=(12, 7))
    x      = np.arange(NUM_STATES)
    width  = 0.28

    for a in range(NUM_ACTIONS):
        ax.bar(x + (a - 1) * width, Q[:, a], width,
               label=ACTION_NAMES[a], color=list(ACTION_COLORS.values())[a], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in range(NUM_STATES)], rotation=45, ha="right")
    ax.set_ylabel("Q(s, a)")
    ax.set_title("Q-Values for All State–Action Pairs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "q_values.png"), dpi=150)
    plt.close()
    print("  Saved: results/q_values.png")


def plot_value_evolution(history: list[dict]) -> None:
    """Show how V(s) evolves across iterations for each state."""
    iterations = [h["iteration"] for h in history]
    V_history  = np.array([h["V"] for h in history])   # shape (iters, states)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap    = plt.get_cmap("tab20")

    for s in range(NUM_STATES):
        ax.plot(iterations, V_history[:, s], label=f"S{s}", color=cmap(s / NUM_STATES), linewidth=1.5)

    ax.set_xlabel("Iteration"); ax.set_ylabel("V(s)")
    ax.set_title("Value Function Evolution Across Iterations")
    ax.legend(ncol=4, fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "value_evolution.png"), dpi=150)
    plt.close()
    print("  Saved: results/value_evolution.png")


if __name__ == "__main__":
    mdp    = AutomaticDoorMDP(gamma=0.9)
    solver = PolicyIterationSolver(mdp, theta=1e-8)
    policy, V = solver.solve()
    solver.print_results(policy, V)

    print("\nGenerating plots …")
    plot_convergence(solver.history)
    plot_optimal_policy(policy, V)
    plot_q_values(solver, V)
    plot_value_evolution(solver.history)
    print("\nAll plots saved to results/")
