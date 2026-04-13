"""
Policy Iteration for Automatic Door Controller
===============================================
Implements:
  1. Policy Evaluation  – compute V(s) for a fixed policy
  2. Policy Improvement – greedily update policy given V(s)
  3. Policy Iteration   – repeat until convergence
"""

import numpy as np
from typing import Tuple
from door_environment import AutomaticDoorMDP, NUM_STATES, NUM_ACTIONS, ACTION_NAMES, STATE_LABELS


class PolicyIterationSolver:

    def __init__(self, mdp: AutomaticDoorMDP, theta: float = 1e-6, max_eval_iter: int = 10_000):
        """
        Parameters
        ----------
        mdp          : AutomaticDoorMDP instance
        theta        : convergence threshold for policy evaluation
        max_eval_iter: max iterations per policy evaluation sweep
        """
        self.mdp           = mdp
        self.theta         = theta
        self.max_eval_iter = max_eval_iter

        self.history: list[dict] = []   # stores per-iteration diagnostics

    # ── Policy Evaluation ─────────────────────────────────────────────────────

    def policy_evaluation(self, policy: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Iterative policy evaluation (in-place update).

        V(s) ← Σ_{s'} P(s'|s,π(s)) [ R(s,π(s)) + γ V(s') ]

        Returns updated V and number of sweeps performed.
        """
        gamma = self.mdp.gamma
        P     = self.mdp.P
        R     = self.mdp.R

        for iteration in range(self.max_eval_iter):
            delta = 0.0
            for s in range(NUM_STATES):
                a       = policy[s]
                v_new   = R[s, a] + gamma * np.dot(P[s, a], V)
                delta   = max(delta, abs(v_new - V[s]))
                V[s]    = v_new

            if delta < self.theta:
                return V, iteration + 1

        return V, self.max_eval_iter

    # ── Policy Improvement ────────────────────────────────────────────────────

    def policy_improvement(self, V: np.ndarray) -> np.ndarray:
        """
        π'(s) = argmax_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ]
        """
        gamma    = self.mdp.gamma
        P        = self.mdp.P
        R        = self.mdp.R
        policy   = np.zeros(NUM_STATES, dtype=int)

        for s in range(NUM_STATES):
            q_values  = R[s, :] + gamma * P[s, :, :] @ V
            policy[s] = np.argmax(q_values)

        return policy

    # ── Full Policy Iteration ─────────────────────────────────────────────────

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Policy Iteration until the policy is stable.

        Returns
        -------
        policy : (NUM_STATES,) array of optimal actions
        V      : (NUM_STATES,) array of optimal state values
        """
        # Initialise with a uniform "WAIT" policy and zero values
        policy = np.full(NUM_STATES, 2, dtype=int)   # 2 = WAIT
        V      = np.zeros(NUM_STATES)

        print("=" * 60)
        print("  Policy Iteration – Automatic Door Controller")
        print("=" * 60)

        for iteration in range(1, 1001):
            # Step 1: Evaluate current policy
            V, eval_sweeps = self.policy_evaluation(policy, V)

            # Step 2: Improve policy
            new_policy = self.policy_improvement(V)

            # Diagnostics
            changed = int(np.sum(new_policy != policy))
            v_norm  = float(np.linalg.norm(V))
            self.history.append({
                "iteration"  : iteration,
                "eval_sweeps": eval_sweeps,
                "changed"    : changed,
                "v_norm"     : v_norm,
                "V"          : V.copy(),
                "policy"     : new_policy.copy(),
            })

            print(f"  Iter {iteration:3d} | eval sweeps: {eval_sweeps:5d} | "
                  f"policy changes: {changed:2d} | ‖V‖: {v_norm:.4f}")

            if np.array_equal(new_policy, policy):
                print(f"\n  ✓ Converged after {iteration} iterations.\n")
                policy = new_policy
                break

            policy = new_policy

        return policy, V

    # ── Display Helpers ───────────────────────────────────────────────────────

    def print_results(self, policy: np.ndarray, V: np.ndarray) -> None:
        print("=" * 60)
        print("  OPTIMAL POLICY")
        print("=" * 60)
        print(f"{'State':<40} {'Action':<8} {'V(s)':>8}")
        print("-" * 60)
        for s in range(NUM_STATES):
            print(f"  {STATE_LABELS[s]:<38} {ACTION_NAMES[policy[s]]:<8} {V[s]:>8.3f}")
        print("=" * 60)

    def q_values(self, V: np.ndarray) -> np.ndarray:
        """Return Q(s,a) for all state-action pairs."""
        gamma = self.mdp.gamma
        return self.mdp.R + gamma * (self.mdp.P @ V)


if __name__ == "__main__":
    mdp    = AutomaticDoorMDP(gamma=0.9)
    solver = PolicyIterationSolver(mdp, theta=1e-8)
    policy, V = solver.solve()
    solver.print_results(policy, V)
