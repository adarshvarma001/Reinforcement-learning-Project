"""
Unit Tests – Automatic Door RL
================================
Run with:  python -m pytest tests/ -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from door_environment import AutomaticDoorMDP, NUM_STATES, NUM_ACTIONS, Action
from policy_iteration  import PolicyIterationSolver


@pytest.fixture(scope="module")
def mdp():
    return AutomaticDoorMDP(gamma=0.9)


@pytest.fixture(scope="module")
def solved(mdp):
    solver = PolicyIterationSolver(mdp, theta=1e-8)
    policy, V = solver.solve()
    return solver, policy, V


# ── MDP structure tests ───────────────────────────────────────────────────────

class TestMDP:
    def test_reward_shape(self, mdp):
        assert mdp.R.shape == (NUM_STATES, NUM_ACTIONS)

    def test_transition_shape(self, mdp):
        assert mdp.P.shape == (NUM_STATES, NUM_ACTIONS, NUM_STATES)

    def test_transition_probabilities_sum_to_one(self, mdp):
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                total = mdp.P[s, a].sum()
                assert abs(total - 1.0) < 1e-6, f"P[{s},{a}] sums to {total}"

    def test_reward_open_when_person_passing(self, mdp):
        # States 4,5,10,11 have person passing
        for s in [4, 5, 10, 11]:
            assert mdp.R[s, Action.OPEN] == 10.0

    def test_reward_close_when_person_passing_is_negative(self, mdp):
        for s in [4, 5, 10, 11]:
            assert mdp.R[s, Action.CLOSE] < 0

    def test_reward_close_when_no_person_is_positive(self, mdp):
        for s in [0, 1, 6, 7]:   # no_person states
            assert mdp.R[s, Action.CLOSE] > 0

    def test_gamma_in_range(self, mdp):
        assert 0 < mdp.gamma < 1


# ── Policy Iteration tests ────────────────────────────────────────────────────

class TestPolicySolver:
    def test_policy_shape(self, solved):
        _, policy, _ = solved
        assert policy.shape == (NUM_STATES,)

    def test_value_shape(self, solved):
        _, _, V = solved
        assert V.shape == (NUM_STATES,)

    def test_policy_actions_valid(self, solved):
        _, policy, _ = solved
        assert all(0 <= a < NUM_ACTIONS for a in policy)

    def test_convergence_history_nonempty(self, solved):
        solver, _, _ = solved
        assert len(solver.history) > 0

    def test_policy_opens_for_person_passing_closed(self, solved):
        """
        When door is CLOSED and person is PASSING, optimal policy should OPEN.
        States 4 (CLOSED_PASS_OFF) and 5 (CLOSED_PASS_PEAK).
        """
        _, policy, _ = solved
        for s in [4, 5]:
            assert policy[s] == Action.OPEN, \
                f"Expected OPEN in state {s}, got {policy[s]}"

    def test_policy_closes_when_open_no_person(self, solved):
        """
        When door is OPEN and no person present, optimal policy should CLOSE.
        States 6 (OPEN_NONE_OFF) and 7 (OPEN_NONE_PEAK).
        """
        _, policy, _ = solved
        for s in [6, 7]:
            assert policy[s] == Action.CLOSE, \
                f"Expected CLOSE in state {s}, got {policy[s]}"

    def test_value_improves_monotonically(self, solved):
        """‖V‖ should be non-decreasing (or stable) over iterations."""
        solver, _, _ = solved
        norms = [h["v_norm"] for h in solver.history]
        for i in range(1, len(norms)):
            assert norms[i] >= norms[i - 1] - 1e-3   # allow tiny numerical dip

    def test_policy_evaluation_convergence(self, solved):
        """Final iteration should have a reasonable number of sweeps."""
        solver, _, _ = solved
        assert solver.history[-1]["eval_sweeps"] <= solver.max_eval_iter


# ── Q-value tests ─────────────────────────────────────────────────────────────

class TestQValues:
    def test_q_values_shape(self, solved):
        solver, _, V = solved
        Q = solver.q_values(V)
        assert Q.shape == (NUM_STATES, NUM_ACTIONS)

    def test_optimal_action_matches_q_argmax(self, solved):
        solver, policy, V = solved
        Q = solver.q_values(V)
        for s in range(NUM_STATES):
            assert policy[s] == np.argmax(Q[s]), \
                f"State {s}: policy={policy[s]}, argmax Q={np.argmax(Q[s])}"
