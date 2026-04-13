"""
Automatic Door Environment for Reinforcement Learning
=====================================================
Models a smart door that decides to OPEN, CLOSE, or WAIT
based on sensor observations using a Markov Decision Process.
"""

import numpy as np
from enum import IntEnum


# ── Actions ──────────────────────────────────────────────────────────────────

class Action(IntEnum):
    OPEN  = 0
    CLOSE = 1
    WAIT  = 2

ACTION_NAMES = {Action.OPEN: "OPEN", Action.CLOSE: "CLOSE", Action.WAIT: "WAIT"}


# ── States ───────────────────────────────────────────────────────────────────

class State(IntEnum):
    """
    Composite state: (door_position, person_presence, time_of_day)

    door_position : CLOSED=0, OPEN=1
    person_presence: NO_PERSON=0, PERSON_DETECTED=1, PERSON_PASSING=2
    time_of_day   : OFF_PEAK=0, PEAK=1

    Total states = 2 × 3 × 2 = 12
    """
    CLOSED_NONE_OFF   = 0
    CLOSED_NONE_PEAK  = 1
    CLOSED_DETECT_OFF = 2
    CLOSED_DETECT_PEAK= 3
    CLOSED_PASS_OFF   = 4
    CLOSED_PASS_PEAK  = 5
    OPEN_NONE_OFF     = 6
    OPEN_NONE_PEAK    = 7
    OPEN_DETECT_OFF   = 8
    OPEN_DETECT_PEAK  = 9
    OPEN_PASS_OFF     = 10
    OPEN_PASS_PEAK    = 11

NUM_STATES  = 12
NUM_ACTIONS = 3

STATE_LABELS = [
    "Closed | No Person | Off-Peak",
    "Closed | No Person | Peak",
    "Closed | Person Detected | Off-Peak",
    "Closed | Person Detected | Peak",
    "Closed | Person Passing | Off-Peak",
    "Closed | Person Passing | Peak",
    "Open   | No Person | Off-Peak",
    "Open   | No Person | Peak",
    "Open   | Person Detected | Off-Peak",
    "Open   | Person Detected | Peak",
    "Open   | Person Passing | Off-Peak",
    "Open   | Person Passing | Peak",
]


class AutomaticDoorMDP:
    """
    MDP for an automatic door controller.

    Reward design
    -------------
    +10  : Door opens when person is detected / passing
    +5   : Door closes when no person present
    -5   : Door stays closed while person is passing (blocking)
    -3   : Door opens when no person present (energy waste)
    -1   : WAIT in any state (small penalty to encourage decisiveness)
    +2   : WAIT when person is only detected (reasonable caution)
    """

    def __init__(self, gamma: float = 0.9):
        self.gamma       = gamma
        self.num_states  = NUM_STATES
        self.num_actions = NUM_ACTIONS

        self.R = self._build_reward_matrix()
        self.P = self._build_transition_matrix()

    # ── Reward Matrix R[s, a] ─────────────────────────────────────────────────

    def _build_reward_matrix(self) -> np.ndarray:
        R = np.full((NUM_STATES, NUM_ACTIONS), -1.0)  # default: small penalty

        for s in range(NUM_STATES):
            door_open      = s >= 6
            no_person      = s % 6 in (0, 1)
            person_detect  = s % 6 in (2, 3)
            person_passing = s % 6 in (4, 5)

            # OPEN action
            if person_detect or person_passing:
                R[s, Action.OPEN] = 10.0   # correct: door opens for person
            else:
                R[s, Action.OPEN] = -3.0   # waste: no one there

            # CLOSE action
            if no_person:
                R[s, Action.CLOSE] = 5.0   # correct: save energy
            elif person_passing:
                R[s, Action.CLOSE] = -8.0  # dangerous: closing on person
            else:
                R[s, Action.CLOSE] = -2.0

            # WAIT action
            if person_passing and not door_open:
                R[s, Action.WAIT] = -5.0   # blocking person
            elif person_detect:
                R[s, Action.WAIT] = 2.0    # reasonable caution
            elif no_person and not door_open:
                R[s, Action.WAIT] = 1.0    # fine to wait when idle
            else:
                R[s, Action.WAIT] = -1.0

        return R

    # ── Transition Matrix P[s, a, s'] ────────────────────────────────────────

    def _build_transition_matrix(self) -> np.ndarray:
        """
        Stochastic transitions reflecting real-world uncertainty:
        - Sensors occasionally miss people (false negatives)
        - Door mechanics may fail or lag
        - People flow is probabilistic
        """
        P = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))

        transition_rules = {
            # (state, action) → [(prob, next_state), ...]
            # ── CLOSED states ────────────────────────────────────────────────
            (State.CLOSED_NONE_OFF, Action.OPEN):   [(0.9, State.OPEN_NONE_OFF),   (0.1, State.CLOSED_NONE_OFF)],
            (State.CLOSED_NONE_OFF, Action.CLOSE):  [(0.8, State.CLOSED_NONE_OFF), (0.2, State.CLOSED_NONE_PEAK)],
            (State.CLOSED_NONE_OFF, Action.WAIT):   [(0.7, State.CLOSED_NONE_OFF), (0.2, State.CLOSED_DETECT_OFF), (0.1, State.CLOSED_NONE_PEAK)],

            (State.CLOSED_NONE_PEAK, Action.OPEN):  [(0.9, State.OPEN_NONE_PEAK),  (0.1, State.CLOSED_NONE_PEAK)],
            (State.CLOSED_NONE_PEAK, Action.CLOSE): [(0.8, State.CLOSED_NONE_PEAK),(0.2, State.CLOSED_DETECT_PEAK)],
            (State.CLOSED_NONE_PEAK, Action.WAIT):  [(0.5, State.CLOSED_NONE_PEAK),(0.4, State.CLOSED_DETECT_PEAK),(0.1, State.CLOSED_NONE_OFF)],

            (State.CLOSED_DETECT_OFF, Action.OPEN): [(0.9, State.OPEN_DETECT_OFF), (0.1, State.OPEN_PASS_OFF)],
            (State.CLOSED_DETECT_OFF, Action.CLOSE):[(0.7, State.CLOSED_DETECT_OFF),(0.2, State.CLOSED_PASS_OFF),(0.1, State.CLOSED_NONE_OFF)],
            (State.CLOSED_DETECT_OFF, Action.WAIT): [(0.6, State.CLOSED_DETECT_OFF),(0.3, State.CLOSED_PASS_OFF),(0.1, State.CLOSED_NONE_OFF)],

            (State.CLOSED_DETECT_PEAK, Action.OPEN):[(0.9, State.OPEN_DETECT_PEAK),(0.1, State.OPEN_PASS_PEAK)],
            (State.CLOSED_DETECT_PEAK, Action.CLOSE):[(0.6,State.CLOSED_DETECT_PEAK),(0.3,State.CLOSED_PASS_PEAK),(0.1,State.CLOSED_NONE_PEAK)],
            (State.CLOSED_DETECT_PEAK, Action.WAIT):[(0.5, State.CLOSED_DETECT_PEAK),(0.4,State.CLOSED_PASS_PEAK),(0.1,State.CLOSED_NONE_PEAK)],

            (State.CLOSED_PASS_OFF, Action.OPEN):   [(0.95, State.OPEN_PASS_OFF),  (0.05, State.OPEN_NONE_OFF)],
            (State.CLOSED_PASS_OFF, Action.CLOSE):  [(0.8, State.CLOSED_PASS_OFF), (0.2, State.CLOSED_NONE_OFF)],
            (State.CLOSED_PASS_OFF, Action.WAIT):   [(0.7, State.CLOSED_PASS_OFF), (0.3, State.CLOSED_NONE_OFF)],

            (State.CLOSED_PASS_PEAK, Action.OPEN):  [(0.95, State.OPEN_PASS_PEAK), (0.05, State.OPEN_NONE_PEAK)],
            (State.CLOSED_PASS_PEAK, Action.CLOSE): [(0.8, State.CLOSED_PASS_PEAK),(0.2, State.CLOSED_NONE_PEAK)],
            (State.CLOSED_PASS_PEAK, Action.WAIT):  [(0.6, State.CLOSED_PASS_PEAK),(0.4, State.CLOSED_NONE_PEAK)],

            # ── OPEN states ──────────────────────────────────────────────────
            (State.OPEN_NONE_OFF, Action.OPEN):     [(0.9, State.OPEN_NONE_OFF),   (0.1, State.OPEN_DETECT_OFF)],
            (State.OPEN_NONE_OFF, Action.CLOSE):    [(0.9, State.CLOSED_NONE_OFF), (0.1, State.OPEN_NONE_OFF)],
            (State.OPEN_NONE_OFF, Action.WAIT):     [(0.7, State.OPEN_NONE_OFF),   (0.2, State.OPEN_DETECT_OFF),(0.1, State.CLOSED_NONE_OFF)],

            (State.OPEN_NONE_PEAK, Action.OPEN):    [(0.8, State.OPEN_NONE_PEAK),  (0.2, State.OPEN_DETECT_PEAK)],
            (State.OPEN_NONE_PEAK, Action.CLOSE):   [(0.9, State.CLOSED_NONE_PEAK),(0.1, State.OPEN_NONE_PEAK)],
            (State.OPEN_NONE_PEAK, Action.WAIT):    [(0.5, State.OPEN_NONE_PEAK),  (0.4, State.OPEN_DETECT_PEAK),(0.1, State.CLOSED_NONE_PEAK)],

            (State.OPEN_DETECT_OFF, Action.OPEN):   [(0.8, State.OPEN_DETECT_OFF), (0.2, State.OPEN_PASS_OFF)],
            (State.OPEN_DETECT_OFF, Action.CLOSE):  [(0.8, State.CLOSED_DETECT_OFF),(0.2, State.OPEN_NONE_OFF)],
            (State.OPEN_DETECT_OFF, Action.WAIT):   [(0.6, State.OPEN_DETECT_OFF), (0.3, State.OPEN_PASS_OFF),(0.1, State.OPEN_NONE_OFF)],

            (State.OPEN_DETECT_PEAK, Action.OPEN):  [(0.7, State.OPEN_DETECT_PEAK),(0.3, State.OPEN_PASS_PEAK)],
            (State.OPEN_DETECT_PEAK, Action.CLOSE): [(0.8, State.CLOSED_DETECT_PEAK),(0.2, State.OPEN_NONE_PEAK)],
            (State.OPEN_DETECT_PEAK, Action.WAIT):  [(0.5, State.OPEN_DETECT_PEAK),(0.4, State.OPEN_PASS_PEAK),(0.1, State.OPEN_NONE_PEAK)],

            (State.OPEN_PASS_OFF, Action.OPEN):     [(0.8, State.OPEN_PASS_OFF),   (0.2, State.OPEN_NONE_OFF)],
            (State.OPEN_PASS_OFF, Action.CLOSE):    [(0.9, State.CLOSED_NONE_OFF), (0.1, State.OPEN_PASS_OFF)],
            (State.OPEN_PASS_OFF, Action.WAIT):     [(0.6, State.OPEN_PASS_OFF),   (0.4, State.OPEN_NONE_OFF)],

            (State.OPEN_PASS_PEAK, Action.OPEN):    [(0.7, State.OPEN_PASS_PEAK),  (0.3, State.OPEN_NONE_PEAK)],
            (State.OPEN_PASS_PEAK, Action.CLOSE):   [(0.9, State.CLOSED_NONE_PEAK),(0.1, State.OPEN_PASS_PEAK)],
            (State.OPEN_PASS_PEAK, Action.WAIT):    [(0.5, State.OPEN_PASS_PEAK),  (0.5, State.OPEN_NONE_PEAK)],
        }

        for (s, a), next_states in transition_rules.items():
            for prob, ns in next_states:
                P[s, a, ns] += prob

        # Normalise rows to ensure valid probability distributions
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                row_sum = P[s, a].sum()
                if row_sum > 0:
                    P[s, a] /= row_sum

        return P
