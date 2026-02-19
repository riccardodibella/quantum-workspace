# pyright: basic

# https://claude.ai/share/41de830c-4b8f-4c4e-8b8b-6b536631ef1d

from dataclasses import dataclass
import inspect
from typing import Self, cast
from line_profiler import profile
import numpy as np
import qutip as qt
import qutip.qip
from math import factorial
from itertools import product


# ─────────────────────────────────────────────────────────────────────────────
# StateManager
# ─────────────────────────────────────────────────────────────────────────────

class StateManager:
    def __init__(self):
        self.systems_list: list[qt.Qobj] = []
        self.state_index_dict: dict[str, tuple[int, int]] = {}

    def add_subsystem(self, dimensions: int, key: str) -> None:
        new_system = qt.basis(dimensions, 0)
        system_index = len(self.systems_list)
        self.systems_list += [new_system]
        self.state_index_dict[key] = (system_index, 0)

    def get_system_subsystems_count(self, system_index: int) -> int:
        count = 0
        for (sys_idx, sub_idx) in self.state_index_dict.values():
            if sys_idx == system_index:
                count += 1
        return count

    def ptrace_subsystem(self, key: str) -> None:
        system_index, target_subsystem_index = self.state_index_dict[key]
        prev_subsystem_count = self.get_system_subsystems_count(system_index)
        self.systems_list[system_index] = self.systems_list[system_index].ptrace(
            [i for i in range(prev_subsystem_count) if i != target_subsystem_index]
        )
        del self.state_index_dict[key]
        if prev_subsystem_count > 1:
            for k, (v_sys, v_sub) in self.state_index_dict.items():
                if v_sys == system_index and v_sub > target_subsystem_index:
                    self.state_index_dict[k] = (system_index, v_sub - 1)
        else:
            self.systems_list.pop(system_index)
            for k, (v_sys, v_sub) in self.state_index_dict.items():
                if v_sys > system_index:
                    self.state_index_dict[k] = (v_sys - 1, v_sub)

    def merge_systems(self, sys1: int, sys2: int) -> None:
        if sys2 < sys1:
            sys1, sys2 = sys2, sys1
        subsystem_count_1 = self.get_system_subsystems_count(sys1)
        s1 = self.systems_list[sys1]
        s2 = self.systems_list[sys2]
        if s1.isket and s2.isket:
            combined = qt.tensor(s1, s2)
        else:
            dm1 = qt.ket2dm(s1) if s1.isket else s1
            dm2 = qt.ket2dm(s2) if s2.isket else s2
            combined = qt.tensor(dm1, dm2)
        self.systems_list[sys1] = combined
        self.systems_list.pop(sys2)
        for k, (v_sys, v_sub) in self.state_index_dict.items():
            if v_sys == sys2:
                self.state_index_dict[k] = (sys1, subsystem_count_1 + v_sub)
            elif v_sys > sys2:
                self.state_index_dict[k] = (v_sys - 1, v_sub)

    def ensure_same_system(self, key1: str, key2: str) -> None:
        sys1, _ = self.state_index_dict[key1]
        sys2, _ = self.state_index_dict[key2]
        if sys1 != sys2:
            self.merge_systems(sys1, sys2)

    def ptrace_keep(self, keep_key_list: list[str], force_density_matrix: bool = True) -> qt.Qobj:
        for i in range(1, len(keep_key_list)):
            k = keep_key_list[i]
            self.ensure_same_system(keep_key_list[0], k)
        starting_key_list = list(self.state_index_dict.keys())
        for k in starting_key_list:
            if k not in keep_key_list:
                self.ptrace_subsystem(k)
        assert len(self.systems_list) == 1
        return qt.ket2dm(self.systems_list[0]) if force_density_matrix and self.systems_list[0].isket else self.systems_list[0]

    def clone(self) -> Self:
        new_manager = StateManager()
        new_manager.systems_list = [state.copy() for state in self.systems_list]
        new_manager.state_index_dict = self.state_index_dict.copy()
        return cast(Self, new_manager)

    def apply_operation(self, system_index: int, operator: qt.Qobj) -> None:
        system = self.systems_list[system_index]
        if system.isket:
            system = (operator @ system).unit()
        else:
            system = operator @ system @ operator.dag()
            tr = system.tr()
            if np.abs(tr) < 1E-5:
                tr = 1E-5
            system = system / tr
        self.systems_list[system_index] = system

    def measure_subsystem(self, key: str, outcome: int) -> None:
        if key not in self.state_index_dict:
            raise ValueError(f"Key '{key}' not found in StateManager.")
        system_index, target_sub_idx = self.state_index_dict[key]
        state = self.systems_list[system_index]
        dims = state.dims[0]
        op_list = [qt.qeye(d) for d in dims]
        op_list[target_sub_idx] = qt.basis(dims[target_sub_idx], outcome).proj()
        projector = qt.tensor(*op_list)
        self.apply_operation(system_index, projector)


# ─────────────────────────────────────────────────────────────────────────────
# Primitive gates
# ─────────────────────────────────────────────────────────────────────────────

def apply_x(sm: StateManager, target_key: str):
    system_index, target_idx = sm.state_index_dict[target_key]
    system = sm.systems_list[system_index]
    dims = system.dims[0]
    op_list = [qt.qeye(d) for d in dims]
    op_list[target_idx] = qt.gates.sigmax()
    sm.apply_operation(system_index, qt.tensor(*op_list))

def apply_z(sm: StateManager, target_key: str):
    system_index, target_idx = sm.state_index_dict[target_key]
    system = sm.systems_list[system_index]
    dims = system.dims[0]
    op_list = [qt.qeye(d) for d in dims]
    op_list[target_idx] = qt.sigmaz()
    sm.apply_operation(system_index, qt.tensor(*op_list))

def apply_hadamard(sm: StateManager, target_key: str):
    system_index, target_idx = sm.state_index_dict[target_key]
    system = sm.systems_list[system_index]
    dims = system.dims[0]
    op_list = [qt.qeye(d) for d in dims]
    op_list[target_idx] = qt.gates.snot()
    sm.apply_operation(system_index, qt.tensor(*op_list))

def apply_cnot(sm: StateManager, control_key: str, target_key: str):
    sm.ensure_same_system(control_key, target_key)
    system_index, control_idx = sm.state_index_dict[control_key]
    _, target_idx = sm.state_index_dict[target_key]
    system = sm.systems_list[system_index]
    dims = system.dims[0]
    op_list_0 = [qt.qeye(d) for d in dims]
    op_list_0[control_idx] = qt.basis(2, 0).proj()
    op_list_1 = [qt.qeye(d) for d in dims]
    op_list_1[control_idx] = qt.basis(2, 1).proj()
    op_list_1[target_idx] = qt.sigmax()
    sm.apply_operation(system_index, qt.tensor(*op_list_0) + qt.tensor(*op_list_1))

def apply_cz(sm: StateManager, control_key: str, target_key: str):
    sm.ensure_same_system(control_key, target_key)
    system_index, control_idx = sm.state_index_dict[control_key]
    _, target_idx = sm.state_index_dict[target_key]
    system = sm.systems_list[system_index]
    dims = system.dims[0]
    op_list_0 = [qt.qeye(d) for d in dims]
    op_list_0[control_idx] = qt.basis(2, 0).proj()
    op_list_1 = [qt.qeye(d) for d in dims]
    op_list_1[control_idx] = qt.basis(2, 1).proj()
    op_list_1[target_idx] = qt.sigmaz()
    sm.apply_operation(system_index, qt.tensor(*op_list_0) + qt.tensor(*op_list_1))

def apply_swap(sm: StateManager, key1: str, key2: str):
    if key1 == key2:
        return
    apply_cnot(sm, key1, key2)
    apply_cnot(sm, key2, key1)
    apply_cnot(sm, key1, key2)


# ─────────────────────────────────────────────────────────────────────────────
# Channel noise + dual-mode mixed PHY layer
# ─────────────────────────────────────────────────────────────────────────────

def apply_kraus_loss(sm: StateManager, key: str, loss_prob: float, k_max: int | None = None):
    system_index, local_idx = sm.state_index_dict[key]
    system = sm.systems_list[system_index]
    if system.isket:
        system = qt.ket2dm(system)
    dims = system.dims[0]
    N = dims[local_idx]
    assert isinstance(N, int)
    eta = 1.0 - loss_prob
    if k_max is None:
        k_max = N - 1
    a = qt.destroy(N)
    n = a.dag() @ a
    eta_n: qt.Qobj = (0.5 * np.log(eta) * n).expm()
    id_ops = [qt.qeye(d) for d in dims]
    rho_out = 0 * system
    for k in range(k_max + 1):
        Ak_local = (
            ((1 - eta) ** (k / 2))
            / np.sqrt(float(factorial(k)))
            * eta_n
            * (a ** k)
        )
        op_list = id_ops.copy()
        op_list[local_idx] = Ak_local
        K = qt.tensor(*op_list)
        rho_out += K @ system @ K.dag()
    sm.systems_list[system_index] = rho_out


def apply_dual_mode_encoding(sm: StateManager, qubit_key: str, mode_1_key: str, mode_2_key: str):
    assert mode_1_key != mode_2_key
    apply_x(sm, mode_1_key)
    apply_cnot(sm, qubit_key, mode_2_key)
    apply_cnot(sm, mode_2_key, mode_1_key)
    apply_cnot(sm, mode_2_key, qubit_key)

def apply_dual_mode_decoding_mixed(sm: StateManager, qubit_key: str, mode_1_key: str, mode_2_key: str):
    assert mode_1_key != mode_2_key
    anc_key = f"dual_mode_decoding_ancilla"
    sm.add_subsystem(2, anc_key)
    apply_cnot(sm, mode_1_key, anc_key)
    apply_cnot(sm, mode_2_key, anc_key)
    # Discard ancilla — decode regardless of photon-loss outcome
    sm.ptrace_subsystem(anc_key)
    apply_cnot(sm, mode_2_key, qubit_key)
    apply_cnot(sm, mode_2_key, mode_1_key)
    apply_cnot(sm, qubit_key, mode_2_key)
    apply_x(sm, mode_1_key)


def apply_dual_mode_mixed_channel(sm: StateManager, tx_key: str, rx_key: str, loss_prob: float):
    """
    Sends qubit `tx_key` through the dual-mode mixed PHY channel:
      encode into two modes → Kraus loss on each mode → mixed decode into `rx_key`.
    Both modes are dim-2 (single-photon subspace).
    """
    mode_1_key = f"{tx_key}_m1"
    mode_2_key = f"{tx_key}_m2"
    sm.add_subsystem(2, mode_1_key)
    sm.add_subsystem(2, mode_2_key)
    apply_dual_mode_encoding(sm, tx_key, mode_1_key, mode_2_key)
    sm.ptrace_subsystem(tx_key)
    apply_kraus_loss(sm, mode_1_key, loss_prob)
    apply_kraus_loss(sm, mode_2_key, loss_prob)
    apply_dual_mode_decoding_mixed(sm, rx_key, mode_1_key, mode_2_key)
    sm.ptrace_subsystem(mode_1_key)
    sm.ptrace_subsystem(mode_2_key)


# ─────────────────────────────────────────────────────────────────────────────
# Steane encode
# ─────────────────────────────────────────────────────────────────────────────

def steane_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    # Fig. 13 of https://doi.org/10.1109/MCAS.2024.3349668
    apply_swap(sm, source_key, target_key_list[0])
    q = target_key_list
    apply_hadamard(sm, q[4])
    apply_hadamard(sm, q[5])
    apply_hadamard(sm, q[6])
    apply_cnot(sm, q[0], q[3])
    apply_cnot(sm, q[0], q[2])
    apply_cnot(sm, q[6], q[3])
    apply_cnot(sm, q[6], q[2])
    apply_cnot(sm, q[6], q[1])
    apply_cnot(sm, q[5], q[3])
    apply_cnot(sm, q[5], q[1])
    apply_cnot(sm, q[5], q[0])
    apply_cnot(sm, q[4], q[2])
    apply_cnot(sm, q[4], q[1])
    apply_cnot(sm, q[4], q[0])


# ─────────────────────────────────────────────────────────────────────────────
# Steane decode variants  — add / swap your variants here
# ─────────────────────────────────────────────────────────────────────────────

@profile
def steane_decode_v1(sm: StateManager, target_key: str, source_key_list: list[str]):
    """Original version — Fig. 14 of https://doi.org/10.1109/MCAS.2024.3349668"""

    anc_keys: list[str] = [f"steane_anc_{i}" for i in range(6)]
    for k in anc_keys:
        sm.add_subsystem(2, k)
    for k in anc_keys:
        apply_hadamard(sm, k)

    m = anc_keys
    q = source_key_list

    apply_cnot(sm, m[5], q[6])
    apply_cnot(sm, m[5], q[3])
    apply_cnot(sm, m[5], q[2])
    apply_cnot(sm, m[5], q[1])

    apply_cnot(sm, m[4], q[5])
    apply_cnot(sm, m[4], q[3])
    apply_cnot(sm, m[4], q[1])
    apply_cnot(sm, m[4], q[0])

    apply_cnot(sm, m[3], q[4])
    apply_cnot(sm, m[3], q[2])
    apply_cnot(sm, m[3], q[1])
    apply_cnot(sm, m[3], q[0])

    apply_cz(sm, m[2], q[6])
    apply_cz(sm, m[2], q[3])
    apply_cz(sm, m[2], q[2])
    apply_cz(sm, m[2], q[1])

    apply_cz(sm, m[1], q[5])
    apply_cz(sm, m[1], q[3])
    apply_cz(sm, m[1], q[1])
    apply_cz(sm, m[1], q[0])

    apply_cz(sm, m[0], q[4])
    apply_cz(sm, m[0], q[2])
    apply_cz(sm, m[0], q[1])
    apply_cz(sm, m[0], q[0])

    for k in anc_keys:
        apply_hadamard(sm, k)

    possible_outcomes = list(product([0, 1], repeat=6))

    new_systems_list: list[qt.Qobj] = []
    for outcome_i in range(len(possible_outcomes)):
        outcome = possible_outcomes[outcome_i]
        outcome_probability = 1.0
        for i, bit in enumerate(outcome):
            sub_dm = sm.clone().ptrace_keep([anc_keys[i]], force_density_matrix=True)
            p = sub_dm[bit, bit].real
            outcome_probability *= p
            # (probability is computed on sm, correction is applied on sm_outcome below)

        sm_outcome = sm.clone()
        for i in range(len(outcome)):
            sm_outcome.measure_subsystem(anc_keys[i], outcome[i])
            sm_outcome.ptrace_subsystem(anc_keys[i])

        x_bits = (outcome[0], outcome[1], outcome[2])
        z_bits = (outcome[3], outcome[4], outcome[5])

        # if x_bits == (0, 0, 0):   pass
        # elif x_bits == (1, 0, 0): apply_x(sm_outcome, q[6])
        # elif x_bits == (0, 1, 0): apply_x(sm_outcome, q[5])
        # elif x_bits == (0, 0, 1): apply_x(sm_outcome, q[4])
        # elif x_bits == (1, 1, 0): apply_x(sm_outcome, q[3])
        # elif x_bits == (1, 0, 1): apply_x(sm_outcome, q[2])
        # elif x_bits == (0, 1, 1): apply_x(sm_outcome, q[1])
        # elif x_bits == (1, 1, 1): apply_x(sm_outcome, q[0])

        # if z_bits == (0, 0, 0):   pass
        # elif z_bits == (1, 0, 0): apply_z(sm_outcome, q[6])
        # elif z_bits == (0, 1, 0): apply_z(sm_outcome, q[5])
        # elif z_bits == (0, 0, 1): apply_z(sm_outcome, q[4])
        # elif z_bits == (1, 1, 0): apply_z(sm_outcome, q[3])
        # elif z_bits == (1, 0, 1): apply_z(sm_outcome, q[2])
        # elif z_bits == (0, 1, 1): apply_z(sm_outcome, q[1])
        # elif z_bits == (1, 1, 1): apply_z(sm_outcome, q[0])

        if outcome_i == 0:
            for sys in sm_outcome.systems_list:
                n = sys.shape[0]
                new_systems_list.append(qt.Qobj(np.zeros((n, n)), dims=[sys.dims[0], sys.dims[0]]))
        new_state_index_dict = sm_outcome.state_index_dict

        for i in range(len(sm_outcome.systems_list)):
            branch_state = sm_outcome.systems_list[i]
            dm_to_add = qt.ket2dm(branch_state) if branch_state.isket else branch_state
            new_systems_list[i] += outcome_probability * dm_to_add

    sm.systems_list = new_systems_list
    sm.state_index_dict = new_state_index_dict.copy()

    apply_cnot(sm, q[4], q[0])
    apply_cnot(sm, q[4], q[1])
    apply_cnot(sm, q[4], q[2])
    apply_cnot(sm, q[5], q[0])
    apply_cnot(sm, q[5], q[1])
    apply_cnot(sm, q[5], q[3])
    apply_cnot(sm, q[6], q[1])
    apply_cnot(sm, q[6], q[2])
    apply_cnot(sm, q[6], q[3])
    apply_cnot(sm, q[0], q[2])
    apply_cnot(sm, q[0], q[3])
    apply_hadamard(sm, q[4])
    apply_hadamard(sm, q[5])
    apply_hadamard(sm, q[6])

    apply_swap(sm, source_key_list[0], target_key)


@profile
def steane_decode_v2(sm: StateManager, target_key: str, source_key_list: list[str]):
    """Original version — Fig. 14 of https://doi.org/10.1109/MCAS.2024.3349668"""

    anc_keys: list[str] = [f"steane_anc_{i}" for i in range(6)]
    for k in anc_keys:
        sm.add_subsystem(2, k)
    for k in anc_keys:
        apply_hadamard(sm, k)

    m = anc_keys
    q = source_key_list

    apply_cnot(sm, m[5], q[6])
    apply_cnot(sm, m[5], q[3])
    apply_cnot(sm, m[5], q[2])
    apply_cnot(sm, m[5], q[1])

    apply_cnot(sm, m[4], q[5])
    apply_cnot(sm, m[4], q[3])
    apply_cnot(sm, m[4], q[1])
    apply_cnot(sm, m[4], q[0])

    apply_cnot(sm, m[3], q[4])
    apply_cnot(sm, m[3], q[2])
    apply_cnot(sm, m[3], q[1])
    apply_cnot(sm, m[3], q[0])

    apply_cz(sm, m[2], q[6])
    apply_cz(sm, m[2], q[3])
    apply_cz(sm, m[2], q[2])
    apply_cz(sm, m[2], q[1])

    apply_cz(sm, m[1], q[5])
    apply_cz(sm, m[1], q[3])
    apply_cz(sm, m[1], q[1])
    apply_cz(sm, m[1], q[0])

    apply_cz(sm, m[0], q[4])
    apply_cz(sm, m[0], q[2])
    apply_cz(sm, m[0], q[1])
    apply_cz(sm, m[0], q[0])

    for k in anc_keys:
        apply_hadamard(sm, k)

    possible_outcomes = list(product([0, 1], repeat=6))

    new_systems_list: list[qt.Qobj] = []
    for outcome_i in range(len(possible_outcomes)):
        outcome = possible_outcomes[outcome_i]
        outcome_probability = 1.0
        for i, bit in enumerate(outcome):
            sub_dm = sm.clone().ptrace_keep([anc_keys[i]], force_density_matrix=True)
            p = sub_dm[bit, bit].real
            outcome_probability *= p
            # (probability is computed on sm, correction is applied on sm_outcome below)

        sm_outcome = sm.clone()
        for i in range(len(outcome)):
            sm_outcome.measure_subsystem(anc_keys[i], outcome[i])
            sm_outcome.ptrace_subsystem(anc_keys[i])

        x_bits = (outcome[0], outcome[1], outcome[2])
        z_bits = (outcome[3], outcome[4], outcome[5])

        if x_bits == (0, 0, 0):   pass
        elif x_bits == (1, 0, 0): apply_x(sm_outcome, q[6])
        elif x_bits == (0, 1, 0): apply_x(sm_outcome, q[5])
        elif x_bits == (0, 0, 1): apply_x(sm_outcome, q[4])
        elif x_bits == (1, 1, 0): apply_x(sm_outcome, q[3])
        elif x_bits == (1, 0, 1): apply_x(sm_outcome, q[2])
        elif x_bits == (0, 1, 1): apply_x(sm_outcome, q[1])
        elif x_bits == (1, 1, 1): apply_x(sm_outcome, q[0])

        if z_bits == (0, 0, 0):   pass
        elif z_bits == (1, 0, 0): apply_z(sm_outcome, q[6])
        elif z_bits == (0, 1, 0): apply_z(sm_outcome, q[5])
        elif z_bits == (0, 0, 1): apply_z(sm_outcome, q[4])
        elif z_bits == (1, 1, 0): apply_z(sm_outcome, q[3])
        elif z_bits == (1, 0, 1): apply_z(sm_outcome, q[2])
        elif z_bits == (0, 1, 1): apply_z(sm_outcome, q[1])
        elif z_bits == (1, 1, 1): apply_z(sm_outcome, q[0])

        if outcome_i == 0:
            for sys in sm_outcome.systems_list:
                n = sys.shape[0]
                new_systems_list.append(qt.Qobj(np.zeros((n, n)), dims=[sys.dims[0], sys.dims[0]]))
        new_state_index_dict = sm_outcome.state_index_dict

        for i in range(len(sm_outcome.systems_list)):
            branch_state = sm_outcome.systems_list[i]
            dm_to_add = qt.ket2dm(branch_state) if branch_state.isket else branch_state
            new_systems_list[i] += outcome_probability * dm_to_add

    sm.systems_list = new_systems_list
    sm.state_index_dict = new_state_index_dict.copy()

    apply_cnot(sm, q[4], q[0])
    apply_cnot(sm, q[4], q[1])
    apply_cnot(sm, q[4], q[2])
    apply_cnot(sm, q[5], q[0])
    apply_cnot(sm, q[5], q[1])
    apply_cnot(sm, q[5], q[3])
    apply_cnot(sm, q[6], q[1])
    apply_cnot(sm, q[6], q[2])
    apply_cnot(sm, q[6], q[3])
    apply_cnot(sm, q[0], q[2])
    apply_cnot(sm, q[0], q[3])
    apply_hadamard(sm, q[4])
    apply_hadamard(sm, q[5])
    apply_hadamard(sm, q[6])

    apply_swap(sm, source_key_list[0], target_key)


@profile
def steane_decode_v3(sm: StateManager, target_key: str, source_key_list: list[str]):
    """Original version — Fig. 14 of https://doi.org/10.1109/MCAS.2024.3349668"""

    anc_keys: list[str] = [f"steane_anc_{i}" for i in range(6)]
    for k in anc_keys:
        sm.add_subsystem(2, k)
    for k in anc_keys:
        apply_hadamard(sm, k)

    m = anc_keys
    q = source_key_list

    apply_cnot(sm, m[5], q[6])
    apply_cnot(sm, m[5], q[3])
    apply_cnot(sm, m[5], q[2])
    apply_cnot(sm, m[5], q[1])

    apply_cnot(sm, m[4], q[5])
    apply_cnot(sm, m[4], q[3])
    apply_cnot(sm, m[4], q[1])
    apply_cnot(sm, m[4], q[0])

    apply_cnot(sm, m[3], q[4])
    apply_cnot(sm, m[3], q[2])
    apply_cnot(sm, m[3], q[1])
    apply_cnot(sm, m[3], q[0])

    apply_cz(sm, m[2], q[6])
    apply_cz(sm, m[2], q[3])
    apply_cz(sm, m[2], q[2])
    apply_cz(sm, m[2], q[1])

    apply_cz(sm, m[1], q[5])
    apply_cz(sm, m[1], q[3])
    apply_cz(sm, m[1], q[1])
    apply_cz(sm, m[1], q[0])

    apply_cz(sm, m[0], q[4])
    apply_cz(sm, m[0], q[2])
    apply_cz(sm, m[0], q[1])
    apply_cz(sm, m[0], q[0])

    for k in anc_keys:
        apply_hadamard(sm, k)

    possible_outcomes = list(product([0, 1], repeat=6))

    new_systems_list: list[qt.Qobj] = []
    for outcome_i in range(len(possible_outcomes)):
        outcome = possible_outcomes[outcome_i]
        outcome_probability = 1.0
        for i, bit in enumerate(outcome):
            sub_dm = sm.clone().ptrace_keep([anc_keys[i]], force_density_matrix=True)
            p = sub_dm[bit, bit].real
            outcome_probability *= p
            # (probability is computed on sm, correction is applied on sm_outcome below)

        sm_outcome = sm.clone()
        for i in range(len(outcome)):
            sm_outcome.measure_subsystem(anc_keys[i], outcome[i])
            sm_outcome.ptrace_subsystem(anc_keys[i])

        z_bits = (outcome[0], outcome[1], outcome[2])
        x_bits = (outcome[3], outcome[4], outcome[5])

        if z_bits == (0, 0, 0) or z_bits == x_bits:
            if x_bits == (0, 0, 0):   pass
            #elif x_bits == (1, 0, 0): apply_x(sm_outcome, q[6])
            #elif x_bits == (0, 1, 0): apply_x(sm_outcome, q[5])
            #elif x_bits == (0, 0, 1): apply_x(sm_outcome, q[4])
            #elif x_bits == (1, 1, 0): apply_x(sm_outcome, q[3])
            #elif x_bits == (1, 0, 1): apply_x(sm_outcome, q[2])
            #elif x_bits == (0, 1, 1): apply_x(sm_outcome, q[1])
            elif x_bits == (1, 1, 1): apply_x(sm_outcome, q[0])

        if x_bits == (0, 0, 0) or x_bits == z_bits:
            if z_bits == (0, 0, 0):   pass
            #elif z_bits == (1, 0, 0): apply_z(sm_outcome, q[6])
            #elif z_bits == (0, 1, 0): apply_z(sm_outcome, q[5])
            #elif z_bits == (0, 0, 1): apply_z(sm_outcome, q[4])
            #elif z_bits == (1, 1, 0): apply_z(sm_outcome, q[3])
            #elif z_bits == (1, 0, 1): apply_z(sm_outcome, q[2])
            #elif z_bits == (0, 1, 1): apply_z(sm_outcome, q[1])
            elif z_bits == (1, 1, 1): apply_z(sm_outcome, q[0])

        if outcome_i == 0:
            for sys in sm_outcome.systems_list:
                n = sys.shape[0]
                new_systems_list.append(qt.Qobj(np.zeros((n, n)), dims=[sys.dims[0], sys.dims[0]]))
        new_state_index_dict = sm_outcome.state_index_dict

        for i in range(len(sm_outcome.systems_list)):
            branch_state = sm_outcome.systems_list[i]
            dm_to_add = qt.ket2dm(branch_state) if branch_state.isket else branch_state
            new_systems_list[i] += outcome_probability * dm_to_add

    sm.systems_list = new_systems_list
    sm.state_index_dict = new_state_index_dict.copy()

    apply_cnot(sm, q[4], q[0])
    apply_cnot(sm, q[4], q[1])
    apply_cnot(sm, q[4], q[2])
    apply_cnot(sm, q[5], q[0])
    apply_cnot(sm, q[5], q[1])
    apply_cnot(sm, q[5], q[3])
    apply_cnot(sm, q[6], q[1])
    apply_cnot(sm, q[6], q[2])
    apply_cnot(sm, q[6], q[3])
    apply_cnot(sm, q[0], q[2])
    apply_cnot(sm, q[0], q[3])
    apply_hadamard(sm, q[4])
    apply_hadamard(sm, q[5])
    apply_hadamard(sm, q[6])

    apply_swap(sm, source_key_list[0], target_key)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NUM_CHANNEL_QUBITS = 7
LOSS_PROB = 0.01   # adjust as needed

ideal_phi_plus = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
                  qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
ideal_rho = qt.ket2dm(ideal_phi_plus)

def build_post_channel_state() -> tuple[StateManager, str]:
    """
    Encodes once with Steane-7, sends each of the 7 qubits through the
    dual-mode mixed PHY channel (encode -> Kraus loss x 2 modes -> mixed decode),
    and returns the frozen StateManager ready to be cloned for each decode variant.
    """
    sm = StateManager()

    # Bell pair: tx_edge stays local, tx_temp is Steane-encoded and sent
    sm.add_subsystem(2, "tx_edge")
    sm.add_subsystem(2, "tx_temp")
    apply_hadamard(sm, "tx_edge")
    apply_cnot(sm, "tx_edge", "tx_temp")

    for i in range(NUM_CHANNEL_QUBITS):
        sm.add_subsystem(2, f"ch_{i}_tx")

    steane_encode(sm, "tx_temp", [f"ch_{i}_tx" for i in range(NUM_CHANNEL_QUBITS)])
    sm.ptrace_subsystem("tx_temp")

    # PHY layer: each of the 7 Steane qubits goes through dual-mode mixed channel
    for i in range(NUM_CHANNEL_QUBITS):
        sm.add_subsystem(2, f"ch_{i}_rx")
        apply_dual_mode_mixed_channel(sm, f"ch_{i}_tx", f"ch_{i}_rx", LOSS_PROB)

    return sm, "tx_edge"


def measure_fidelity(sm: StateManager, tx_edge_key: str, decoded_key: str) -> float:
    edge_qubits = sm.clone().ptrace_keep([tx_edge_key, decoded_key]).unit()
    print(edge_qubits)
    return qt.fidelity(edge_qubits, ideal_rho)


# ─────────────────────────────────────────────────────────────────────────────
# Main — encode once, channel once, decode many times
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building post-channel state (encode + dual-mode mixed channel) …")
    sm_post_channel, tx_edge_key = build_post_channel_state()
    channel_keys = [f"ch_{i}_rx" for i in range(NUM_CHANNEL_QUBITS)]

    sm = sm_post_channel.clone()
    sm.add_subsystem(2, "rx")
    steane_decode_v1(sm, "rx", channel_keys)
    fid = measure_fidelity(sm, tx_edge_key, "rx")
    print(f"[v1] Fidelity = {fid:.6f}")


    sm = sm_post_channel.clone()
    sm.add_subsystem(2, "rx")
    steane_decode_v2(sm, "rx", channel_keys)
    fid = measure_fidelity(sm, tx_edge_key, "rx")
    print(f"[v2] Fidelity = {fid:.6f}")

    sm = sm_post_channel.clone()
    sm.add_subsystem(2, "rx")
    steane_decode_v3(sm, "rx", channel_keys)
    fid = measure_fidelity(sm, tx_edge_key, "rx")
    print(f"[v3] Fidelity = {fid:.6f}")

    print("\nDone.")