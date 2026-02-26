# pyright: basic

from dataclasses import dataclass
import inspect
import itertools
from typing import Self, cast
from line_profiler import profile
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import qutip as qt
import qutip.qip
from enum import Enum
from math import factorial
from itertools import combinations, product

# set a parameter to see animations in line
from matplotlib import rc
rc('animation', html='jshtml')

# static image plots
# %matplotlib inline
# interactive 3D plots
# %matplotlib widget



_qeye_dict = {}
def get_qeye(dim) -> qt.Qobj:
    if dim not in _qeye_dict:
        _qeye_dict[dim] = qt.qeye(dim)
    return _qeye_dict[dim]
                


class StateManager:
    def __init__(self):
        self.systems_list: list[qt.Qobj] = []

        # first index is system (entry of systems_list), second index is subsystem within that systems_list entry
        self.state_index_dict: dict[str, tuple[int, int]] = {}

    def add_subsystem(self, dimensions: int, key: str) -> None:
        # when we add a new subsystem, it will always be alone in a new independent system

        new_system = qt.basis(dimensions, 0)
        system_index = len(self.systems_list)
        self.systems_list += [new_system]
        self.state_index_dict[key] = (system_index, 0)
    
    @profile
    def get_system_subsystems_count(self, system_index: int) -> int:
        count = 0
        for (sys_idx, sub_idx) in self.state_index_dict.values():
            if sys_idx == system_index:
                count += 1
        return count
    
    @profile
    def ptrace_subsystem(self, key: str) -> None:
        system_index, target_subsystem_index = self.state_index_dict[key]
        prev_subsystem_count = self.get_system_subsystems_count(system_index)
        remaining_subsystem_idxs = [i for i in range(prev_subsystem_count) if i != target_subsystem_index]
        self.systems_list[system_index] = self.systems_list[system_index].ptrace(remaining_subsystem_idxs)
        del self.state_index_dict[key]
        if(prev_subsystem_count > 1):
            for k, (v_sys, v_sub) in self.state_index_dict.items():
                if v_sys == system_index and v_sub > target_subsystem_index:
                    self.state_index_dict[k] = (system_index, v_sub-1)
        else:
            self.systems_list.pop(system_index)
            for k, (v_sys, v_sub) in self.state_index_dict.items():
                if v_sys > system_index:
                    self.state_index_dict[k] = (v_sys - 1, v_sub)

    @profile    
    def ptrace_subsystems(self, keys: list[str]) -> None: # https://gemini.google.com/share/8fb811e25871         
        if not keys:
            return

        # 1. Group target subsystem indices by their system index
        # We use a dict: { system_index: [list_of_sub_indices_to_remove] }
        targets_by_system: dict[int, list[int]] = {}
        for key in keys:
            if key not in self.state_index_dict:
                raise ValueError(f"Key '{key}' not found.")
            sys_idx, sub_idx = self.state_index_dict[key]
            if sys_idx not in targets_by_system:
                targets_by_system[sys_idx] = []
            targets_by_system[sys_idx].append(sub_idx)

        # 2. Process systems in reverse order
        # We go backwards so that popping a system from systems_list 
        # doesn't invalidate the system_indices we are about to process.
        sorted_systems = sorted(targets_by_system.keys(), reverse=True)

        for sys_idx in sorted_systems:
            remove_sub_idxs = targets_by_system[sys_idx]
            total_subs = self.get_system_subsystems_count(sys_idx)
            
            # Identify which indices to keep
            keep_sub_idxs = [i for i in range(total_subs) if i not in remove_sub_idxs]

            if not keep_sub_idxs:
                # The entire system is being traced out
                self.systems_list.pop(sys_idx)
                # Remove keys associated with this system
                keys_to_del = [k for k, v in self.state_index_dict.items() if v[0] == sys_idx]
                for k in keys_to_del:
                    del self.state_index_dict[k]
                
                # Shift all subsequent system indices down by 1
                for k, (v_sys, v_sub) in self.state_index_dict.items():
                    if v_sys > sys_idx:
                        self.state_index_dict[k] = (v_sys - 1, v_sub)
            else:
                # Apply ptrace (takes list of indices to KEEP)
                self.systems_list[sys_idx] = self.systems_list[sys_idx].ptrace(keep_sub_idxs)
                
                # Remove the keys we traced out
                for key in keys:
                    if key in self.state_index_dict and self.state_index_dict[key][0] == sys_idx:
                        del self.state_index_dict[key]

                # Update sub_idx for remaining keys in this system
                # The new sub_idx is the rank of the old sub_idx among the kept indices
                for k, (v_sys, v_sub) in self.state_index_dict.items():
                    if v_sys == sys_idx:
                        new_sub_idx = keep_sub_idxs.index(v_sub)
                        self.state_index_dict[k] = (sys_idx, new_sub_idx)
        
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
                self.state_index_dict[k] = (sys1, subsystem_count_1+v_sub)
            elif v_sys > sys2:
                self.state_index_dict[k] = (v_sys - 1, v_sub)

    def ensure_same_system(self, key1: str, key2: str) -> None:
        sys1, _ = self.state_index_dict[key1]
        sys2, _ = self.state_index_dict[key2]
        if(sys1 != sys2):
            self.merge_systems(sys1, sys2)
    
    @profile
    def ptrace_keep(self, keep_key_list: list[str]) -> qt.Qobj:
        """
        The list is used for the reordering of the returned density matrix, but the system index mappings are not
        changed based on that list, so after this call they might be out of order.
        """
        for i in range(1, len(keep_key_list)):
            k = keep_key_list[i]
            self.ensure_same_system(keep_key_list[0], k)
        keys_to_remove = [k for k in self.state_index_dict if k not in keep_key_list]
        self.ptrace_subsystems(keys_to_remove)  
        assert len(self.systems_list) == 1
        to_return = qt.ket2dm(self.systems_list[0]) if self.systems_list[0].isket else self.systems_list[0]
        new_order_list: list[int] = []
        for kk in keep_key_list:
            new_order_list += [self.state_index_dict[kk][1]]
        to_return.permute(new_order_list)
        return to_return

    def clone(self) -> Self:
        new_manager = StateManager()
        # Create a new list with copies of each Qobj
        new_manager.systems_list = [state.copy() for state in self.systems_list]
        # Copy the dictionary
        new_manager.state_index_dict = self.state_index_dict.copy()
        return cast(Self, new_manager)

    def print_dimensions(self) -> None:
        print("\n--- Current StateManager Dimensions ---")
        for sys_idx, system in enumerate(self.systems_list):
            # QuTiP dims are formatted as [ [subsystem_dims], [subsystem_dims] ]
            # For a Qobj, dims[0] represents the dimensions of the Hilbert space
            current_dims = system.dims[0]
            
            # Find which keys belong to this system
            keys_in_system = [
                f"{k}(sub_idx:{sub})" 
                for k, (s_idx, sub) in self.state_index_dict.items() 
                if s_idx == sys_idx
            ]
            
            type_str = "Ket" if system.isket else "Density Matrix"
            print(f"System {sys_idx} ({type_str}):")
            print(f"  Dimensions: {current_dims}")
            print(f"  Keys: {', '.join(keys_in_system)}")
        print(f"Raw dict: {self.state_index_dict}")
        print("---------------------------------------\n")
    
    @profile
    def apply_operation(self, system_index: int, operator: qt.Qobj) -> None:
        system = self.systems_list[system_index]
        if system.isket:
            system = (operator @ system).unit()
        else:
            system = operator @ system @ operator.dag()
            tr = system.tr()
            if(np.abs(tr) < 1E-5):
                tr = 1E-5
            if tr != 1.0:
                system = system / tr
        self.systems_list[system_index] = system


    # def measure_subsystem(self, key: str, outcome: int) -> None:
    #     """
    #     Applies a projective measurement to the subsystem identified by 'key'.
    #     Updates the system state and normalizes it.
    #     """
    #     if key not in self.state_index_dict:
    #         raise ValueError(f"Key '{key}' not found in StateManager.")
    #     system_index, target_sub_idx = self.state_index_dict[key]
    #     state = self.systems_list[system_index]
    #     dims = state.dims[0]  
    #     op_list = [get_qeye(d) for d in dims]
    #     op_list[target_sub_idx] = qt.basis(dims[target_sub_idx], outcome).proj()   
    #     projector = qt.tensor(*op_list)
    #     self.apply_operation(system_index, projector)
    
    @profile
    def measure_subsystems(self, measurements: list[tuple[str, int]]) -> None: # https://gemini.google.com/share/8fb811e25871
        """
        Applies projective measurements to multiple subsystems. 
        Measurements on the same system are applied simultaneously as a single operator.
        """
        if not measurements:
            return

        # 1. Group (sub_idx, outcome) by their system_index
        # Format: { system_index: [(sub_idx, outcome), ...] }
        by_system: dict[int, list[tuple[int, int]]] = {}
        
        for key, outcome in measurements:
            if key not in self.state_index_dict:
                raise ValueError(f"Key '{key}' not found in StateManager.")
            
            sys_idx, sub_idx = self.state_index_dict[key]
            if sys_idx not in by_system:
                by_system[sys_idx] = []
            by_system[sys_idx].append((sub_idx, outcome))

        # 2. Apply one aggregate projector per independent system
        for sys_idx, sub_measurements in by_system.items():
            state = self.systems_list[sys_idx]
            dims = state.dims[0]
            
            # Start with identities for all subsystems in this system
            op_list = [get_qeye(d) for d in dims]
            
            # Replace identities with projectors for the targeted subsystems
            for sub_idx, outcome in sub_measurements:
                # Build projector |outcome><outcome|
                op_list[sub_idx] = qt.basis(dims[sub_idx], outcome).proj()
            
            # Create the full tensor product projector
            projector = qt.tensor(*op_list)
            
            # Apply and normalize (normalization is handled by apply_operation)
            self.apply_operation(sys_idx, projector)


def apply_dual_mode_encoding(sm: StateManager, qubit_key: str, mode_1_key: str, mode_2_key: str):
    assert mode_1_key != mode_2_key
    apply_x(sm, mode_1_key)
    apply_cnot(sm, qubit_key, mode_2_key)
    apply_cnot(sm, mode_2_key, mode_1_key)
    apply_cnot(sm, mode_2_key, qubit_key)

def apply_dual_mode_decoding_mixed(sm: StateManager, qubit_key: str, mode_1_key: str, mode_2_key: str):
    assert mode_1_key != mode_2_key

    anc_key = f"{inspect.currentframe().f_code.co_name} ancilla" # type: ignore
    sm.add_subsystem(2, anc_key)
    apply_cnot(sm, mode_1_key, anc_key)
    apply_cnot(sm, mode_2_key, anc_key)
    # We don't care about the ancilla and we decode regardless
    # If a photon was lost we will arrive at some mixed state after the ptraces, but in this decoding we don't care 
    sm.ptrace_subsystem(anc_key) 

    apply_cnot(sm, mode_2_key, qubit_key)
    apply_cnot(sm, mode_2_key, mode_1_key)
    apply_cnot(sm, qubit_key, mode_2_key)
    apply_x(sm, mode_1_key) 

def apply_single_mode_encoding(sm: StateManager, qubit_key: str, cv_key: str):
    """
    Maps a qubit state to the first two Fock states of a CV mode.
    |0>_q |vac>_cv -> |0>_q |0>_cv
    |1>_q |vac>_cv -> |0>_q |1>_cv
    """
    sm.ensure_same_system(qubit_key, cv_key)
    system_index, qubit_pos = sm.state_index_dict[qubit_key]
    _, cv_pos = sm.state_index_dict[cv_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]
    N_cv = dims[cv_pos]
    assert type(N_cv) is int

    # Operators for CV mode
    # Map qubit |0> to CV |0> (Identity on CV if CV starts in vacuum)
    # Map qubit |1> to CV |1> (Creation operator on CV)
    
    # We use projectors to define the mapping
    # Projector for qubit |0> and leave CV as is (assuming vacuum)
    op0 = [get_qeye(d) for d in dims]
    op0[qubit_pos] = qt.basis(2, 0).proj()
    # op0[cv_pos] is already identity
    
    # Projector for qubit |1> and create a photon in CV, then flip qubit to 0
    op1 = [get_qeye(d) for d in dims]
    op1[qubit_pos] = qt.basis(2, 0) @ qt.basis(2, 1).dag()
    op1[cv_pos] = qt.create(N_cv) 

    U_encode = qt.tensor(*op0) + qt.tensor(*op1)

    sm.apply_operation(system_index, U_encode)

def apply_single_mode_decoding(sm: StateManager, qubit_key: str, cv_key: str):
    """
    The inverse of encoding: transfers the Fock state back to the qubit.
    |0>_cv -> |0>_q
    |1>_cv -> |1>_q
    """
    sm.ensure_same_system(qubit_key, cv_key)
    system_index, qubit_pos = sm.state_index_dict[qubit_key]
    _, cv_pos = sm.state_index_dict[cv_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]
    N_cv = dims[cv_pos]

    # Map CV |0> to qubit |0>
    op0 = [get_qeye(d) for d in dims]
    op0[qubit_pos] = qt.basis(2, 0).proj()
    op0[cv_pos] = qt.basis(N_cv, 0).proj()

    # Map CV |1> to qubit |1>
    op1 = [get_qeye(d) for d in dims]
    op1[qubit_pos] = qt.basis(2, 1) @ qt.basis(2, 0).dag()
    op1[cv_pos] = qt.basis(2, 0) @ qt.basis(N_cv, 1).dag() # Map Fock 1 to Fock 0

    U_decode = qt.tensor(*op0) + qt.tensor(*op1)

    sm.apply_operation(system_index, U_decode)


def apply_cat_state_encoding(sm: StateManager, qubit_key: str, cv_key: str, vertical_displacement: float, N: int):
    # 1. Prepare CV states
    vacuum = qt.basis(N, 0)
    alpha_coeff = (vertical_displacement / np.sqrt(2)) * 1j
    
    pos_disp = qt.displace(N, alpha_coeff)
    neg_disp = qt.displace(N, -alpha_coeff)
    
    logical_zero = (pos_disp @ vacuum + neg_disp @ vacuum).unit()
    logical_one  = (pos_disp @ vacuum - neg_disp @ vacuum).unit()
    
    # Define the mapping operators for the CV mode
    map_zero = logical_zero @ vacuum.dag()
    map_one  = logical_one @ vacuum.dag()

    # 2. Build the operator list for the tensor product
    sm.ensure_same_system(qubit_key, cv_key)
    system_index, qubit_position = sm.state_index_dict[qubit_key]
    _, cv_position = sm.state_index_dict[cv_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]
    num_subsystems = len(dims)

    def build_gate(qubit_state_index: int) -> qt.Qobj:
        op_list = [get_qeye(dims[i]) for i in range(num_subsystems)]
        
        if qubit_state_index == 0:
            # Map: |0>_q |vac>_cv  ->  |0>_q |cat+>_cv
            op_list[qubit_position] = qt.basis(2, 0).proj()
            op_list[cv_position] = map_zero
        else:
            # Map: |1>_q |vac>_cv  ->  |0>_q |cat->_cv
            # We use |0><1| to flip the qubit from 1 to 0 during the transfer
            op_list[qubit_position] = qt.basis(2, 0) @ qt.basis(2, 1).dag()
            op_list[cv_position] = map_one
        
        return qt.tensor(*op_list)

    # 3. Combine into the full encoding operator
    U_encode = build_gate(0) + build_gate(1)

    sm.apply_operation(system_index, U_encode)

def apply_ideal_cat_state_decoding(sm: StateManager, qubit_key: str, cv_key: str, vertical_displacement: float, N: int):
    sm.ensure_same_system(qubit_key, cv_key)
    system_index, qubit_position = sm.state_index_dict[qubit_key]
    _, cv_position = sm.state_index_dict[cv_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]
    
    # 1. Define states
    vacuum = qt.basis(N, 0)
    alpha_coeff = (vertical_displacement / np.sqrt(2)) * 1j
    
    # Define logical states for parity detection
    pos_disp = qt.displace(N, alpha_coeff)
    neg_disp = qt.displace(N, -alpha_coeff)
    logical_zero_cv = (pos_disp @ vacuum + neg_disp @ vacuum).unit()
    logical_one_cv  = (pos_disp @ vacuum - neg_disp @ vacuum).unit()

    # 2. Step A: Parity-Controlled Qubit Flip (The "Decoding")
    # This maps: |cat+>|0> -> |cat+>|0>  AND  |cat->|0> -> |cat->|1>
    def build_flip():
        # Project CV onto Parity, apply corresponding gate to Qubit
        op_plus = [get_qeye(d) for d in dims]
        op_plus[cv_position] = logical_zero_cv.proj()
        # Qubit stays same (Identity)
        
        op_minus = [get_qeye(d) for d in dims]
        op_minus[cv_position] = logical_one_cv.proj()
        op_minus[qubit_position] = qt.sigmax() # Flip if odd parity
        
        return qt.tensor(*op_plus) + qt.tensor(*op_minus)

    # 3. Step B: Qubit-Controlled Un-displacement (The "Cleaning")
    # This returns the CV mode to vacuum: |cat+>|0> -> |vac>|0> AND |cat->|1> -> |vac>|1>
    # Note: This is essentially the inverse of your encoding function.
    def build_clean():
        # This part ensures the operation is unitary by resetting the CV mode
        op_zero = [get_qeye(d) for d in dims]
        op_zero[qubit_position] = qt.basis(2, 0).proj()
        op_zero[cv_position] = vacuum @ logical_zero_cv.dag()
        
        op_one = [get_qeye(d) for d in dims]
        op_one[qubit_position] = qt.basis(2, 1).proj()
        op_one[cv_position] = vacuum @ logical_one_cv.dag()
        
        return qt.tensor(*op_zero) + qt.tensor(*op_one)

    # Combined Unitary: First flip the qubit, then clean the CV mode
    U_total = build_clean() @ build_flip()
    
    # Inside apply_ideal_cat_state_decoding:
    sm.apply_operation(system_index, U_total)

def apply_kitten_state_encoding(sm: StateManager, qubit_key: str, cv_key: str, N: int):
    sm.ensure_same_system(qubit_key, cv_key)
    system_index, qubit_pos = sm.state_index_dict[qubit_key]
    _, cv_pos = sm.state_index_dict[cv_key]
    
    system = sm.systems_list[system_index]
    dims = system.dims[0]

    logical_zero = (qt.basis(N, 0) + qt.basis(N, 4)).unit()
    logical_one = qt.basis(N, 2)

    map_zero = logical_zero @ qt.basis(N, 0).dag()
    map_one  = logical_one @ qt.basis(N, 0).dag()

    op0 = [get_qeye(d) for d in dims]
    op0[qubit_pos] = qt.basis(2, 0).proj()
    op0[cv_pos] = map_zero

    op1 = [get_qeye(d) for d in dims]
    op1[qubit_pos] = qt.basis(2, 0) @ qt.basis(2, 1).dag()
    op1[cv_pos] = map_one

    U_encode = qt.tensor(*op0) + qt.tensor(*op1)
    
    sm.apply_operation(system_index, U_encode)

def apply_kitten_state_decoding(sm: StateManager, qubit_key: str, cv_key: str, N: int):
    sm.ensure_same_system(qubit_key, cv_key)
    anc_key = "parity_ancilla"
    sm.add_subsystem(2, anc_key)
    sm.ensure_same_system(cv_key, anc_key)
    
    sys_idx, qubit_pos = sm.state_index_dict[qubit_key]
    _, cv_pos = sm.state_index_dict[cv_key]
    _, anc_pos = sm.state_index_dict[anc_key]
    
    system = sm.systems_list[sys_idx]
    dims = system.dims[0]

    parity_check = 0
    for n in range(N):
        proj_n = qt.basis(N, n).proj()
        op_list = [get_qeye(d) for d in dims]
        op_list[cv_pos] = proj_n
        if n % 2 != 0:
            op_list[anc_pos] = qt.sigmax()
        parity_check += qt.tensor(*op_list)
    parity_check = cast(qt.Qobj, parity_check)

    l0 = (qt.basis(N, 0) + qt.basis(N, 4)).unit()
    l1 = qt.basis(N, 2)

    a = qt.destroy(N)

    e0 = (a @ l0).unit()
    e1 = (a @ l1).unit()

    P_l0 = l0 @ l0.dag()
    P_l1 = l1 @ l1.dag()
    P_e0 = e0 @ e0.dag()
    P_e1 = e1 @ e1.dag()

    I_cv = get_qeye(N)
    P_sub = P_l0 + P_l1 + P_e0 + P_e1
    P_perp = I_cv - P_sub

    U_corr_cv = (
        l0 @ e0.dag()
        + l1 @ e1.dag()
        + e0 @ l0.dag()
        + e1 @ l1.dag()
        + P_perp
    )

    op_list_no_err = [get_qeye(d) for d in dims]
    op_list_no_err[anc_pos] = qt.basis(2, 0).proj()
    term_no_err = qt.tensor(*op_list_no_err)

    op_list_err = [get_qeye(d) for d in dims]
    op_list_err[anc_pos] = qt.basis(2, 1).proj()
    op_list_err[cv_pos] = U_corr_cv
    term_err = qt.tensor(*op_list_err)

    recovery_gate = term_no_err + term_err

    U_correct = recovery_gate @ parity_check

    sm.apply_operation(sys_idx, U_correct)
    
    sm.ptrace_subsystem(anc_key)

    sys_idx, qubit_pos = sm.state_index_dict[qubit_key]
    _, cv_pos = sm.state_index_dict[cv_key]
    system = sm.systems_list[sys_idx]
    dims = system.dims[0]

    l0 = (qt.basis(N, 0) + qt.basis(N, 4)).unit()
    l1 = qt.basis(N, 2)
    vac = qt.basis(N, 0)

    op_map0 = [get_qeye(d) for d in dims]
    op_map0[qubit_pos] = qt.basis(2, 0).proj()
    op_map0[cv_pos] = vac @ l0.dag()

    op_map1 = [get_qeye(d) for d in dims]
    op_map1[qubit_pos] = qt.basis(2, 1) @ qt.basis(2, 0).dag()
    op_map1[cv_pos] = vac @ l1.dag()

    U_decode = qt.tensor(*op_map0) + qt.tensor(*op_map1)

    sm.apply_operation(sys_idx, U_decode)

def apply_4_legged_cat_encoding(sm: StateManager, qubit_key: str, cv_key: str, alpha: complex, N: int):
    sm.ensure_same_system(qubit_key, cv_key)
    system_index, qubit_pos = sm.state_index_dict[qubit_key]
    _, cv_pos = sm.state_index_dict[cv_key]
    
    system = sm.systems_list[system_index]
    dims = system.dims[0]

    vac = qt.basis(N, 0)
    c_p = qt.displace(N, alpha) @ vac # type: ignore
    c_m = qt.displace(N, -alpha) @ vac # type: ignore
    c_ip = qt.displace(N, 1j * alpha) @ vac # type: ignore
    c_im = qt.displace(N, -1j * alpha) @ vac # type: ignore

    logical_zero = (c_p + c_m + c_ip + c_im).unit()
    logical_one = (c_p + c_m - c_ip - c_im).unit()

    map_zero = logical_zero @ vac.dag()
    map_one  = logical_one @ vac.dag()

    op0 = [get_qeye(d) for d in dims]
    op0[qubit_pos] = qt.basis(2, 0).proj()
    op0[cv_pos] = map_zero

    op1 = [get_qeye(d) for d in dims]
    op1[qubit_pos] = qt.basis(2, 0) @ qt.basis(2, 1).dag()
    op1[cv_pos] = map_one

    U_encode = qt.tensor(*op0) + qt.tensor(*op1)
    
    sm.apply_operation(system_index, U_encode)

@profile
def apply_4_legged_cat_decoding(sm: StateManager, qubit_key: str, cv_key: str, alpha: complex, N: int):
    sm.ensure_same_system(qubit_key, cv_key)
    anc_key = "cat_decoding_ancilla"
    sm.add_subsystem(2, anc_key)
    sm.ensure_same_system(cv_key, anc_key)
    
    sys_idx, qubit_pos = sm.state_index_dict[qubit_key]
    _, cv_pos = sm.state_index_dict[cv_key]
    _, anc_pos = sm.state_index_dict[anc_key]
    
    system = sm.systems_list[sys_idx]
    dims = system.dims[0]


    parity_check = 0
    for n in range(N):
        proj_n = qt.basis(N, n).proj()
        op_list = [get_qeye(d) for d in dims]
        op_list[cv_pos] = proj_n
        if n % 2 != 0:
            op_list[anc_pos] = qt.sigmax()
        parity_check += qt.tensor(*op_list)
    parity_check = cast(qt.Qobj, parity_check)


    vac = qt.basis(N, 0)
    l0 = (qt.displace(N, alpha) @ vac + qt.displace(N, -alpha) @ vac + qt.displace(N, 1j*alpha) @ vac + qt.displace(N, -1j*alpha) @ vac).unit() # type: ignore
    l1 = (qt.displace(N, alpha) @ vac + qt.displace(N, -alpha) @ vac - qt.displace(N, 1j*alpha) @ vac - qt.displace(N, -1j*alpha) @ vac).unit() # type: ignore

    a = qt.destroy(N)
    e0 = (a @ l0).unit()
    e1 = (a @ l1).unit()

    P_e0 = e0 @ e0.dag()
    P_e1 = e1 @ e1.dag()
    P_l0 = l0 @ l0.dag()
    P_l1 = l1 @ l1.dag()
    
    I_cv = get_qeye(N)
    U_corr_cv = (l0 @ e0.dag()) + (l1 @ e1.dag()) + (e0 @ l0.dag()) + (e1 @ l1.dag()) + (I_cv - P_e0 - P_e1 - P_l0 - P_l1)

    op_list_no_err = [get_qeye(d) for d in dims]
    op_list_no_err[anc_pos] = qt.basis(2, 0).proj()
    
    op_list_err = [get_qeye(d) for d in dims]
    op_list_err[anc_pos] = qt.basis(2, 1).proj()
    op_list_err[cv_pos] = U_corr_cv
    
    recovery_gate = qt.tensor(*op_list_no_err) + qt.tensor(*op_list_err)
    U_total = recovery_gate @ parity_check

    sm.apply_operation(sys_idx, U_total)
    
    sm.ptrace_subsystem(anc_key)

    sys_idx, qubit_pos = sm.state_index_dict[qubit_key]
    _, cv_pos = sm.state_index_dict[cv_key]
    system = sm.systems_list[sys_idx]
    dims = system.dims[0]

    op_map0 = [get_qeye(d) for d in dims]
    op_map0[qubit_pos] = qt.basis(2, 0).proj()
    op_map0[cv_pos] = vac @ l0.dag()

    op_map1 = [get_qeye(d) for d in dims]
    op_map1[qubit_pos] = qt.basis(2, 1) @ qt.basis(2, 0).dag()
    op_map1[cv_pos] = vac @ l1.dag()

    U_decode = qt.tensor(*op_map0) + qt.tensor(*op_map1)

    sm.apply_operation(sys_idx, U_decode)

def apply_kraus_loss(
    sm: StateManager,
    key: str,
    loss_prob: float,
    k_max: int | None = None
):
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

    id_ops = [get_qeye(d) for d in dims]
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

def apply_qubit_gate(sm: StateManager, target_key: str, gate: qt.Qobj):
    system_index, target_idx = sm.state_index_dict[target_key]
    
    system = sm.systems_list[system_index]
    dims = system.dims[0]

    l = [get_qeye(dims[i]) if i != target_idx else gate for i in range(len(dims))]
    X_total = qt.tensor(*l)
    sm.apply_operation(system_index, X_total)

def apply_x(sm: StateManager, target_key: str):
    apply_qubit_gate(sm, target_key, qt.sigmax())

def apply_z(sm: StateManager, target_key: str):
    apply_qubit_gate(sm, target_key, qt.sigmaz())

def apply_y(sm: StateManager, target_key: str):
    apply_qubit_gate(sm, target_key, qt.sigmay())

def apply_hadamard(sm: StateManager, target_key: str):
    apply_qubit_gate(sm, target_key, qt.gates.snot())

def apply_s(sm: StateManager, target_key: str):
    apply_qubit_gate(sm, target_key, qt.gates.s_gate())

def apply_cnot(sm: StateManager, control_key: str, target_key: str):
    sm.ensure_same_system(control_key, target_key)

    system_index, control_idx = sm.state_index_dict[control_key]
    _, target_idx = sm.state_index_dict[target_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]
    
    # Part 1: Control is in |0> (Identity on target)
    op_list_0 = [get_qeye(d) for d in dims]
    op_list_0[control_idx] = qt.basis(2, 0).proj()
    # Target stays Identity, so no change needed to op_list_0
    
    # Part 2: Control is in |1> (X on target)
    op_list_1 = [get_qeye(d) for d in dims]
    op_list_1[control_idx] = qt.basis(2, 1).proj()
    op_list_1[target_idx] = qt.sigmax()
    
    CNOT_total = qt.tensor(*op_list_0) + qt.tensor(*op_list_1)
    sm.apply_operation(system_index, CNOT_total)

def apply_cz(sm: StateManager, control_key: str, target_key: str):
    sm.ensure_same_system(control_key, target_key)

    system_index, control_idx = sm.state_index_dict[control_key]
    _, target_idx = sm.state_index_dict[target_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]

    # Control in |0>
    op_list_0 = [get_qeye(d) for d in dims]
    op_list_0[control_idx] = qt.basis(2, 0).proj()

    # Control in |1>
    op_list_1 = [get_qeye(d) for d in dims]
    op_list_1[control_idx] = qt.basis(2, 1).proj()
    op_list_1[target_idx] = qt.sigmaz()

    CZ_total = qt.tensor(*op_list_0) + qt.tensor(*op_list_1)
    sm.apply_operation(system_index, CZ_total)

def apply_cy(sm: StateManager, control_key: str, target_key: str):
    sm.ensure_same_system(control_key, target_key)

    system_index, control_idx = sm.state_index_dict[control_key]
    _, target_idx = sm.state_index_dict[target_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]

    op_list_0 = [get_qeye(d) for d in dims]
    op_list_0[control_idx] = qt.basis(2, 0).proj()

    op_list_1 = [get_qeye(d) for d in dims]
    op_list_1[control_idx] = qt.basis(2, 1).proj()
    op_list_1[target_idx] = qt.sigmay()

    CY_total = qt.tensor(*op_list_0) + qt.tensor(*op_list_1)
    sm.apply_operation(system_index, CY_total)

def apply_swap(sm: StateManager, key1: str, key2: str):
    if(key1 == key2):
        return
    
    # A SWAP is 3 CNOTs
    apply_cnot(sm, key1, key2)
    apply_cnot(sm, key2, key1)
    apply_cnot(sm, key1, key2)

def apply_toffoli(sm: StateManager, ctrl1_key: str, ctrl2_key: str, target_key: str):
    sm.ensure_same_system(ctrl1_key, ctrl2_key)
    sm.ensure_same_system(ctrl1_key, target_key)

    system_index, ctrl1 = sm.state_index_dict[ctrl1_key]
    _, ctrl2 = sm.state_index_dict[ctrl2_key]
    _, target = sm.state_index_dict[target_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]

    # Identity on all subsystems
    op_list_id = [get_qeye(d) for d in dims]
    
    # The Toffoli gate: I + |11><11| ⊗ (X - I)
    # This only acts when both controls are in the |1> state
    proj_11 = [get_qeye(d) for d in dims]
    proj_11[ctrl1] = qt.basis(2, 1).proj()
    proj_11[ctrl2] = qt.basis(2, 1).proj()
    
    # The operator (X - I) on the target
    op_x_minus_i = [get_qeye(d) for d in dims]
    op_x_minus_i[target] = qt.sigmax() - get_qeye(2)
    
    # Combine: U = Identity + (Projector_11 * Target_Flip_Logic)
    # We use element-wise multiplication of the lists to build the tensor components
    U_toffoli = qt.tensor(*op_list_id) + (qt.tensor(*proj_11) @ qt.tensor(*op_list_id).dag() @ qt.tensor(*op_x_minus_i))
    
    sm.apply_operation(system_index, U_toffoli)



def swap_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    apply_swap(sm, source_key, target_key_list[0])

def swap_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    apply_swap(sm, source_key_list[0], target_key)


def repetition_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    apply_swap(sm, source_key, target_key_list[0])
    for i in range(1, len(target_key_list)):
        target_key = target_key_list[i]
        apply_cnot(sm, target_key_list[0], target_key)

@profile
def repetition_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    if len(source_key_list) % 2 != 1:
        print("unsupported repetition_n_qubit_decode for n even")
        exit()
    
    num_ancillas = len(source_key_list)-1
    max_errors = num_ancillas / 2
    assert max_errors.is_integer()
    max_errors = int(max_errors)
    
    anc_keys: list[str] = [f"rnqd_anc_{i}" for i in range(num_ancillas)]

    for i in range(num_ancillas):
        sm.add_subsystem(2, anc_keys[i])
        apply_cnot(sm, source_key_list[i], anc_keys[i])
        apply_cnot(sm, source_key_list[i+1], anc_keys[i])

    ancilla_dm = sm.clone().ptrace_keep(anc_keys)

    possible_outcomes = list(itertools.product([0, 1], repeat=num_ancillas))
    syndrome_correction_map = {outcome: "" for outcome in possible_outcomes}
    for num_errors in range(max_errors+1):

        # generate all the possible error positions with num_errors 1s out of len(source_key_list) bits
        for possible_error_positions in combinations(range(len(source_key_list)), num_errors):
            # Each iteration looks at a different valid combination of errors with num_errors total errors
            # possible_error_positions is a list containing the indices indices of bits that are 1

            correction_string = "I" * num_ancillas
            error_syndrome = [0] * num_ancillas
            for bit_pos in possible_error_positions:
                if bit_pos != len(source_key_list)-1: # Not the last
                    error_syndrome[bit_pos] = 1 - error_syndrome[bit_pos]
                if bit_pos != 0: # Not the first
                    error_syndrome[bit_pos-1] = 1 - error_syndrome[bit_pos-1]
                correction_string = correction_string[:bit_pos] + "X" + correction_string[bit_pos+1:]

            syndrome_tuple = tuple(error_syndrome)
            if(syndrome_correction_map[syndrome_tuple] == ""):
                syndrome_correction_map[syndrome_tuple] = correction_string
    

    new_systems_list: list[qt.Qobj] = []
    for outcome_i in range(len(possible_outcomes)):
        outcome = possible_outcomes[outcome_i]
        outcome_probability: complex = ancilla_dm[outcome_i, outcome_i]
        outcome_probability = outcome_probability.real

        sm_outcome = sm.clone()
        # for i in range(num_ancillas):
        #     sm_outcome.measure_subsystem(anc_keys[i], outcome[i])
        
        sm_outcome.measure_subsystems(list(zip(anc_keys, outcome)))
        sm_outcome.ptrace_subsystems(anc_keys)

        correction_str = syndrome_correction_map[outcome] # type: ignore
        if(correction_str == "" or correction_str == None):
            print(f"WARNING: no correction for syndrome {outcome}")
            correction_str = "I" * len(source_key_list)
        for qubit_num in range(len(correction_str)):
            qubit_correction = correction_str[qubit_num]
            if qubit_correction == "I":
                pass
            elif qubit_correction == "X":
                apply_x(sm_outcome, source_key_list[qubit_num])
            elif qubit_correction == "Z":
                apply_z(sm_outcome, source_key_list[qubit_num])
            elif qubit_correction == "Y":
                apply_y(sm_outcome, source_key_list[qubit_num])
        
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

    for i in range(1, len(source_key_list)):
        # Reset all qubits apart from the first to |0>
        apply_cnot(sm, source_key_list[0], source_key_list[i])
        sm.ptrace_subsystem(source_key_list[i])

    apply_swap(sm, source_key_list[0], target_key)

    for i in range(1, len(source_key_list)):
        sm.add_subsystem(2, source_key_list[i])

def phase_repetition_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    apply_hadamard(sm, source_key)
    repetition_encode(sm, source_key, target_key_list)
    for idx in target_key_list:
        apply_hadamard(sm, idx)

def phase_repetition_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    for idx in source_key_list:
        apply_hadamard(sm, idx)
    repetition_decode(sm, target_key, source_key_list)
    apply_hadamard(sm, target_key)

def shor_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    if len(target_key_list) != 9:
        raise ValueError("Shor code requires exactly 9 channel qubits")

    q = target_key_list

    phase_repetition_encode(sm, source_key, [q[0], q[3], q[6]])

    repetition_encode(sm, q[0], [q[0], q[1], q[2]])
    repetition_encode(sm, q[3], [q[3], q[4], q[5]])
    repetition_encode(sm, q[6], [q[6], q[7], q[8]])

def shor_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    if len(source_key_list) != 9:
        raise ValueError("Shor code requires exactly 9 channel qubits")

    q = source_key_list

    repetition_decode(sm, q[0], [q[0], q[1], q[2]])
    repetition_decode(sm, q[3], [q[3], q[4], q[5]])
    repetition_decode(sm, q[6], [q[6], q[7], q[8]])

    phase_repetition_decode(sm, target_key, [q[0], q[3], q[6]])

def wrap_repetition_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    if len(target_key_list) != 9:
        raise ValueError("wrap_repetition code requires exactly 9 channel qubits")

    q = target_key_list

    repetition_encode(sm, source_key, [q[0], q[3], q[6]])

    repetition_encode(sm, q[0], [q[0], q[1], q[2]])
    repetition_encode(sm, q[3], [q[3], q[4], q[5]])
    repetition_encode(sm, q[6], [q[6], q[7], q[8]])

def wrap_repetition_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    if len(source_key_list) != 9:
        raise ValueError("wrap_repetition code requires exactly 9 channel qubits")

    q = source_key_list

    repetition_decode(sm, q[0], [q[0], q[1], q[2]])
    repetition_decode(sm, q[3], [q[3], q[4], q[5]])
    repetition_decode(sm, q[6], [q[6], q[7], q[8]])

    repetition_decode(sm, target_key, [q[0], q[3], q[6]])

def phase_wrap_repetition_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    apply_hadamard(sm, source_key)
    wrap_repetition_encode(sm, source_key, target_key_list)
    for idx in target_key_list:
        apply_hadamard(sm, idx)

def phase_wrap_repetition_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    for idx in source_key_list:
        apply_hadamard(sm, idx)
    wrap_repetition_decode(sm, target_key, source_key_list)
    apply_hadamard(sm, target_key)


def five_qubit_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    assert len(target_key_list) == 5

    q = target_key_list

    apply_swap(sm, source_key, q[4])

    apply_hadamard(sm, q[0])
    apply_s(sm, q[0])
    apply_cy(sm, q[0], q[4])
    apply_hadamard(sm, q[1])
    apply_cnot(sm, q[1], q[4])
    apply_hadamard(sm, q[2])
    apply_cz(sm, q[2], q[0])
    apply_cz(sm, q[2], q[1])
    apply_cnot(sm, q[2], q[4])
    apply_hadamard(sm, q[3])
    apply_s(sm, q[3])
    apply_cz(sm, q[3], q[0])
    apply_cz(sm, q[3], q[2])
    apply_cy(sm, q[3], q[4])

def five_qubit_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    assert len(source_key_list) == 5
    q = source_key_list

    num_ancillas = 4
    anc_keys: list[str] = [f"5q_anc_{i}" for i in range(num_ancillas)]

    for i in range(num_ancillas):
        sm.add_subsystem(2, anc_keys[i])

    m = anc_keys
    for i in range(num_ancillas):
        apply_hadamard(sm, m[i])
    
    apply_cnot(sm, m[0], q[0])
    apply_cz(sm, m[0], q[1])
    apply_cz(sm, m[0], q[2])
    apply_cnot(sm, m[0], q[3])

    apply_cnot(sm, m[1], q[1])
    apply_cz(sm, m[1], q[2])
    apply_cz(sm, m[1], q[3])
    apply_cnot(sm, m[1], q[4])

    apply_cnot(sm, m[2], q[0])
    apply_cnot(sm, m[2], q[2])
    apply_cz(sm, m[2], q[3])
    apply_cz(sm, m[2], q[4])

    apply_cz(sm, m[3], q[0])
    apply_cnot(sm, m[3], q[1])
    apply_cnot(sm, m[3], q[3])
    apply_cz(sm, m[3], q[4])

    for i in range(num_ancillas):
        apply_hadamard(sm, m[i])
    
    ancilla_dm = sm.clone().ptrace_keep(anc_keys)

    possible_outcomes = list(itertools.product([0, 1], repeat=num_ancillas))

    syndrome_correction_map = {outcome: "" for outcome in possible_outcomes}
    syndrome_correction_map[(0,0,0,0)] = "IIIII"

    syndrome_correction_map[(0,0,0,1)] = "XIIII"
    syndrome_correction_map[(1,0,1,0)] = "ZIIII"
    syndrome_correction_map[(1,0,1,1)] = "YIIII"

    syndrome_correction_map[(1,0,0,0)] = "IXIII"
    syndrome_correction_map[(0,1,0,1)] = "IZIII"
    syndrome_correction_map[(1,1,0,1)] = "IYIII"

    syndrome_correction_map[(1,1,0,0)] = "IIXII"
    syndrome_correction_map[(0,0,1,0)] = "IIZII"
    syndrome_correction_map[(1,1,1,0)] = "IIYII"

    syndrome_correction_map[(0,1,1,0)] = "IIIXI"
    syndrome_correction_map[(1,0,0,1)] = "IIIZI"
    syndrome_correction_map[(1,1,1,1)] = "IIIYI"

    syndrome_correction_map[(0,0,1,1)] = "IIIIX"
    syndrome_correction_map[(0,1,0,0)] = "IIIIZ"
    syndrome_correction_map[(0,1,1,1)] = "IIIIY"

    new_systems_list: list[qt.Qobj] = []
    for outcome_i in range(len(possible_outcomes)):
        outcome = possible_outcomes[outcome_i]
        outcome_probability: complex = ancilla_dm[outcome_i, outcome_i]
        outcome_probability = outcome_probability.real

        sm_outcome = sm.clone()

        sm_outcome.measure_subsystems(list(zip(anc_keys, outcome)))
        sm_outcome.ptrace_subsystems(anc_keys)

        correction_str = syndrome_correction_map[outcome] # type: ignore
        if(correction_str == "" or correction_str == None):
            print(f"WARNING: no correction for syndrome {outcome}")
            correction_str = "I" * len(source_key_list)
        for qubit_num in range(len(correction_str)):
            qubit_correction = correction_str[qubit_num]
            if qubit_correction == "I":
                pass
            elif qubit_correction == "X":
                apply_x(sm_outcome, source_key_list[qubit_num])
            elif qubit_correction == "Z":
                apply_z(sm_outcome, source_key_list[qubit_num])
            elif qubit_correction == "Y":
                apply_y(sm_outcome, source_key_list[qubit_num])
        
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
    


    apply_cy(sm, q[3], q[4])
    apply_cz(sm, q[3], q[2])
    apply_cz(sm, q[3], q[0])
    apply_s(sm, q[3])
    apply_hadamard(sm, q[3])
    apply_cnot(sm, q[2], q[4])
    apply_cz(sm, q[2], q[1])
    apply_cz(sm, q[2], q[0])
    apply_hadamard(sm, q[2])
    apply_cnot(sm, q[1], q[4])
    apply_hadamard(sm, q[1])
    apply_cy(sm, q[0], q[4])
    apply_s(sm, q[0])
    apply_hadamard(sm, q[0])

    apply_swap(sm, target_key, q[4])







def steane_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    assert len(target_key_list) == 7

    q = target_key_list

    apply_swap(sm, source_key, q[6])

    apply_hadamard(sm, q[0])
    apply_hadamard(sm, q[1])
    apply_hadamard(sm, q[2])
    apply_cnot(sm, q[6], q[3])
    apply_cnot(sm, q[6], q[4])
    apply_cnot(sm, q[0], q[3])
    apply_cnot(sm, q[0], q[4])
    apply_cnot(sm, q[0], q[5])
    apply_cnot(sm, q[1], q[3])
    apply_cnot(sm, q[1], q[5])
    apply_cnot(sm, q[1], q[6])
    apply_cnot(sm, q[2], q[4])
    apply_cnot(sm, q[2], q[5])
    apply_cnot(sm, q[2], q[6])


def steane_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    assert len(source_key_list) == 7
    q = source_key_list

    num_ancillas = 6
    anc_keys: list[str] = [f"5q_anc_{i}" for i in range(num_ancillas)]

    for i in range(num_ancillas):
        sm.add_subsystem(2, anc_keys[i])

    m = anc_keys
    for i in range(num_ancillas):
        apply_hadamard(sm, m[i])
    
    apply_cnot(sm, m[0], q[0])
    apply_cnot(sm, m[0], q[3])
    apply_cnot(sm, m[0], q[4])
    apply_cnot(sm, m[0], q[5])

    apply_cnot(sm, m[1], q[1])
    apply_cnot(sm, m[1], q[3])
    apply_cnot(sm, m[1], q[5])
    apply_cnot(sm, m[1], q[6])

    apply_cnot(sm, m[2], q[2])
    apply_cnot(sm, m[2], q[4])
    apply_cnot(sm, m[2], q[5])
    apply_cnot(sm, m[2], q[6])

    apply_cz(sm, m[3], q[0])
    apply_cz(sm, m[3], q[3])
    apply_cz(sm, m[3], q[4])
    apply_cz(sm, m[3], q[5])

    apply_cz(sm, m[4], q[1])
    apply_cz(sm, m[4], q[3])
    apply_cz(sm, m[4], q[5])
    apply_cz(sm, m[4], q[6])

    apply_cz(sm, m[5], q[2])
    apply_cz(sm, m[5], q[4])
    apply_cz(sm, m[5], q[5])
    apply_cz(sm, m[5], q[6])

    for i in range(num_ancillas):
        apply_hadamard(sm, m[i])
    
    ancilla_dm = sm.clone().ptrace_keep(anc_keys)

    possible_outcomes = list(itertools.product([0, 1], repeat=num_ancillas))

    syndrome_correction_map = {outcome: "IIIIIII" for outcome in possible_outcomes}
    syndrome_correction_map[(0,0,0,0,0,0)] = "IIIIIII"

    syndrome_correction_map[(0,0,0,1,0,0)] = "XIIIIII"
    syndrome_correction_map[(1,0,0,0,0,0)] = "ZIIIIII"
    syndrome_correction_map[(1,0,0,1,0,0)] = "YIIIIII"

    syndrome_correction_map[(0,0,0,0,1,0)] = "IXIIIII"
    syndrome_correction_map[(0,1,0,0,0,0)] = "IZIIIII"
    syndrome_correction_map[(0,1,0,0,1,0)] = "IYIIIII"

    syndrome_correction_map[(0,0,0,0,0,1)] = "IIXIIII"
    syndrome_correction_map[(0,0,1,0,0,0)] = "IIZIIII"
    syndrome_correction_map[(0,0,1,0,0,1)] = "IIYIIII"

    syndrome_correction_map[(0,0,0,1,1,0)] = "IIIXIII"
    syndrome_correction_map[(1,1,0,0,0,0)] = "IIIZIII"
    syndrome_correction_map[(1,1,0,1,1,0)] = "IIIYIII"

    syndrome_correction_map[(0,0,0,1,0,1)] = "IIIIXII"
    syndrome_correction_map[(1,0,1,0,0,0)] = "IIIIZII"
    syndrome_correction_map[(1,0,1,1,0,1)] = "IIIIYII"

    syndrome_correction_map[(0,0,0,1,1,1)] = "IIIIIXI"
    syndrome_correction_map[(1,1,1,0,0,0)] = "IIIIIZI"
    syndrome_correction_map[(1,1,1,1,1,1)] = "IIIIIYI"

    syndrome_correction_map[(0,0,0,0,1,1)] = "IIIIIIX"
    syndrome_correction_map[(0,1,1,0,0,0)] = "IIIIIIZ"
    syndrome_correction_map[(0,1,1,0,1,1)] = "IIIIIIY"

    new_systems_list: list[qt.Qobj] = []
    for outcome_i in range(len(possible_outcomes)):
        outcome = possible_outcomes[outcome_i]
        outcome_probability: complex = ancilla_dm[outcome_i, outcome_i]
        outcome_probability = outcome_probability.real

        sm_outcome = sm.clone()

        sm_outcome.measure_subsystems(list(zip(anc_keys, outcome)))
        sm_outcome.ptrace_subsystems(anc_keys)

        correction_str = syndrome_correction_map[outcome] # type: ignore
        if(correction_str == "" or correction_str == None):
            print(f"WARNING: no correction for syndrome {outcome}")
            correction_str = "I" * len(source_key_list)
        for qubit_num in range(len(correction_str)):
            qubit_correction = correction_str[qubit_num]
            if qubit_correction == "I":
                pass
            elif qubit_correction == "X":
                apply_x(sm_outcome, source_key_list[qubit_num])
            elif qubit_correction == "Z":
                apply_z(sm_outcome, source_key_list[qubit_num])
            elif qubit_correction == "Y":
                apply_y(sm_outcome, source_key_list[qubit_num])
        
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




    apply_cnot(sm, q[2], q[6])
    apply_cnot(sm, q[2], q[5])
    apply_cnot(sm, q[2], q[4])
    apply_cnot(sm, q[1], q[6])
    apply_cnot(sm, q[1], q[5])
    apply_cnot(sm, q[1], q[3])
    apply_cnot(sm, q[0], q[5])
    apply_cnot(sm, q[0], q[4])
    apply_cnot(sm, q[0], q[3])
    apply_cnot(sm, q[6], q[4])
    apply_cnot(sm, q[6], q[3])
    apply_hadamard(sm, q[2])
    apply_hadamard(sm, q[1])
    apply_hadamard(sm, q[0])

    apply_swap(sm, target_key, q[6])





class ChannelType(Enum):
    CV_CAT = 1
    CV_KITTEN = 2
    DV_SINGLE_MODE = 3
    DV_DUAL_MODE_MIXED = 4
    CV_CAT_4 = 5

@dataclass
class PhyLayerConfiguration:
    channel_type: ChannelType
    N: int = 2
    vertical_displacement: float | None = None
    alpha: complex | None = None

    def __post_init__(self):
        if self.channel_type == ChannelType.CV_CAT:
            if self.N is None or self.vertical_displacement is None:
                raise ValueError(
                    f"ChannelType.CV_CAT requires both 'N' and 'vertical_displacement'. "
                    f"Got: N={self.N}, displacement={self.vertical_displacement}"
                )
        if self.channel_type == ChannelType.CV_CAT_4:
            if self.N is None or self.alpha is None:
                raise ValueError(
                    f"ChannelType.CV_CAT_4 requires both 'N' and 'alpha'. "
                    f"Got: N={self.N}, alpha={self.alpha}"
                )
        if self.channel_type == ChannelType.CV_KITTEN:
            if self.N < 5:
                raise ValueError(
                    f"N={self.N} too low for Kitten state (min N=5)"
                )

class EncodingType(Enum):
    SWAP_DUMMY_ENCODING = 1
    REPETITION_BIT_FLIP = 2
    REPETITION_PHASE_FLIP = 3
    SHOR = 4
    REPETITION_BIT_FLIP_WRAP = 5
    REPETITION_PHASE_FLIP_WRAP = 6
    FIVE_QUBIT = 7
    STEANE = 8

def generic_encode(sm: StateManager, source_key: str, target_key_list: list[str], encoding: EncodingType):
    if encoding is EncodingType.SWAP_DUMMY_ENCODING:
        swap_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.REPETITION_BIT_FLIP:
        repetition_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.REPETITION_PHASE_FLIP:
        phase_repetition_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.SHOR:
        shor_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.REPETITION_BIT_FLIP_WRAP:
        wrap_repetition_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.REPETITION_PHASE_FLIP_WRAP:
        phase_wrap_repetition_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.FIVE_QUBIT:
        five_qubit_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.STEANE:
        steane_encode(sm, source_key, target_key_list)

def generic_decode(sm: StateManager, target_key: str, source_key_list: list[str], encoding: EncodingType):
    if encoding is EncodingType.SWAP_DUMMY_ENCODING:
        swap_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.REPETITION_BIT_FLIP:
        repetition_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.REPETITION_PHASE_FLIP:
        phase_repetition_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.SHOR:
        shor_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.REPETITION_BIT_FLIP_WRAP:
        wrap_repetition_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.REPETITION_PHASE_FLIP_WRAP:
        phase_wrap_repetition_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.FIVE_QUBIT:
        five_qubit_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.STEANE:
        steane_decode(sm, target_key, source_key_list)

ideal_phi_plus = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()
ideal_rho = qt.ket2dm(ideal_phi_plus)





@profile
def run_fidelity_simulation(ph: PhyLayerConfiguration, loss_prob: float, NUM_CHANNEL_QUBITS: int, encoding_type: EncodingType) -> tuple[float, float]:
    N = ph.N

    sm = StateManager()

    sm.add_subsystem(2, "tx_edge")
    sm.add_subsystem(2, "tx_temp")

    apply_hadamard(sm, "tx_edge")
    apply_cnot(sm, "tx_edge", "tx_temp")

    for i in range(NUM_CHANNEL_QUBITS):
        sm.add_subsystem(2, f"channel_{i}_tx")
    
    generic_encode(sm, "tx_temp", [f"channel_{i}_tx" for i in range(NUM_CHANNEL_QUBITS)], encoding_type)
    sm.ptrace_subsystem("tx_temp")

    for i in range(NUM_CHANNEL_QUBITS):
        sm.add_subsystem(2, f"channel_{i}_rx")
        if ph.channel_type is ChannelType.CV_CAT:
            assert ph.vertical_displacement is not None
            sm.add_subsystem(N, f"channel_{i}_cat")
            apply_cat_state_encoding(sm, f"channel_{i}_tx", f"channel_{i}_cat", ph.vertical_displacement, N)
            sm.ptrace_subsystem(f"channel_{i}_tx")
            apply_kraus_loss(sm, f"channel_{i}_cat", loss_prob)
            apply_ideal_cat_state_decoding(sm, f"channel_{i}_rx", f"channel_{i}_cat", ph.vertical_displacement, N)
            sm.ptrace_subsystem(f"channel_{i}_cat")
        if ph.channel_type is ChannelType.CV_CAT_4:
            assert ph.alpha is not None
            sm.add_subsystem(N, f"channel_{i}_cat")
            apply_4_legged_cat_encoding(sm, f"channel_{i}_tx", f"channel_{i}_cat", ph.alpha, N)
            sm.ptrace_subsystem(f"channel_{i}_tx")
            apply_kraus_loss(sm, f"channel_{i}_cat", loss_prob)
            apply_4_legged_cat_decoding(sm, f"channel_{i}_rx", f"channel_{i}_cat", ph.alpha, N)
            sm.ptrace_subsystem(f"channel_{i}_cat")
        if ph.channel_type is ChannelType.CV_KITTEN:
            sm.add_subsystem(N, f"channel_{i}_kitten")
            apply_kitten_state_encoding(sm, f"channel_{i}_tx", f"channel_{i}_kitten", N)
            sm.ptrace_subsystem(f"channel_{i}_tx")
            apply_kraus_loss(sm, f"channel_{i}_kitten", loss_prob)
            apply_kitten_state_decoding(sm, f"channel_{i}_rx", f"channel_{i}_kitten", N)
            sm.ptrace_subsystem(f"channel_{i}_kitten")
        elif ph.channel_type is ChannelType.DV_SINGLE_MODE:
            sm.add_subsystem(N, f"channel_{i}_dv_mode")
            apply_single_mode_encoding(sm, f"channel_{i}_tx", f"channel_{i}_dv_mode")
            sm.ptrace_subsystem(f"channel_{i}_tx")
            apply_kraus_loss(sm, f"channel_{i}_dv_mode", loss_prob)
            apply_single_mode_decoding(sm, f"channel_{i}_rx", f"channel_{i}_dv_mode")
            sm.ptrace_subsystem(f"channel_{i}_dv_mode")
        elif ph.channel_type is ChannelType.DV_DUAL_MODE_MIXED:
            sm.add_subsystem(N, f"channel_{i}_mode_1")
            sm.add_subsystem(N, f"channel_{i}_mode_2")
            apply_dual_mode_encoding(sm, f"channel_{i}_tx", f"channel_{i}_mode_1", f"channel_{i}_mode_2")
            sm.ptrace_subsystem(f"channel_{i}_tx")
            apply_kraus_loss(sm, f"channel_{i}_mode_1", loss_prob)
            apply_kraus_loss(sm, f"channel_{i}_mode_2", loss_prob)
            apply_dual_mode_decoding_mixed(sm, f"channel_{i}_rx", f"channel_{i}_mode_1", f"channel_{i}_mode_2")
            sm.ptrace_subsystem(f"channel_{i}_mode_1")
            sm.ptrace_subsystem(f"channel_{i}_mode_2")

    sm.add_subsystem(2, "rx_edge")

    generic_decode(sm, "rx_edge", [f"channel_{i}_rx" for i in range(NUM_CHANNEL_QUBITS)], encoding_type)

    for i in range(NUM_CHANNEL_QUBITS):
        sm.ptrace_subsystem(f"channel_{i}_rx")

    edge_qubits = sm.ptrace_keep(["tx_edge", "rx_edge"]).unit()
    fid = qt.fidelity(edge_qubits, ideal_rho)

    conc = qt.concurrence(edge_qubits)

    return fid, conc




phy_config_list: list[tuple[PhyLayerConfiguration, list[tuple[EncodingType, int]]]] = [
    # (
    #     PhyLayerConfiguration(channel_type=ChannelType.CV_KITTEN, N=5), 
    #     [
    #         (EncodingType.SWAP_DUMMY_ENCODING, 1),
    #         (EncodingType.REPETITION_BIT_FLIP, 3),
    #         # (EncodingType.REPETITION_BIT_FLIP, 9),
    #         # (EncodingType.REPETITION_BIT_FLIP_WRAP, 9),
    #         (EncodingType.SHOR, 9),
    #         (EncodingType.FIVE_QUBIT, 5),
    #         (EncodingType.STEANE, 7),
    #     ]
    # ),
    # (
    #     PhyLayerConfiguration(channel_type=ChannelType.CV_CAT, N=18, vertical_displacement=1.5), 
    #     [
    #         (EncodingType.SWAP_DUMMY_ENCODING, 1),
    #         (EncodingType.REPETITION_BIT_FLIP, 3),
    #         (EncodingType.REPETITION_BIT_FLIP, 9),
    #         (EncodingType.REPETITION_BIT_FLIP_WRAP, 9),
    #         (EncodingType.SHOR, 9),
    #         (EncodingType.FIVE_QUBIT, 5),
    #     ]
    # ),
    # (
    #     PhyLayerConfiguration(channel_type=ChannelType.CV_CAT_4, N=20, alpha=1.5), 
    #     [
    #         (EncodingType.SWAP_DUMMY_ENCODING, 1),
    #         (EncodingType.REPETITION_BIT_FLIP, 3),
    #         (EncodingType.REPETITION_BIT_FLIP, 9),
    #         (EncodingType.REPETITION_BIT_FLIP_WRAP, 9),
    #         (EncodingType.SHOR, 9),
    #         (EncodingType.FIVE_QUBIT, 5),
    #     ]
    # ),
    # (
    #     PhyLayerConfiguration(channel_type=ChannelType.DV_SINGLE_MODE), 
    #     [
    #         (EncodingType.SWAP_DUMMY_ENCODING, 1),
    #         (EncodingType.REPETITION_BIT_FLIP, 3),
    #         # (EncodingType.REPETITION_BIT_FLIP, 9),
    #         # (EncodingType.REPETITION_BIT_FLIP_WRAP, 9),
    #         (EncodingType.SHOR, 9),
    #         (EncodingType.FIVE_QUBIT, 5),
    #         (EncodingType.STEANE, 7),
    #     ]
    # ),
    (
        PhyLayerConfiguration(channel_type=ChannelType.DV_DUAL_MODE_MIXED), 
        [
            (EncodingType.SWAP_DUMMY_ENCODING, 1),
            (EncodingType.SHOR, 9),
            (EncodingType.FIVE_QUBIT, 5),
        ]
    ),
]

loss_prob_list = list(np.logspace(np.log10(0.05), np.log10(0.25), num=5)) + list(np.linspace(0.26, 0.99, 20))

# Define line styles for different physical layers to distinguish them
styles = {
    ChannelType.CV_CAT: "solid",
    ChannelType.CV_CAT_4: "solid",
    ChannelType.CV_KITTEN: "dotted",
    ChannelType.DV_SINGLE_MODE: "dashed",
    ChannelType.DV_DUAL_MODE_MIXED: "dashdot",
}


# Store results to reuse for both plots
results = []

for phy_config, codes in phy_config_list:
    mode_name = (
        "CV-CAT" if phy_config.channel_type is ChannelType.CV_CAT else
        "CV-CAT4" if phy_config.channel_type is ChannelType.CV_CAT_4 else
        "CV-KIT" if phy_config.channel_type is ChannelType.CV_KITTEN else
        "DV-1M" if phy_config.channel_type is ChannelType.DV_SINGLE_MODE else
        "DV-2M-MIXED"
    )
    ls = styles[phy_config.channel_type]

    print(f"phy_config.channel_type: {phy_config.channel_type.name}")

    for encoding_type, num_qubits in codes:
        fidelities = []
        concurrences = []
        label = f"[{mode_name}] {encoding_type.name} ({num_qubits} qubits)"

        print(f"\tencoding_type: {encoding_type.name} ({num_qubits} qubits)")

        for loss_prob in loss_prob_list:
            print(f"\t\tloss_prob: {loss_prob}")
            fid, conc = run_fidelity_simulation(
                ph=phy_config,
                loss_prob=loss_prob,
                NUM_CHANNEL_QUBITS=num_qubits,
                encoding_type=encoding_type
            )
            fidelities.append(fid)
            concurrences.append(conc)

        # Save results for plotting
        results.append((loss_prob_list, fidelities, concurrences, ls, label))

# ------------------------
# Log-Log Plot
plt.figure()
for x, y, y2, ls, label in results:
    plt.loglog(x, 1-np.array(y), ls=ls, label=label)

plt.xlabel('Loss Probability')
plt.ylabel('Infidelity')
plt.title('Infidelity vs Channel Loss (Log-Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)

# ------------------------
# Linear Plot
plt.figure()
for x, y, y2, ls, label in results:
    plt.plot(x, y, ls=ls, label=label)

plt.xlabel('Loss Probability')
plt.ylabel('Fidelity')
plt.title('Fidelity vs Channel Loss (Linear Scale)')
plt.legend()
plt.grid(True, ls="--", alpha=0.6)


plt.figure()
for x, y, y2, ls, label in results:
    plt.plot(x, y2, ls=ls, label=label)

plt.xlabel('Loss Probability')
plt.ylabel('Concurrence')
plt.title('Concurrence vs Channel Loss (Linear Scale)')
plt.legend()
plt.grid(True, ls="--", alpha=0.6)

plt.show()
