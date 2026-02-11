# pyright: basic

from dataclasses import dataclass
from typing import Self, cast
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import qutip as qt
import qutip.qip
from enum import Enum
from math import factorial

# set a parameter to see animations in line
from matplotlib import rc
rc('animation', html='jshtml')

# static image plots
# %matplotlib inline
# interactive 3D plots
# %matplotlib widget






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
    
    def get_system_subsystems_count(self, system_index: int) -> int:
        count = 0
        for (sys_idx, sub_idx) in self.state_index_dict.values():
            if sys_idx == system_index:
                count += 1
        return count
        
    def ptrace_subsystem(self, key: str) -> None:
        system_index, target_subsystem_index = self.state_index_dict[key]
        prev_subsystem_count = self.get_system_subsystems_count(system_index)
        self.systems_list[system_index] = self.systems_list[system_index].ptrace([i for i in range(prev_subsystem_count) if i != target_subsystem_index])
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
    op0 = [qt.qeye(d) for d in dims]
    op0[qubit_pos] = qt.basis(2, 0).proj()
    # op0[cv_pos] is already identity
    
    # Projector for qubit |1> and create a photon in CV, then flip qubit to 0
    op1 = [qt.qeye(d) for d in dims]
    op1[qubit_pos] = qt.basis(2, 0) @ qt.basis(2, 1).dag()
    op1[cv_pos] = qt.create(N_cv) 

    U_encode = qt.tensor(*op0) + qt.tensor(*op1)

    if system.isket:
        sm.systems_list[system_index] = U_encode @ system
    else:
        sm.systems_list[system_index] = U_encode @ system @ U_encode.dag()

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
    op0 = [qt.qeye(d) for d in dims]
    op0[qubit_pos] = qt.basis(2, 0).proj()
    op0[cv_pos] = qt.basis(N_cv, 0).proj()

    # Map CV |1> to qubit |1>
    op1 = [qt.qeye(d) for d in dims]
    op1[qubit_pos] = qt.basis(2, 1) @ qt.basis(2, 0).dag()
    op1[cv_pos] = qt.basis(2, 0) @ qt.basis(N_cv, 1).dag() # Map Fock 1 to Fock 0

    U_decode = qt.tensor(*op0) + qt.tensor(*op1)

    if system.isket:
        sm.systems_list[system_index] = U_decode @ system
    else:
        sm.systems_list[system_index] = U_decode @ system @ U_decode.dag()


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
        op_list = [qt.qeye(dims[i]) for i in range(num_subsystems)]
        
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

    # return U_encode * input_states
    if system.isket:
        sm.systems_list[system_index] = U_encode @ system
    else:
        sm.systems_list[system_index] = U_encode @ system @ U_encode.dag()

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
        op_plus = [qt.qeye(d) for d in dims]
        op_plus[cv_position] = logical_zero_cv.proj()
        # Qubit stays same (Identity)
        
        op_minus = [qt.qeye(d) for d in dims]
        op_minus[cv_position] = logical_one_cv.proj()
        op_minus[qubit_position] = qt.sigmax() # Flip if odd parity
        
        return qt.tensor(*op_plus) + qt.tensor(*op_minus)

    # 3. Step B: Qubit-Controlled Un-displacement (The "Cleaning")
    # This returns the CV mode to vacuum: |cat+>|0> -> |vac>|0> AND |cat->|1> -> |vac>|1>
    # Note: This is essentially the inverse of your encoding function.
    def build_clean():
        # This part ensures the operation is unitary by resetting the CV mode
        op_zero = [qt.qeye(d) for d in dims]
        op_zero[qubit_position] = qt.basis(2, 0).proj()
        op_zero[cv_position] = vacuum @ logical_zero_cv.dag()
        
        op_one = [qt.qeye(d) for d in dims]
        op_one[qubit_position] = qt.basis(2, 1).proj()
        op_one[cv_position] = vacuum @ logical_one_cv.dag()
        
        return qt.tensor(*op_zero) + qt.tensor(*op_one)

    # Combined Unitary: First flip the qubit, then clean the CV mode
    U_total = build_clean() @ build_flip()
    
    # Inside apply_ideal_cat_state_decoding:
    if system.isket:
        sm.systems_list[system_index] = U_total @ system
    else:
        sm.systems_list[system_index] = U_total @ system @ U_total.dag()


def apply_direct_loss(sm: StateManager, key: str, loss_prob: float):
    system_index, local_idx = sm.state_index_dict[key]
    system = sm.systems_list[system_index]
    
    # 1. Ensure we are working with a density matrix
    # Loss is a non-unitary process, so the state must become a DM
    if system.isket:
        system = qt.ket2dm(system)
    
    dims = system.dims[0]
    N = dims[local_idx]
    assert isinstance(N, int)
    
    # 2. Construct the collapse operator (A_global)
    # This acts as 'a' on the target mode and Identity everywhere else
    a = qt.destroy(N)
    op_list = [qt.qeye(d) for d in dims]
    op_list[local_idx] = a
    A_global = qt.tensor(*op_list)
    
    # 3. Solve the Lindblad master equation
    # The 'time' here represents the strength of the coupling to the environment
    t_loss = -np.log(1 - loss_prob)
    
    # H=0 because we only care about the dissipation (loss)
    # We use a simple 2-point tlist to get the final state
    result = qt.mesolve(
        H=0 * A_global, # Zero Hamiltonian
        rho0=system, 
        tlist=[0, t_loss], 
        c_ops=[A_global]
    )
    
    # 4. Update the manager with the resulting density matrix
    sm.systems_list[system_index] = result.states[-1]

def apply_kraus_loss(
    sm: StateManager,
    key: str,
    loss_prob: float,
    k_max: int | None = None
):
    system_index, local_idx = sm.state_index_dict[key]
    system = sm.systems_list[system_index]

    # Ensure density matrix
    if system.isket:
        system = qt.ket2dm(system)

    dims = system.dims[0]
    N = dims[local_idx]
    assert isinstance(N, int)

    eta = 1.0 - loss_prob
    if k_max is None:
        k_max = N - 1

    # Single-mode operators
    a = qt.destroy(N)
    n = a.dag() @ a

    # THIS is the key fix
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


def apply_hadamard(sm: StateManager, target_key: str):
    system_index, target_idx = sm.state_index_dict[target_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]

    op_list = [qt.qeye(d) for d in dims]
    
    # Place Hadamard at the target index
    op_list[target_idx] = qt.gates.snot()
    
    H_total = qt.tensor(*op_list)
    if system.isket:
        sm.systems_list[system_index] = H_total @ system
    else:
        sm.systems_list[system_index] =  H_total @ system @ H_total.dag()

def apply_cnot(sm: StateManager, control_key: str, target_key: str):
    sm.ensure_same_system(control_key, target_key)

    system_index, control_idx = sm.state_index_dict[control_key]
    _, target_idx = sm.state_index_dict[target_key]

    system = sm.systems_list[system_index]
    dims = system.dims[0]
    
    # Part 1: Control is in |0> (Identity on target)
    op_list_0 = [qt.qeye(d) for d in dims]
    op_list_0[control_idx] = qt.basis(2, 0).proj()
    # Target stays Identity, so no change needed to op_list_0
    
    # Part 2: Control is in |1> (X on target)
    op_list_1 = [qt.qeye(d) for d in dims]
    op_list_1[control_idx] = qt.basis(2, 1).proj()
    op_list_1[target_idx] = qt.sigmax()
    
    CNOT_total = qt.tensor(*op_list_0) + qt.tensor(*op_list_1)
    if system.isket:
        sm.systems_list[system_index] = CNOT_total @ system
    else:
        sm.systems_list[system_index] = CNOT_total @ system @ CNOT_total.dag()
    
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
    op_list_id = [qt.qeye(d) for d in dims]
    
    # The Toffoli gate: I + |11><11| ⊗ (X - I)
    # This only acts when both controls are in the |1> state
    proj_11 = [qt.qeye(d) for d in dims]
    proj_11[ctrl1] = qt.basis(2, 1).proj()
    proj_11[ctrl2] = qt.basis(2, 1).proj()
    
    # The operator (X - I) on the target
    op_x_minus_i = [qt.qeye(d) for d in dims]
    op_x_minus_i[target] = qt.sigmax() - qt.qeye(2)
    
    # Combine: U = Identity + (Projector_11 * Target_Flip_Logic)
    # We use element-wise multiplication of the lists to build the tensor components
    U_toffoli = qt.tensor(*op_list_id) + (qt.tensor(*proj_11) @ qt.tensor(*op_list_id).dag() @ qt.tensor(*op_x_minus_i))
    
    # Faster/Cleaner alternative for U_toffoli if dimensions are standard:
    # U_toffoli = qt.tensor(op_list_id) + qt.tensor([qt.basis(2,1).proj() if i == ctrl1 or i == ctrl2 else (qt.sigmax()-qt.qeye(2)) if i == target else qt.qeye(d) for i, d in enumerate(dims)])

    if system.isket:
        sm.systems_list[system_index] = U_toffoli @ system
    else:
        sm.systems_list[system_index] = U_toffoli @ system @ U_toffoli.dag()



def swap_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    apply_swap(sm, source_key, target_key_list[0])

def swap_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    apply_swap(sm, source_key_list[0], target_key)

def repetition_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    apply_swap(sm, source_key, target_key_list[0])
    for i in range(1, len(target_key_list)):
        target_key = target_key_list[i]
        apply_cnot(sm, target_key_list[0], target_key)

def repetition_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    if len(source_key_list) == 3:
        for i in range(1, len(source_key_list)):
            apply_cnot(sm, source_key_list[0], source_key_list[i])
        apply_toffoli(sm, source_key_list[1], source_key_list[2], source_key_list[0])
    else:
        print("unsupported decoding for n != 3")
        exit()
    apply_swap(sm, source_key_list[0], target_key)

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

    # Phase repetition on the first qubit into q[0], q[3], q[6]
    phase_repetition_encode(sm, source_key, [q[0], q[3], q[6]])

    # Bit repetition on each block
    repetition_encode(sm, q[0], [q[0], q[1], q[2]])
    repetition_encode(sm, q[3], [q[3], q[4], q[5]])
    repetition_encode(sm, q[6], [q[6], q[7], q[8]])



def shor_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    if len(source_key_list) != 9:
        raise ValueError("Shor code requires exactly 9 channel qubits")

    q = source_key_list

    # Decode bit-flip repetition in each block
    repetition_decode(sm, q[0], [q[0], q[1], q[2]])
    repetition_decode(sm, q[3], [q[3], q[4], q[5]])
    repetition_decode(sm, q[6], [q[6], q[7], q[8]])

    # Decode phase repetition across the blocks
    phase_repetition_decode(sm, target_key, [q[0], q[3], q[6]])

def wrap_repetition_encode(sm: StateManager, source_key: str, target_key_list: list[str]):
    if len(target_key_list) != 9:
        raise ValueError("wrap_repetition code requires exactly 9 channel qubits")

    q = target_key_list

    # Phase repetition on the first qubit into q[0], q[3], q[6]
    repetition_encode(sm, source_key, [q[0], q[3], q[6]])

    # Bit repetition on each block
    repetition_encode(sm, q[0], [q[0], q[1], q[2]])
    repetition_encode(sm, q[3], [q[3], q[4], q[5]])
    repetition_encode(sm, q[6], [q[6], q[7], q[8]])

def wrap_repetition_decode(sm: StateManager, target_key: str, source_key_list: list[str]):
    if len(source_key_list) != 9:
        raise ValueError("wrap_repetition code requires exactly 9 channel qubits")

    q = source_key_list

    # Decode bit-flip repetition in each block
    repetition_decode(sm, q[0], [q[0], q[1], q[2]])
    repetition_decode(sm, q[3], [q[3], q[4], q[5]])
    repetition_decode(sm, q[6], [q[6], q[7], q[8]])

    # Decode phase repetition across the blocks
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











class ChannelType(Enum):
    CV_CAT = 1
    DV_SINGLE_MODE = 2

@dataclass
class PhyLayerConfiguration:
    channel_type: ChannelType
    N: int = 2
    vertical_displacement: float | None = None

    def __post_init__(self):
        if self.channel_type == ChannelType.CV_CAT:
            if self.N is None or self.vertical_displacement is None:
                raise ValueError(
                    f"ChannelType.CV_CAT requires both 'N' and 'vertical_displacement'. "
                    f"Got: N={self.N}, displacement={self.vertical_displacement}"
                )

class EncodingType(Enum):
    SWAP_DUMMY_ENCODING = 1
    REPETITION_BIT_FLIP = 2
    REPETITION_PHASE_FLIP = 3
    SHOR_9_QUBITS = 4
    REPETITION_BIT_FLIP_WRAP = 5
    REPETITION_PHASE_FLIP_WRAP = 6

def generic_encode(sm: StateManager, source_key: str, target_key_list: list[str], encoding: EncodingType):
    if encoding is EncodingType.SWAP_DUMMY_ENCODING:
        swap_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.REPETITION_BIT_FLIP:
        repetition_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.REPETITION_PHASE_FLIP:
        phase_repetition_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.SHOR_9_QUBITS:
        shor_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.REPETITION_BIT_FLIP_WRAP:
        wrap_repetition_encode(sm, source_key, target_key_list)
    elif encoding is EncodingType.REPETITION_PHASE_FLIP_WRAP:
        phase_wrap_repetition_encode(sm, source_key, target_key_list)

def generic_decode(sm: StateManager, target_key: str, source_key_list: list[str], encoding: EncodingType):
    if encoding is EncodingType.SWAP_DUMMY_ENCODING:
        swap_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.REPETITION_BIT_FLIP:
        repetition_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.REPETITION_PHASE_FLIP:
        phase_repetition_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.SHOR_9_QUBITS:
        shor_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.REPETITION_BIT_FLIP_WRAP:
        wrap_repetition_decode(sm, target_key, source_key_list)
    elif encoding is EncodingType.REPETITION_PHASE_FLIP_WRAP:
        phase_wrap_repetition_decode(sm, target_key, source_key_list)


ideal_phi_plus = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()
ideal_rho = qt.ket2dm(ideal_phi_plus)







def run_fidelity_simulation(ph: PhyLayerConfiguration, loss_prob: float, NUM_CHANNEL_QUBITS: int, encoding_type: EncodingType) -> float:
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
        elif ph.channel_type is ChannelType.DV_SINGLE_MODE:
            sm.add_subsystem(N, f"channel_{i}_dv_mode")
            apply_single_mode_encoding(sm, f"channel_{i}_tx", f"channel_{i}_dv_mode")
            sm.ptrace_subsystem(f"channel_{i}_tx")
            apply_kraus_loss(sm, f"channel_{i}_dv_mode", loss_prob)
            apply_single_mode_decoding(sm, f"channel_{i}_rx", f"channel_{i}_dv_mode")
            sm.ptrace_subsystem(f"channel_{i}_dv_mode")
            

    sm.add_subsystem(2, "rx_edge")

    generic_decode(sm, "rx_edge", [f"channel_{i}_rx" for i in range(NUM_CHANNEL_QUBITS)], encoding_type)

    for i in range(NUM_CHANNEL_QUBITS):
        sm.ptrace_subsystem(f"channel_{i}_rx")


    edge_qubits = sm.ptrace_keep(["tx_edge", "rx_edge"])
    fid = qt.fidelity(edge_qubits, ideal_rho)
    return fid




phy_config_list: list[tuple[PhyLayerConfiguration, list[tuple[EncodingType, int]]]] = [
    (
        PhyLayerConfiguration(channel_type=ChannelType.CV_CAT, N=18, vertical_displacement=1.5), 
        [
            (EncodingType.SWAP_DUMMY_ENCODING, 1),
            (EncodingType.REPETITION_BIT_FLIP, 3),
            (EncodingType.SHOR_9_QUBITS, 9),
            (EncodingType.REPETITION_BIT_FLIP_WRAP, 9),
        ]
    ),
    (
        PhyLayerConfiguration(channel_type=ChannelType.DV_SINGLE_MODE), 
        [
            (EncodingType.SWAP_DUMMY_ENCODING, 1),
            (EncodingType.SHOR_9_QUBITS, 9),
        ]
    ),
]

loss_prob_list = np.logspace(np.log10(0.001), np.log10(0.75), num=20)


# Define line styles for different physical layers to distinguish them
styles = {
    ChannelType.CV_CAT: "solid",
    ChannelType.DV_SINGLE_MODE: "dashed"
}


for phy_config, codes in phy_config_list:
    mode_name = "CAT" if phy_config.channel_type == ChannelType.CV_CAT else "DV-1M"
    ls = styles[phy_config.channel_type]
    
    print(f"phy_config.channel_type: {phy_config.channel_type.name}")

    # For each encoding/qubit-count pair in the config
    for encoding_type, num_qubits in codes:
        fidelities = []
        label = f"[{mode_name}] {encoding_type.name} ({num_qubits} qubits)"
        
        print(f"\tencoding_type: {encoding_type.name}")
        
        for loss_prob in loss_prob_list:
            print(f"\t\tloss_prob: {loss_prob}")
            fid = run_fidelity_simulation(
                ph=phy_config, 
                loss_prob=loss_prob, 
                NUM_CHANNEL_QUBITS=num_qubits, 
                encoding_type=encoding_type
            )
            fidelities.append(fid)
            
        # Plot this specific line
        plt.loglog(loss_prob_list, fidelities, ls=ls, label=label)

# Formatting the Plot
plt.xlabel('Loss Probability')
plt.ylabel('Fidelity')
plt.title('Fidelity vs Channel Loss for Different Encodings')

# Place legend outside if it gets too crowded
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)

plt.show()