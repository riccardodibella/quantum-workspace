import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from math import *
import qutip as qt
import qutip.qip
from enum import Enum

# set a parameter to see animations in line
from matplotlib import rc
rc('animation', html='jshtml')

# static image plots
# %matplotlib inline
# interactive 3D plots
# %matplotlib widget



def apply_cat_state_encoding(input_states: qt.Qobj, qubit_position: int, cv_position: int, vertical_displacement=2.5, N=20) -> qt.Qobj:
    # 1. Prepare CV states
    vacuum = qt.basis(N, 0)
    alpha_coeff = (vertical_displacement / np.sqrt(2)) * 1j
    
    pos_disp = qt.displace(N, alpha_coeff)
    neg_disp = qt.displace(N, -alpha_coeff)
    
    logical_zero = (pos_disp * vacuum + neg_disp * vacuum).unit()
    logical_one  = (pos_disp * vacuum - neg_disp * vacuum).unit()
    
    # Define the mapping operators for the CV mode
    map_zero = logical_zero * vacuum.dag()
    map_one  = logical_one * vacuum.dag()

    # 2. Build the operator list for the tensor product
    dims = input_states.dims[0] 
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
            op_list[qubit_position] = qt.basis(2, 0) * qt.basis(2, 1).dag()
            op_list[cv_position] = map_one
        
        return qt.tensor(op_list)

    # 3. Combine into the full encoding operator
    U_encode = build_gate(0) + build_gate(1)

    # return U_encode * input_states
    if input_states.isket:
        return U_encode * input_states
    else:
        return U_encode * input_states * U_encode.dag()

def apply_ideal_cat_state_decoding(input_states: qt.Qobj, qubit_position: int, cv_position: int, vertical_displacement=2.5, N=20) -> qt.Qobj:
    dims = input_states.dims[0]
    
    # 1. Define states
    vacuum = qt.basis(N, 0)
    alpha_coeff = (vertical_displacement / np.sqrt(2)) * 1j
    
    # Define logical states for parity detection
    pos_disp = qt.displace(N, alpha_coeff)
    neg_disp = qt.displace(N, -alpha_coeff)
    logical_zero_cv = (pos_disp * vacuum + neg_disp * vacuum).unit()
    logical_one_cv  = (pos_disp * vacuum - neg_disp * vacuum).unit()

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
        
        return qt.tensor(op_plus) + qt.tensor(op_minus)

    # 3. Step B: Qubit-Controlled Un-displacement (The "Cleaning")
    # This returns the CV mode to vacuum: |cat+>|0> -> |vac>|0> AND |cat->|1> -> |vac>|1>
    # Note: This is essentially the inverse of your encoding function.
    def build_clean():
        # This part ensures the operation is unitary by resetting the CV mode
        op_zero = [qt.qeye(d) for d in dims]
        op_zero[qubit_position] = qt.basis(2, 0).proj()
        op_zero[cv_position] = vacuum * logical_zero_cv.dag()
        
        op_one = [qt.qeye(d) for d in dims]
        op_one[qubit_position] = qt.basis(2, 1).proj()
        op_one[cv_position] = vacuum * logical_one_cv.dag()
        
        return qt.tensor(op_zero) + qt.tensor(op_one)

    # Combined Unitary: First flip the qubit, then clean the CV mode
    U_total = build_clean() * build_flip()
    
    # Inside apply_ideal_cat_state_decoding:
    if input_states.isket:
        return U_total * input_states
    else:
        return U_total * input_states * U_total.dag()

def beamsplitter_general(input_state: qt.Qobj, idx1: int, idx2: int, transmissivity: float) -> qt.Qobj:
    dims = input_state.dims[0]
    num_subsystems = len(dims)
    
    # 1. Define local subspace dimensions
    N1 = dims[idx1]
    N2 = dims[idx2]
    
    # 2. Compute the Beam Splitter Unitary on the small (N x N) subsystem
    #    This is fast (e.g., 16x16 matrix for N=4)
    a1_loc = qt.tensor(qt.destroy(N1), qt.qeye(N2))
    a2_loc = qt.tensor(qt.qeye(N1), qt.destroy(N2))
    
    theta = np.arcsin(np.sqrt(transmissivity))
    generator = theta * (a1_loc.dag() * a2_loc - a1_loc * a2_loc.dag())
    U_local = generator.expm()

    # 3. Prepare Permutation
    #    Identify indices that are NOT the beam splitter modes
    other_indices = [i for i in range(num_subsystems) if i != idx1 and i != idx2]
    
    #    Calculate the dimension of the "rest" of the system
    dim_rest = 1
    for idx in other_indices:
        dim_rest *= dims[idx]
        
    #    Order: [Rest of System] followed by [Mode 1, Mode 2]
    perm_order = other_indices + [idx1, idx2]
    
    #    Inverse permutation to restore order later
    inv_perm_order = np.argsort(perm_order).tolist()

    # 4. Permute the Input State
    state_permuted = input_state.permute(perm_order)

    # 5. Construct the Global Unitary
    #    We create (Identity_Rest) ⊗ (U_local)
    #    Initially, this has dims [[dim_rest, N1, N2], [dim_rest, N1, N2]]
    #    We must FIX the dims to match the individual qubits of state_permuted
    U_global_ordered = qt.tensor(qt.qeye(dim_rest), U_local)
    
    #    CRITICAL FIX: Force the unitary to have the detailed [2, 2, ..., 4, 4] structure
    #    so QuTiP allows the multiplication.
    target_dims = state_permuted.dims[0]
    U_global_ordered.dims = [target_dims, target_dims]

    # 6. Apply Unitary
    if input_state.isket:
        result_permuted = U_global_ordered * state_permuted
    else:
        result_permuted = U_global_ordered * state_permuted * U_global_ordered.dag()

    # 7. Restore original order
    return result_permuted.permute(inv_perm_order)


def apply_hadamard(state: qt.Qobj, target_idx: int) -> qt.Qobj:
    dims = state.dims[0]
    op_list = [qt.qeye(d) for d in dims]
    
    # Place Hadamard at the target index
    op_list[target_idx] = qt.gates.snot()
    
    H_total = qt.tensor(op_list)
    if state.isket:
        return H_total * state
    else:
        return H_total * state * H_total.dag()

def apply_cnot(state: qt.Qobj, control_idx: int, target_idx: int) -> qt.Qobj:
    dims = state.dims[0]
    
    # Part 1: Control is in |0> (Identity on target)
    op_list_0 = [qt.qeye(d) for d in dims]
    op_list_0[control_idx] = qt.basis(2, 0).proj()
    # Target stays Identity, so no change needed to op_list_0
    
    # Part 2: Control is in |1> (X on target)
    op_list_1 = [qt.qeye(d) for d in dims]
    op_list_1[control_idx] = qt.basis(2, 1).proj()
    op_list_1[target_idx] = qt.sigmax()
    
    CNOT_total = qt.tensor(op_list_0) + qt.tensor(op_list_1)
    if state.isket:
        return CNOT_total * state
    else:
        return CNOT_total * state * CNOT_total.dag()
    
def apply_swap(state: qt.Qobj, idx1: int, idx2: int) -> qt.Qobj:
    # A SWAP is 3 CNOTs
    state = apply_cnot(state, idx1, idx2)
    state = apply_cnot(state, idx2, idx1)
    state = apply_cnot(state, idx1, idx2)
    return state

def apply_toffoli(state: qt.Qobj, ctrl1: int, ctrl2: int, target: int) -> qt.Qobj:
    dims = state.dims[0]
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
    U_toffoli = qt.tensor(op_list_id) + (qt.tensor(proj_11) * qt.tensor(op_list_id).dag() * qt.tensor(op_x_minus_i))
    
    # Faster/Cleaner alternative for U_toffoli if dimensions are standard:
    # U_toffoli = qt.tensor(op_list_id) + qt.tensor([qt.basis(2,1).proj() if i == ctrl1 or i == ctrl2 else (qt.sigmax()-qt.qeye(2)) if i == target else qt.qeye(d) for i, d in enumerate(dims)])

    if state.isket:
        return U_toffoli * state
    else:
        return U_toffoli * state * U_toffoli.dag()



def swap_encode(state: qt.Qobj, source_index: int, target_index_list: list[int]) -> qt.Qobj:
    state = apply_swap(state, source_index, target_index_list[0])
    return state

def swap_decode(state: qt.Qobj, target_index: int, source_index_list: list[int]) -> qt.Qobj:
    state = apply_swap(state, source_index_list[0], target_index)
    return state

def repetition_encode(state: qt.Qobj, source_index: int, target_index_list: list[int]) -> qt.Qobj:
    state = apply_swap(state, source_index, target_index_list[0])
    for i in range(1, len(target_index_list)):
        target_index = target_index_list[i]
        state = apply_cnot(state, target_index_list[0], target_index)
    return state

def repetition_decode(state: qt.Qobj, target_index: int, source_index_list: list[int]) -> qt.Qobj:
    for i in range(1, len(source_index_list)):
        state = apply_cnot(state, source_index_list[0], source_index_list[i])
    if len(source_index_list) == 3:
        state = apply_toffoli(state, source_index_list[1], source_index_list[2], source_index_list[0])
    else:
        print("unsupported decoding for n != 3")
        exit()
    state = apply_swap(state, source_index_list[0], target_index)
    return state

def phase_repetition_encode(state: qt.Qobj, source_index: int, target_index_list: list[int]) -> qt.Qobj:
    state = apply_hadamard(state, source_index)
    state = repetition_encode(state, source_index, target_index_list)
    for idx in target_index_list:
        state = apply_hadamard(state, idx)
    return state

def phase_repetition_decode(state: qt.Qobj, target_index: int, source_index_list: list[int]) -> qt.Qobj:
    for idx in source_index_list:
        state = apply_hadamard(state, idx)
    state = repetition_decode(state, target_index, source_index_list)
    state = apply_hadamard(state, target_index)
    return state

def shor_encode(state: qt.Qobj, source_index: int, target_index_list: list[int]) -> qt.Qobj:
    if len(target_index_list) != 9:
        raise ValueError("Shor code requires exactly 9 channel qubits")

    # 1. Move logical state into the first qubit of the block (q0)
    q0 = target_index_list[0]
    state = apply_swap(state, source_index, q0)

    # Indices for clarity
    q = target_index_list # q[0]...q[8]

    # 2. Phase-flip protection encoding (Repetition in Z basis)
    # CNOT logical qubit to leaders of block 2 (q3) and block 3 (q6)
    state = apply_cnot(state, q[0], q[3])
    state = apply_cnot(state, q[0], q[6])

    # 3. Hadamard on leaders (basis change for inner bit-flip codes)
    state = apply_hadamard(state, q[0])
    state = apply_hadamard(state, q[3])
    state = apply_hadamard(state, q[6])

    # 4. Bit-flip protection encoding (Repetition in X basis for each block)
    # Block 1
    state = apply_cnot(state, q[0], q[1])
    state = apply_cnot(state, q[0], q[2])
    # Block 2
    state = apply_cnot(state, q[3], q[4])
    state = apply_cnot(state, q[3], q[5])
    # Block 3
    state = apply_cnot(state, q[6], q[7])
    state = apply_cnot(state, q[6], q[8])

    return state

def shor_decode(state: qt.Qobj, target_index: int, source_index_list: list[int]) -> qt.Qobj:
    if len(source_index_list) != 9:
        raise ValueError("Shor code requires exactly 9 channel qubits")
    
    q = source_index_list

    # --- Level 1: Bit-flip correction (inner layer) ---
    
    # Block 1 (q0, q1, q2) - Correct q0
    state = apply_cnot(state, q[0], q[1])
    state = apply_cnot(state, q[0], q[2])
    state = apply_toffoli(state, q[1], q[2], q[0])

    # Block 2 (q3, q4, q5) - Correct q3
    state = apply_cnot(state, q[3], q[4])
    state = apply_cnot(state, q[3], q[5])
    state = apply_toffoli(state, q[4], q[5], q[3])

    # Block 3 (q6, q7, q8) - Correct q6
    state = apply_cnot(state, q[6], q[7])
    state = apply_cnot(state, q[6], q[8])
    state = apply_toffoli(state, q[7], q[8], q[6])

    # --- Level 2: Phase-flip correction (outer layer) ---

    # Basis change back to computational
    state = apply_hadamard(state, q[0])
    state = apply_hadamard(state, q[3])
    state = apply_hadamard(state, q[6])

    # Correct phase flip on q0 using q3 and q6
    state = apply_cnot(state, q[0], q[3])
    state = apply_cnot(state, q[0], q[6])
    state = apply_toffoli(state, q[3], q[6], q[0])

    # --- Level 3: Transfer to RX edge ---
    state = apply_swap(state, q[0], target_index)

    return state



class EncodingType(Enum):
    SWAP_DUMMY_ENCODING = 1
    REPETITION_BIT_FLIP = 2
    REPETITION_PHASE_FLIP = 3
    SHOR_9_QUBITS = 4

def generic_encode(state: qt.Qobj, source_index: int, target_index_list: list[int], encoding: EncodingType) -> qt.Qobj:
    if encoding is EncodingType.SWAP_DUMMY_ENCODING:
        return swap_encode(state, source_index, target_index_list)
    if encoding is EncodingType.REPETITION_BIT_FLIP:
        return repetition_encode(state, source_index, target_index_list)
    if encoding is EncodingType.REPETITION_PHASE_FLIP:
        return phase_repetition_encode(state, source_index, target_index_list)
    if encoding is EncodingType.SHOR_9_QUBITS:
        return shor_encode(state, source_index, target_index_list)

def generic_decode(state: qt.Qobj, target_index: int, source_index_list: list[int], encoding: EncodingType) -> qt.Qobj:
    if encoding is EncodingType.SWAP_DUMMY_ENCODING:
        return swap_decode(state, target_index, source_index_list)
    if encoding is EncodingType.REPETITION_BIT_FLIP:
        return repetition_decode(state, target_index, source_index_list)
    if encoding is EncodingType.REPETITION_PHASE_FLIP:
        return phase_repetition_decode(state, target_index, source_index_list)
    if encoding is EncodingType.SHOR_9_QUBITS:
        return shor_decode(state, target_index, source_index_list)


ideal_phi_plus = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()
ideal_rho = qt.ket2dm(ideal_phi_plus)

state_index_dict = {}
all_states: qt.Qobj | None = None

def add_subsystem(dimensions, key):
    global all_states, state_index_dict
    num_systems = 0 if all_states is None else len(all_states.dims[0])
    state_index_dict[key] = num_systems

    if all_states is None:
        all_states = qt.basis(dimensions, 0)
    elif all_states.isket:
        all_states = qt.tensor(all_states, qt.basis(dimensions, 0))
    else:
        all_states = qt.tensor(all_states, qt.ket2dm(qt.basis(dimensions, 0)))

def ptrace_subsystem(key):
    global all_states, state_index_dict
    if(all_states is None):
        raise Exception("ptrace_system None state")
    index = state_index_dict[key]
    num_systems = 0 if all_states is None else len(all_states.dims[0])
    all_states = all_states.ptrace([i for i in range(num_systems) if i != index])

    del state_index_dict[key]
    for k, v in state_index_dict.items():
        if v > index:
            state_index_dict[k] = v-1







def run_fidelity_simulation(N: int, vertical_displacement: float, loss_prob: float, NUM_CHANNEL_QUBITS: int, encoding_type: EncodingType) -> float:
    global all_states, state_index_dict
    state_index_dict = {}
    all_states = None



    # all_states = qt.ket2dm(qt.tensor([qt.basis(2,0)]*num_tx_qubits + initial_channel_states*NUM_CHANNEL_QUBITS + [qt.basis(2,0)]*num_rx_qubits))
    add_subsystem(2, "tx_edge")
    add_subsystem(2, "tx_temp")

    all_states = apply_hadamard(all_states, state_index_dict["tx_edge"])
    all_states = apply_cnot(all_states, state_index_dict["tx_edge"], state_index_dict["tx_temp"])

    for i in range(NUM_CHANNEL_QUBITS):
        add_subsystem(2, f"channel_{i}_tx")

    #all_states = apply_swap(all_states, state_index_dict["tx_temp"], state_index_dict[f"channel_{0}_tx"])
    all_states = generic_encode(all_states, state_index_dict["tx_temp"], [state_index_dict[f"channel_{i}_tx"] for i in range(NUM_CHANNEL_QUBITS)], encoding_type)

    ptrace_subsystem("tx_temp")

    for i in range(NUM_CHANNEL_QUBITS):
        add_subsystem(N, f"channel_{i}_cat")
        all_states = apply_cat_state_encoding(all_states, state_index_dict[f"channel_{i}_tx"], state_index_dict[f"channel_{i}_cat"], vertical_displacement, N)
        ptrace_subsystem(f"channel_{i}_tx")
        add_subsystem(N, f"channel_{i}_vacuum")
        all_states = beamsplitter_general(all_states, state_index_dict[f"channel_{i}_cat"], state_index_dict[f"channel_{i}_vacuum"], loss_prob)
        ptrace_subsystem(f"channel_{i}_vacuum")
        add_subsystem(2, f"channel_{i}_rx")
        all_states = apply_ideal_cat_state_decoding(all_states, state_index_dict[f"channel_{i}_rx"], state_index_dict[f"channel_{i}_cat"], vertical_displacement, N)
        ptrace_subsystem(f"channel_{i}_cat")

    add_subsystem(2, "rx_edge")

    #all_states = apply_swap(all_states, state_index_dict[f"channel_{0}_rx"], state_index_dict["rx_edge"])
    all_states = generic_decode(all_states, state_index_dict["rx_edge"], [state_index_dict[f"channel_{i}_rx"] for i in range(NUM_CHANNEL_QUBITS)], encoding_type)

    for i in range(NUM_CHANNEL_QUBITS):
        ptrace_subsystem(f"channel_{i}_rx")


    edge_qubits = all_states.ptrace([state_index_dict["tx_edge"], state_index_dict["rx_edge"]])

    fid = qt.fidelity(edge_qubits, ideal_rho)
    return fid



N = 18
vertical_displacement = 1.5


loss_prob_list = np.logspace(np.log10(0.75), np.log10(0.01), num=6)

res_shor_list = []
res_bit_repetition_list = []
res_phase_repetition_list = []
res_swap_list = []

for loss_prob in loss_prob_list:
    print(f"{loss_prob}")
    res_shor_list += [run_fidelity_simulation(N, vertical_displacement, loss_prob, NUM_CHANNEL_QUBITS=9, encoding_type=EncodingType.SHOR_9_QUBITS)]
    res_bit_repetition_list += [run_fidelity_simulation(N, vertical_displacement, loss_prob, NUM_CHANNEL_QUBITS=3, encoding_type=EncodingType.REPETITION_BIT_FLIP)]
    res_phase_repetition_list += [run_fidelity_simulation(N, vertical_displacement, loss_prob, NUM_CHANNEL_QUBITS=3, encoding_type=EncodingType.REPETITION_PHASE_FLIP)]
    res_swap_list += [run_fidelity_simulation(N, vertical_displacement, loss_prob, NUM_CHANNEL_QUBITS=1, encoding_type=EncodingType.SWAP_DUMMY_ENCODING)]


# Add 'label' to each plot, and markers (o, s, ^) to distinguish points
plt.loglog(loss_prob_list, res_shor_list,    'o-', label='Shor Code (9 qubits)')
plt.loglog(loss_prob_list, res_bit_repetition_list, 's-', label='Bit Flip Repetition Code (3 qubits)')
plt.loglog(loss_prob_list, res_phase_repetition_list, 's-', label='Phase Flip Repetition Code (3 qubits)')
plt.loglog(loss_prob_list, res_swap_list,    '^-', label='No Encoding (1 qubit)')

# Add axis labels and title
plt.xlabel('Loss Probability')
plt.ylabel('Fidelity')
plt.title('Fidelity vs Channel Loss for Different Encodings')

# Enable the legend
plt.legend()

# Add a grid (highly recommended for log plots)
plt.grid(True, which="both", ls="--", alpha=0.6)

plt.show()