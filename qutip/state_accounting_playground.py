import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from math import *
import qutip as qt
import qutip.qip
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
    # 1. Get the global dimensions of the system
    # dims will be something like [2, 2, 20, 20] (Qubit0, Qubit1, CV0, CV1)
    dims = input_state.dims[0]
    num_subsystems = len(dims)
    
    # 2. Extract cutoff dimensions for the two target CV modes
    N1 = dims[idx1]
    N2 = dims[idx2]
    
    # 3. Create the annihilation operators in the full Hilbert space
    # We start with a list of identities for every subsystem
    op_list1 = [qt.qeye(d) for d in dims]
    op_list2 = [qt.qeye(d) for d in dims]
    
    # Replace the identities at the target indices with destroy operators
    op_list1[idx1] = qt.destroy(N1)
    op_list2[idx2] = qt.destroy(N2)
    
    # Tensor them together to get operators acting on the full system
    a1 = qt.tensor(op_list1)
    a2 = qt.tensor(op_list2)

    # 4. Calculate mixing angle
    theta = np.arcsin(np.sqrt(transmissivity))

    # 5. Build the Unitary for the full space
    # U = exp( theta * (a1^dag a2 - a1 a2^dag) )
    generator = theta * (a1.dag() * a2 - a1 * a2.dag())
    U_bs = generator.expm()

    # 6. Apply and return
    if input_state.isket:
        return U_bs * input_state
    else:
        return U_bs * input_state * U_bs.dag()

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


def repetition_encode(state: qt.Qobj, source_index: int, target_index_list: list[int]) -> qt.Qobj:
    state = apply_swap(state, source_index, target_index_list[0])
    for i in range(1, len(target_index_list)):
        target_index = target_index_list[i]
        state = apply_cnot(state, target_index_list[0], target_index)
    return state

def repetition_decode(state: qt.Qobj, target_index: int, source_index_list: list[int]) -> qt.Qobj:
    # 1. Map the error syndromes
    # We use source_index_list[0] as the 'main' qubit.
    # We CNOT it into the others to see if they differ.
    for i in range(1, len(source_index_list)):
        state = apply_cnot(state, source_index_list[0], source_index_list[i])
    
    # 2. Majority Vote (The Correction Step)
    # If source_index_list[1] AND source_index_list[2] are both 1, 
    # it means the 'main' qubit (index 0) is the one that actually flipped.
    if len(source_index_list) == 3:
        state = apply_toffoli(state, source_index_list[1], source_index_list[2], source_index_list[0])
    else:
        print("unsupported decoding for n != 3")
        exit()
    
    # 3. Transfer the corrected state to the target (rx_edge)
    state = apply_swap(state, source_index_list[0], target_index)
    
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




ideal_phi_plus = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()
ideal_rho = qt.ket2dm(ideal_phi_plus)





N = 20
vertical_displacement = 2
loss_prob = 0.2



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


NUM_CHANNEL_QUBITS = 3

# all_states = qt.ket2dm(qt.tensor([qt.basis(2,0)]*num_tx_qubits + initial_channel_states*NUM_CHANNEL_QUBITS + [qt.basis(2,0)]*num_rx_qubits))
add_subsystem(2, "tx_edge")
add_subsystem(2, "tx_temp")

all_states = apply_hadamard(all_states, state_index_dict["tx_edge"])
all_states = apply_cnot(all_states, state_index_dict["tx_edge"], state_index_dict["tx_temp"])

for i in range(NUM_CHANNEL_QUBITS):
    add_subsystem(2, f"channel_{i}_tx")

#all_states = apply_swap(all_states, state_index_dict["tx_temp"], state_index_dict[f"channel_{0}_tx"])
all_states = repetition_encode(all_states, state_index_dict["tx_temp"], [state_index_dict[f"channel_{0}_tx"], state_index_dict[f"channel_{1}_tx"], state_index_dict[f"channel_{2}_tx"]])

ptrace_subsystem("tx_temp")

for i in range(NUM_CHANNEL_QUBITS):
    add_subsystem(N, f"channel_{i}_cat")
    all_states = apply_cat_state_encoding(all_states, state_index_dict[f"channel_{i}_tx"], state_index_dict[f"channel_{i}_cat"], vertical_displacement, N)
    ptrace_subsystem(f"channel_{i}_tx")
    add_subsystem(N, f"channel_{i}_vacuum")
    all_states = beamsplitter_general(all_states, state_index_dict[f"channel_{i}_cat"], state_index_dict[f"channel_{i}_vacuum"], loss_prob)
    add_subsystem(2, f"channel_{i}_rx")
    all_states = apply_ideal_cat_state_decoding(all_states, state_index_dict[f"channel_{i}_rx"], state_index_dict[f"channel_{i}_cat"], vertical_displacement, N)
    ptrace_subsystem(f"channel_{i}_cat")
    ptrace_subsystem(f"channel_{i}_vacuum")

add_subsystem(2, "rx_edge")

#all_states = apply_swap(all_states, state_index_dict[f"channel_{0}_rx"], state_index_dict["rx_edge"])
all_states = repetition_decode(all_states, state_index_dict["rx_edge"], [state_index_dict[f"channel_{0}_rx"], state_index_dict[f"channel_{1}_rx"], state_index_dict[f"channel_{2}_rx"]])

for i in range(NUM_CHANNEL_QUBITS):
    ptrace_subsystem(f"channel_{i}_rx")


edge_qubits = all_states.ptrace([state_index_dict["tx_edge"], state_index_dict["rx_edge"]])

print(edge_qubits)
fid = qt.fidelity(edge_qubits, ideal_rho)
print(f"Fidelity with Phi+ (ideal): {fid:.4f}")