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





ideal_phi_plus = (qt.tensor(qt.basis(2,0), qt.basis(2,0)) + qt.tensor(qt.basis(2,1), qt.basis(2,1))).unit()
ideal_rho = qt.ket2dm(ideal_phi_plus)





N = 2
vertical_displacement = 1
loss_prob = 1E-2

NUM_CHANNEL_QUBITS = 2
num_tx_qubits = 2
num_rx_qubits = 1

initial_channel_states = [qt.basis(2, 0), qt.basis(N, 0), qt.basis(N, 0), qt.basis(2, 0)]
states_per_channel_qubit = len(initial_channel_states)

all_states = qt.ket2dm(qt.tensor([qt.basis(2,0)]*num_tx_qubits + initial_channel_states*NUM_CHANNEL_QUBITS + [qt.basis(2,0)]*num_rx_qubits))

def tx_qubit_positions() -> list[int]:
    return list(range(num_tx_qubits))
    
def channel_states_positions():
    starting_offset = len(tx_qubit_positions())

    pos_list = []
    for num_channel_qubit in range(NUM_CHANNEL_QUBITS):
        for i in range(states_per_channel_qubit):
            pos_list+=[starting_offset + num_channel_qubit*states_per_channel_qubit + i]
    return pos_list

def rx_qubit_positions() -> list[int]:
    # The RX block starts where the Channel block ends
    start_idx = num_tx_qubits + (NUM_CHANNEL_QUBITS * states_per_channel_qubit)
    return list(range(start_idx, start_idx + num_rx_qubits))

def ptrace_away_positions(states: qt.Qobj, positions: int | list[int]) -> qt.Qobj:
    if type(positions) is int:
        positions = [positions]

    keep_indices = [i for i in range(len(all_states.dims[0])) if i not in positions]
    return states.ptrace(keep_indices)




state_index_dict = {}
all_states: qt.Qobj | None = None

def add_system(dimensions, key):
    global all_states, state_index_dict
    num_systems = 0 if all_states is None else len(all_states.dims[0])
    state_index_dict[key] = num_systems

    if all_states is None:
        all_states = qt.basis(dimensions, 0)
    elif all_states.isket:
        all_states = qt.tensor(all_states, qt.basis(dimensions, 0))
    else:
        all_states = qt.tensor(all_states, qt.ket2dm(qt.basis(dimensions, 0)))

def ptrace_system(key):
    global all_states, state_index_dict
    if(all_states is None):
        raise Exception("ptrace_system None state")
    index = state_index_dict[key]
    num_systems = 0 if all_states is None else len(all_states.dims[0])
    all_states = all_states.ptrace()



















all_states = apply_hadamard(all_states, tx_qubit_positions()[0])
all_states = apply_cnot(all_states, tx_qubit_positions()[0], tx_qubit_positions()[1])

all_states = apply_swap(all_states, tx_qubit_positions()[1], channel_states_positions()[0])

for channel_qubit_index in range(NUM_CHANNEL_QUBITS):
    all_states = apply_cat_state_encoding(all_states, channel_states_positions()[channel_qubit_index*states_per_channel_qubit+0], channel_states_positions()[channel_qubit_index*states_per_channel_qubit+1], vertical_displacement, N)


for channel_qubit_index in range(NUM_CHANNEL_QUBITS):
    all_states = beamsplitter_general(all_states, channel_states_positions()[channel_qubit_index*states_per_channel_qubit+1], channel_states_positions()[channel_qubit_index*states_per_channel_qubit+2], loss_prob)

for channel_qubit_index in range(NUM_CHANNEL_QUBITS):
    all_states = apply_ideal_cat_state_decoding(all_states, channel_states_positions()[channel_qubit_index*states_per_channel_qubit+3], channel_states_positions()[channel_qubit_index*states_per_channel_qubit+1], vertical_displacement, N)

all_states = apply_swap(all_states, channel_states_positions()[states_per_channel_qubit-1], rx_qubit_positions()[0])
edge_qubits = all_states.ptrace([tx_qubit_positions()[0], rx_qubit_positions()[0]])

print(edge_qubits)
fid = qt.fidelity(edge_qubits, ideal_rho)
print(f"Fidelity with Phi+ (ideal): {fid:.4f}")