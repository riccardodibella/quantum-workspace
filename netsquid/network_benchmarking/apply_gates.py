import netsquid as ns
from netsquid.nodes import Node
from netsquid.components.models import DelayModel
from netsquid.components import QuantumChannel
from netsquid.nodes import DirectConnection
from netsquid.protocols import NodeProtocol

from netsquid.components import QuantumProcessor
from netsquid.qubits import qubitapi as qapi

from random_clifford_lib import *

m = 100
local_seed = 0
remote_seed = 2


random_clifford_set_seed(local_seed)
local_list = [get_random_clifford() for i in range(m)]

random_clifford_set_seed(remote_seed)
remote_list = [get_random_clifford() for i in range(m)]




# Create a noiseless quantum processor with 1 qubit
qprocessor = QuantumProcessor("processor", num_positions=1)

# Put qubit at position 0 (automatically initialized to |0>)
qprocessor.put(ns.qubits.create_qubits(1))

# Apply gates alternately using operate method
for i in range(m):
    # Apply local gate
    qprocessor.operate(get_operator_from_clifford(local_list[i]), positions=[0])
    
    # Apply remote gate
    qprocessor.operate(get_operator_from_clifford(remote_list[i]), positions=[0])

print(f"Applied {m} local gates and {m} remote gates")

# Get final state
qubit = qprocessor.peek([0])[0]
final_state = qapi.reduced_dm(qubit)
print("After m boundes qubit state:")
print(final_state)




# Calculate inverse of all operations
inverse_gate = ns.I  # Start with identity
for i in range(m-1, -1, -1):  # Reverse order
    inverse_gate = get_operator_from_clifford(remote_list[i].dagger) * inverse_gate
    inverse_gate = get_operator_from_clifford(local_list[i].dagger) * inverse_gate

# Apply inverse gate
qprocessor.operate(inverse_gate, positions=[0])

# Verify we're back to |0>
qubit = qprocessor.peek([0])[0]
final_state = qapi.reduced_dm(qubit)
print("State after inverse (should be |0>):")
print(final_state)


# Apply another gate (choose any Clifford gate)
measurement_basis_gate = get_random_clifford()
qprocessor.operate(get_operator_from_clifford(measurement_basis_gate), positions=[0])

print("Applied measurement basis gate")

# To measure in the new basis, we need a Hermitian observable
# Transform Z to the new basis: gate * Z * gate_dagger
gate_op = get_operator_from_clifford(measurement_basis_gate)
gate_dagger_op = get_operator_from_clifford(measurement_basis_gate.dagger)
measurement_observable = gate_op * ns.Z * gate_dagger_op

measurement_result, prob = qapi.measure(qprocessor.peek([0])[0], observable=measurement_observable)

print(f"Measurement result: {measurement_result} (error prob {1-prob})")
print(f"Expected |0> equivalent, got: {'|0>' if measurement_result == 0 else '|1>'}")