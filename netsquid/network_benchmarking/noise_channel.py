import netsquid as ns
from netsquid.nodes import Node
from netsquid.components.models import FibreDelayModel
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components import QuantumChannel
from netsquid.nodes import DirectConnection
from netsquid.protocols import NodeProtocol

from netsquid.components import QuantumProcessor
from netsquid.qubits import qubitapi as qapi


from random_clifford_lib import *


class LocalPingPongProtocol(NodeProtocol):
    def __init__(self, node, m, local_seed, remote_seed):
        super().__init__(node)

        self.m = m
        random_clifford_set_seed(local_seed)
        self.local_clifford_list = [get_random_clifford() for i in range(m)]
        random_clifford_set_seed(remote_seed)
        self.remote_clifford_list = [get_random_clifford() for i in range(m)]

        self.qprocessor = QuantumProcessor("loc_processor", num_positions=1)
        self.node.add_subcomponent(self.qprocessor)
        self.node.ports["qubitIO"].forward_input(self.qprocessor.ports["qin"])
        self.qprocessor.ports["qout"].forward_output(self.node.ports["qubitIO"])

    def run(self):

        # Put qubit at position 0 (automatically initialized to |0>)
        self.qprocessor.put(ns.qubits.create_qubits(1))
        self.qprocessor.pop([0])

        for i in range(self.m):
            self.qprocessor.pop([0])
            yield self.await_port_input(self.qprocessor.ports["qin"])
            self.qprocessor.operate(get_operator_from_clifford(self.local_clifford_list[i]), positions=[0])
        
        # Calculate inverse of all operations
        inverse_gate = ns.I  # Start with identity
        for i in range(self.m-1, -1, -1):  # Reverse order
            inverse_gate = get_operator_from_clifford(self.local_clifford_list[i].dagger) * inverse_gate
            inverse_gate = get_operator_from_clifford(self.remote_clifford_list[i].dagger) * inverse_gate

        # Apply inverse gate
        self.qprocessor.operate(inverse_gate, positions=[0])

        # Apply another gate (choose any Clifford gate)
        measurement_basis_gate = get_random_clifford()
        self.qprocessor.operate(get_operator_from_clifford(measurement_basis_gate), positions=[0])

        # To measure in the new basis, we need a Hermitian observable
        # Transform Z to the new basis: gate * Z * gate_dagger
        gate_op = get_operator_from_clifford(measurement_basis_gate)
        gate_dagger_op = get_operator_from_clifford(measurement_basis_gate.dagger)
        measurement_observable = gate_op * ns.Z * gate_dagger_op

        measurement_result, prob = qapi.measure(self.qprocessor.peek([0])[0], observable=measurement_observable)

        print(f"Measurement result: {measurement_result}")
        print(f"Expected |0> equivalent, got: {'|0>' if measurement_result == 0 else '|1>'}")

class RemotePingPongProtocol(NodeProtocol):
    def __init__(self, node, m, seed):
        super().__init__(node)
        self.m = m
        random_clifford_set_seed(seed)
        self.clifford_list = [get_random_clifford() for i in range(m)]

        self.qprocessor = QuantumProcessor("rem_processor", num_positions=1)
        self.node.add_subcomponent(self.qprocessor)
        self.node.ports["qubitIO"].forward_input(self.qprocessor.ports["qin"])
        self.qprocessor.ports["qout"].forward_output(self.node.ports["qubitIO"])

    def run(self):

        for i in range(self.m):
            yield self.await_port_input(self.qprocessor.ports["qin"])
            self.qprocessor.operate(get_operator_from_clifford(self.clifford_list[i]), positions=[0])
            self.qprocessor.pop([0])



        

    


m = 100
local_seed = 0
remote_seed = 2



loc_node = Node(name="Ping")
rem_node = Node(name="Pong")

distance = 1000 #km
depolar_rate = 1 # Depolarization rate in Hz
fibre_delay_model = FibreDelayModel()
fibre_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
channel_1 = QuantumChannel("ch1", length=distance, models={"delay_model": fibre_delay_model, "quantum_noise_model": fibre_noise_model})
channel_2 = QuantumChannel("ch2", length=distance, models={"delay_model": fibre_delay_model, "quantum_noise_model": fibre_noise_model})


connection = DirectConnection("conn", channel_AtoB=channel_1, channel_BtoA=channel_2)
loc_node.connect_to(remote_node=rem_node, connection=connection, local_port_name="qubitIO", remote_port_name="qubitIO")

loc_pr = LocalPingPongProtocol(loc_node, m, local_seed=local_seed, remote_seed=remote_seed)
rem_pr = RemotePingPongProtocol(rem_node, m, remote_seed)

rem_pr.start()
loc_pr.start()
run_stats = ns.sim_run(duration=1e12)

#print(run_stats)
