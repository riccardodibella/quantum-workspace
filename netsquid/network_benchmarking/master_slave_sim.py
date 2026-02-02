import numpy as np
import matplotlib.pyplot as plt
import netsquid as ns
from netsquid.nodes import Node
from netsquid.components.models import FibreDelayModel
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components import QuantumChannel, ClassicalChannel
from netsquid.nodes import DirectConnection
from netsquid.protocols import NodeProtocol
from netsquid.components import QuantumProcessor
from netsquid.qubits import qubitapi as qapi

from random_clifford_lib import *

seed_rng = np.random.default_rng(None)

class RemoteBenchmarkingProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)
        self.m = None
        self.clifford_list = None

        self.qprocessor = QuantumProcessor("rem_processor", num_positions=1)
        self.node.add_subcomponent(self.qprocessor)
        self.node.ports["qubitIO"].forward_input(self.qprocessor.ports["qin"])
        self.qprocessor.ports["qout"].forward_output(self.node.ports["qubitIO"])

    def run(self):
        while(True):
            # Wait for initialization message from local node
            yield self.await_port_input(self.node.ports["classicalIO"])
            message = self.node.ports["classicalIO"].rx_input().items[0]
            self.m = message['m']
            seed = message['seed']

            cmd = {"type": "ACK"}
            self.node.ports["classicalIO"].tx_output(cmd)
            
            # Initialize Clifford list with received seed
            random_clifford_set_seed(seed)
            self.clifford_list = [get_random_clifford() for i in range(self.m)]
            
            for i in range(self.m):
                yield self.await_port_input(self.qprocessor.ports["qin"])
                self.qprocessor.operate(get_operator_from_clifford(self.clifford_list[i]), positions=[0])
                self.qprocessor.pop([0])
            
            self.m = None
            self.clifford_list = None


class LocalBenchmarkingProtocol(NodeProtocol):
    def __init__(self, node, test_items):
        super().__init__(node)

        self.test_items = test_items # array of tuples (m, count) where count is the number of measures to take for that m
        self.results = []

        self.qprocessor = QuantumProcessor("loc_processor", num_positions=1)
        self.node.add_subcomponent(self.qprocessor)
        self.node.ports["qubitIO"].forward_input(self.qprocessor.ports["qin"])
        self.qprocessor.ports["qout"].forward_output(self.node.ports["qubitIO"])

    def run(self):
        for m, tot_count in self.test_items:
            tot = 0
            correct = 0
            wrong = 0

            for cur_meas in range(tot_count):
                local_seed = seed_rng.integers(2**32, dtype=np.uint32)
                remote_seed = seed_rng.integers(2**32, dtype=np.uint32)
                random_clifford_set_seed(remote_seed)
                self.remote_clifford_list = [get_random_clifford() for i in range(m)]
                random_clifford_set_seed(local_seed)
                self.local_clifford_list = [get_random_clifford() for i in range(m)]
                


                cmd = {"type": "CMD", 'm': m, 'seed': remote_seed}
                self.node.ports["classicalIO"].tx_output(cmd)

                yield self.await_port_input(self.node.ports["classicalIO"])
                _ = self.node.ports["classicalIO"].rx_input().items[0]

                # Put qubit at position 0 (automatically initialized to |0>)
                self.qprocessor.put(ns.qubits.create_qubits(1))
                self.qprocessor.pop([0])

                for i in range(m):
                    self.qprocessor.pop([0])
                    yield self.await_port_input(self.qprocessor.ports["qin"])
                    self.qprocessor.operate(get_operator_from_clifford(self.local_clifford_list[i]), positions=[0])
                
                # Calculate inverse of all operations
                inverse_gate = ns.I  # Start with identity
                for i in range(m-1, -1, -1):  # Reverse order
                    inverse_gate = get_operator_from_clifford(self.local_clifford_list[i].dagger) * inverse_gate
                    inverse_gate = get_operator_from_clifford(self.remote_clifford_list[i].dagger) * inverse_gate

                # Apply inverse gate
                self.qprocessor.operate(inverse_gate, positions=[0])

                # Apply another gate (choose any Clifford gate)
                while True:
                    measurement_basis_gate = get_random_clifford()
                    if measurement_basis_gate not in [ns.qubits.cliffords.CLIFF_SX, ns.qubits.cliffords.CLIFF_SY, ns.qubits.cliffords.CLIFF_SZ]:
                        break
                self.qprocessor.operate(get_operator_from_clifford(measurement_basis_gate), positions=[0])

                # To measure in the new basis, we need a Hermitian observable
                # Transform Z to the new basis: gate * Z * gate_dagger
                gate_op = get_operator_from_clifford(measurement_basis_gate)
                gate_dagger_op = get_operator_from_clifford(measurement_basis_gate.dagger)
                measurement_observable = gate_op * ns.Z * gate_dagger_op

                measurement_result, prob = qapi.measure(self.qprocessor.peek([0])[0], observable=measurement_observable)

                tot += 1
                if measurement_result == 0:
                    correct += 1
                else:
                    wrong += 1
            
            self.results.append((m, tot, correct, wrong))


        

    




res_dict = {}

for m in list(range(1, 10, 1)) + list(range(10, 101, 10)) + list(range(100, 1001, 100)):
    print(m)
    for _ in range(20 if m >= 1000 else 100 if m >= 100 else 1000):
        loc_node = Node(name="loc_node")
        rem_node = Node(name="rem_node")

        distance = 100 #km
        depolar_rate = 1 # Depolarization rate in Hz
        fibre_delay_model = FibreDelayModel()
        fibre_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
        channel_1 = QuantumChannel("ch1", length=distance, models={"delay_model": fibre_delay_model, "quantum_noise_model": fibre_noise_model})
        channel_2 = QuantumChannel("ch2", length=distance, models={"delay_model": fibre_delay_model, "quantum_noise_model": fibre_noise_model})


        cmd_channel = ClassicalChannel("cl1", length=distance, models={"delay_model": fibre_delay_model})
        ack_channel = ClassicalChannel("cl2", length=distance, models={"delay_model": fibre_delay_model})


        q_connection = DirectConnection("q_conn", channel_AtoB=channel_1, channel_BtoA=channel_2)
        loc_node.connect_to(remote_node=rem_node, connection=q_connection, local_port_name="qubitIO", remote_port_name="qubitIO")

        c_connection = DirectConnection("c_conn", channel_AtoB=cmd_channel, channel_BtoA=ack_channel)
        loc_node.connect_to(remote_node=rem_node, connection=c_connection, local_port_name="classicalIO", remote_port_name="classicalIO")

        loc_pr = LocalBenchmarkingProtocol(loc_node, [(m, 1)])
        rem_pr = RemoteBenchmarkingProtocol(rem_node)


        rem_pr.start()
        loc_pr.start()
        run_stats = ns.sim_run()

        for tup in loc_pr.results:
            cur = res_dict.get(tup[0], (0,0,0))
            res_dict[tup[0]] = (cur[0]+tup[1], cur[1]+tup[2], cur[2]+tup[3])

x_arr = list(res_dict.keys())
y_arr = [res_dict[x][1] / res_dict[x][0] for x in x_arr]
print(x_arr)
print(y_arr)
plt.plot(x_arr, y_arr)
plt.show()
