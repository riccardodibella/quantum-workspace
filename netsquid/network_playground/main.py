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

class AckPingProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["classicalIO"])
            # Get current time in nanoseconds
            arrival_time = ns.sim_time() 
            message = self.node.ports["classicalIO"].rx_input().items[0]
            print(f"[{arrival_time:.2f} ns] {self.node.name} received {message['type']}")

            cmd = {"type": "ACK"}
            self.node.ports["classicalIO"].tx_output(cmd)


class SendPingProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)

    def run(self):
        start_time = ns.sim_time()
        print(f"[{start_time:.2f} ns] {self.node.name} sending CMD")
        
        cmd = {"type": "CMD"}
        self.node.ports["classicalIO"].tx_output(cmd)

        yield self.await_port_input(self.node.ports["classicalIO"])
        end_time = ns.sim_time()
        print(f"[{end_time:.2f} ns] {self.node.name} received ACK. Round trip: {end_time - start_time:.2f} ns")



loc_node = Node(name="loc_node")
rem_node = Node(name="rem_node")
distance = 1 #km
fibre_delay_model = FibreDelayModel()

cmd_channel = ClassicalChannel("cl1", length=distance, models={"delay_model": fibre_delay_model})
ack_channel = ClassicalChannel("cl2", length=distance, models={"delay_model": fibre_delay_model})

c_connection = DirectConnection("c_conn", channel_AtoB=cmd_channel, channel_BtoA=ack_channel)
loc_node.connect_to(remote_node=rem_node, connection=c_connection, local_port_name="classicalIO", remote_port_name="classicalIO")

loc_pr = SendPingProtocol(loc_node)
rem_pr = AckPingProtocol(rem_node)

rem_pr.start()
loc_pr.start()
run_stats = ns.sim_run()

print(run_stats)
