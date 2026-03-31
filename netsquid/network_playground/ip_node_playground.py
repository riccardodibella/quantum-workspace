from dataclasses import dataclass
from enum import IntEnum
import ipaddress
import copy

@dataclass
class InterfaceConfig:
    name: str
    conf: ipaddress.IPv4Interface

@dataclass
class RoutingTableEntry:
    net: ipaddress.IPv4Network
    next_hop: ipaddress.IPv4Address

def sort_routing_table(routing_table: list[RoutingTableEntry]) -> list[RoutingTableEntry]:
    # Sort based on the size of the network netmask (most restrictive netmask first)
    return sorted(routing_table, key=lambda x: x.net.prefixlen, reverse=True)

def find_route(ip: ipaddress.IPv4Address, routing_table: list[RoutingTableEntry]) -> RoutingTableEntry | None:
    routing_table = sort_routing_table(routing_table)
    for r in routing_table:
        if ip in r.net:
            return r
    return None

def find_next_hop_interface(next_hop_ip: ipaddress.IPv4Address, ifs: list[InterfaceConfig]) -> InterfaceConfig | None:
    for i in ifs:
        if next_hop_ip in i.conf.network:
            return i
    return None

def find_forward_interface(ip: ipaddress.IPv4Address, routes: list[RoutingTableEntry], ints: list[InterfaceConfig]) -> InterfaceConfig | None:
    # TODO we should also check if the destination is a local address (the address of one of the interfaces), and in that case return None 
    route = find_route(ip, routes)
    if route is not None:
        return find_next_hop_interface(route.next_hop, ints)
    return None



class IPProto(IntEnum):
    ICMP = 1        # Internet Control Message Protocol
    TCP = 6         # Transmission Control Protocol
    UDP = 17        # User Datagram Protocol

@dataclass
class IPv4Packet:
    src: ipaddress.IPv4Address
    dst: ipaddress.IPv4Address
    payload_proto: IPProto
    payload: any = None
    payload_length: int = 0
    ttl: int = 255
    # Identification, Flags, Fragment Offset not used because IP fragmentation is ignored
    # Type of Service is ignored for now
    # Options are ignored for now

    @property
    def total_length(self) -> int:
        return 20 + self.payload_length # Options not supported

class ICMPType(IntEnum):
    ECHO_MSG = 8    # ICMP Echo Request (Code 0)
    ECHO_REP = 0    # ICMP Echo Reply (Code 0)


@dataclass
class ICMPPacket:
    type: ICMPType
    code: int = 0
    payload: dict | None = None
    total_length: int = 8

def make_ICMP_Echo_Request(identifier: int = 0, sequence: int = 0, data: bytes | None = None) -> ICMPPacket:
    if data is None:
        data = b = b'\x00' * 56
    return ICMPPacket(type=ICMPType.ECHO_MSG, code=0, payload={"identifier": identifier, "sequence": sequence, "data": data}, total_length = 8 + len(data))

def make_ICMP_Echo_Reply(echo_request: ICMPPacket) -> ICMPPacket:
    reply = copy.deepcopy(echo_request)
    reply.type = ICMPType.ECHO_REP
    return reply



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

class IPNetworkNode(Node):
    def __init__(self, name, interface_configs: list[InterfaceConfig], routing_table: list[RoutingTableEntry]):
        port_names = []
        for i in interface_configs:
            port_names += [i.name]
        super().__init__(name, port_names=port_names)
        self.if_configs = interface_configs
        self.routing_table = routing_table
    
    @property
    def interface_names(self) -> list[str]:
        to_return = []
        for i in self.if_configs:
            to_return += [i.name]
        return to_return


class AckPingProtocol(NodeProtocol):
    def __init__(self, node):
        if not isinstance(node, IPNetworkNode):
            raise TypeError("Node is not IP")
        super().__init__(node)


    def run(self):
        tx_rx_port = self.node.ports[self.node.interface_names[0]]
        while True:
            yield self.await_port_input(tx_rx_port)
            # Get current time in nanoseconds
            arrival_time = ns.sim_time() 
            in_pkt = tx_rx_port.rx_input().items[0]
            print(f"[{arrival_time:.2f} ns] {self.node.name} received something")

            assert isinstance(in_pkt, IPv4Packet)
            assert isinstance(in_pkt.payload, ICMPPacket)
            out_pkt = IPv4Packet(in_pkt.dst, in_pkt.src, IPProto.ICMP, make_ICMP_Echo_Reply(in_pkt.payload))
            tx_rx_port.tx_output(out_pkt)


class SendPingProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)
        if not isinstance(self.node, IPNetworkNode):
            raise TypeError("Node is not IP")
        print(node.if_configs)
        print(node.routing_table)

    def run(self):
        if not isinstance(self.node, IPNetworkNode):
            raise TypeError("Node is not IP")
        start_time = ns.sim_time()
        print(f"[{start_time:.2f} ns] {self.node.name} sending CMD")
        
        tx_rx_port = self.node.ports[self.node.interface_names[0]]
        icmp = make_ICMP_Echo_Request()
        ip = IPv4Packet(ipaddress.IPv4Interface("192.168.1.1"), ipaddress.IPv4Interface("192.168.1.2"), IPProto.ICMP, icmp, icmp.total_length)
        tx_rx_port.tx_output(ip)

        yield self.await_port_input(tx_rx_port)
        end_time = ns.sim_time()
        print(f"[{end_time:.2f} ns] {self.node.name} received ACK. Round trip: {end_time - start_time:.2f} ns")


if __name__ == "__main__":
    node_a_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.1.1/30"))]
    node_a_routing_table = [RoutingTableEntry(ipaddress.IPv4Network("192.168.1.0/30"), ipaddress.IPv4Address("192.168.1.2"))]

    node_b_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.1.2/30"))]
    node_b_routing_table = [RoutingTableEntry(ipaddress.IPv4Network("192.168.1.0/30"), ipaddress.IPv4Address("192.168.1.1"))]

    print(find_forward_interface(ipaddress.IPv4Address("192.168.1.2"), node_b_routing_table, node_b_ifs))

    req_node = IPNetworkNode("req", node_a_ifs, node_a_routing_table)
    rep_node = IPNetworkNode("rep", node_b_ifs, node_b_routing_table)

    distance = 1 #km
    fibre_delay_model = FibreDelayModel()

    cmd_channel = ClassicalChannel("cl1", length=distance, models={"delay_model": fibre_delay_model})
    ack_channel = ClassicalChannel("cl2", length=distance, models={"delay_model": fibre_delay_model})

    c_connection = DirectConnection("c_conn", channel_AtoB=cmd_channel, channel_BtoA=ack_channel)
    req_node.connect_to(remote_node=rep_node, connection=c_connection, local_port_name=req_node.interface_names[0], remote_port_name=req_node.interface_names[0])

    loc_pr = SendPingProtocol(req_node)
    rem_pr = AckPingProtocol(rep_node)

    rem_pr.start()
    loc_pr.start()
    run_stats = ns.sim_run()

    print(run_stats)
