from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
import ipaddress
import copy
from operator import or_

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
    if is_ip_addr_local(ip, ints):
        return None
    route = find_route(ip, routes)
    if route is not None:
        return find_next_hop_interface(route.next_hop, ints)
    return None


def is_ip_addr_local(addr: ipaddress.IPv4Address, ifs: list[InterfaceConfig]) -> bool:
    for i in ifs:
        if addr == i.conf.ip:
            return True
    return False

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
    ttl: int = 64
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

def is_packet_destination_local(ip: IPv4Packet, ifs: list[InterfaceConfig]) -> bool:
    return is_ip_addr_local(ip.dst, ifs)

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


class PassiveRouterProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)
        if not isinstance(self.node, IPNetworkNode):
            raise TypeError("Node is not IP")


    def run(self):
        if not isinstance(self.node, IPNetworkNode):
            raise TypeError("Node is not IP")
        ports = [self.node.ports[name] for name in self.node.interface_names]
        while True:
            combined_input_event = reduce(or_, [self.await_port_input(p) for p in ports])
        
            yield combined_input_event

            for port in ports:
                rx = port.rx_input()
                if rx is None:
                    continue
                for in_pkt in rx.items:
                    assert isinstance(in_pkt, IPv4Packet)

                    if is_packet_destination_local(in_pkt, self.node.if_configs):
                        print(f"Router {self.node.name} local dst {in_pkt.dst}")
                        if isinstance(in_pkt.payload, ICMPPacket):
                            icmp = in_pkt.payload
                            if icmp.type == ICMPType.ECHO_MSG:
                                out_pkt = IPv4Packet(in_pkt.dst, in_pkt.src, IPProto.ICMP, make_ICMP_Echo_Reply(in_pkt.payload))
                                next_hop_int = find_forward_interface(out_pkt.dst, self.node.routing_table, self.node.if_configs)
                                if next_hop_int is not None:
                                    self.node.ports[next_hop_int.name].tx_output(out_pkt)
                    else:
                        next_hop_int = find_forward_interface(in_pkt.dst, self.node.routing_table, self.node.if_configs)
                        if next_hop_int is not None:
                            in_pkt.ttl -= 1
                            if in_pkt.ttl > 0:
                                print(f"Router {self.node.name} fwd to interface {next_hop_int.name}")
                                self.node.ports[next_hop_int.name].tx_output(in_pkt)
                            else:
                                print(f"Packet dropped at node {self.node.name} (TTL expired)")
                        else:
                            print(f"Packet dropped at node {self.node.name} (Routing failed)")


class SendPingProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)
        if not isinstance(self.node, IPNetworkNode):
            raise TypeError("Node is not IP")

    def run(self):
        if not isinstance(self.node, IPNetworkNode):
            raise TypeError("Node is not IP")
        start_time = ns.sim_time()
        print(f"[{start_time:.2f} ns] {self.node.name} sending CMD")
        
        tx_rx_port = self.node.ports[self.node.interface_names[0]]
        icmp = make_ICMP_Echo_Request()
        ip = IPv4Packet(ipaddress.IPv4Address("192.168.1.1"), ipaddress.IPv4Address("192.168.1.6"), IPProto.ICMP, icmp, icmp.total_length)
        tx_rx_port.tx_output(ip)

        yield self.await_port_input(tx_rx_port)
        end_time = ns.sim_time()
        print(f"[{end_time:.2f} ns] {self.node.name} received ACK. Round trip: {end_time - start_time:.2f} ns")

def connect_nodes(node_a, node_b, port_a, port_b, distance: float):
    """Utility to bridge two nodes with a standard classical connection."""
    delay_model = FibreDelayModel()
    
    # Create channels
    c_abc = ClassicalChannel(f"ch_{node_a.name}_{node_b.name}", length=distance, models={"delay_model": delay_model})
    c_bac = ClassicalChannel(f"ch_{node_b.name}_{node_a.name}", length=distance, models={"delay_model": delay_model})
    
    # Create connection
    conn = DirectConnection(f"conn_{node_a.name}_{node_b.name}", channel_AtoB=c_abc, channel_BtoA=c_bac)
    
    # Connect
    node_a.connect_to(remote_node=node_b, connection=conn, local_port_name=port_a, remote_port_name=port_b)

if __name__ == "__main__":
    node_a_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.1.1/30"))]
    node_a_routing_table = [RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.1.2"))]

    node_int_ifs = [
        InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.1.2/30")), 
        InterfaceConfig("eht1", ipaddress.IPv4Interface("192.168.1.5/30"))
        ]
    node_int_routing_table = [
        RoutingTableEntry(ipaddress.IPv4Network("192.168.1.0/30"), ipaddress.IPv4Address("192.168.1.1")),
        RoutingTableEntry(ipaddress.IPv4Network("192.168.1.4/30"), ipaddress.IPv4Address("192.168.1.6"))
        ]

    node_b_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.1.6/30"))]
    node_b_routing_table = [RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.1.5"))]

    req_node = IPNetworkNode("req", node_a_ifs, node_a_routing_table)
    int_node = IPNetworkNode("int", node_int_ifs, node_int_routing_table)
    rep_node = IPNetworkNode("rep", node_b_ifs, node_b_routing_table)

    connect_nodes(req_node, int_node, req_node.interface_names[0], int_node.interface_names[0], 1)
    connect_nodes(int_node, rep_node, int_node.interface_names[1], rep_node.interface_names[0], 1)

    loc_pr = SendPingProtocol(req_node)
    int_pr = PassiveRouterProtocol(int_node)
    rem_pr = PassiveRouterProtocol(rep_node)

    rem_pr.start()
    int_pr.start()
    loc_pr.start()
    run_stats = ns.sim_run()

    print(run_stats)
