# pyright: basic

from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
import ipaddress
import copy
from operator import or_
from typing import Any

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
    payload: Any | None = None
    payload_length: int = 0
    ttl: int = 64
    # Identification, Flags, Fragment Offset not used because IP fragmentation is ignored
    # Type of Service is ignored for now
    # Options are ignored for now

    @property
    def total_length(self) -> int:
        return 20 + self.payload_length # Options not supported

def is_packet_destination_local(ip: IPv4Packet, ifs: list[InterfaceConfig]) -> bool:
    return is_ip_addr_local(ip.dst, ifs)

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

UDP_HEADER_LENGTH = 8

@dataclass
class UDPPacket:
    source_port: int
    dst_port: int
    payload: Any | None = None
    payload_length: int = 0

class UDPPort(IntEnum):
    QOTD = 17       # Quote Of The Day

import numpy as np
import matplotlib.pyplot as plt
import netsquid as ns
from netsquid.nodes import Node
from netsquid.components.models import FibreDelayModel
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components import QuantumChannel, ClassicalChannel, QuantumErrorModel
from netsquid.nodes import DirectConnection
from netsquid.protocols import NodeProtocol
from netsquid.qubits import qubitapi as qapi

def qport(classical_port_name: str):
    return f"{classical_port_name}_q"

class IPNetworkNode(Node):
    def __init__(self, name, interface_configs: list[InterfaceConfig], routing_table: list[RoutingTableEntry]):
        port_names = []
        for i in interface_configs:
            port_names += [i.name, qport(i.name)]
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
                        if in_pkt.payload_proto == IPProto.ICMP:
                            assert isinstance(in_pkt.payload, ICMPPacket)
                            icmp = in_pkt.payload
                            if icmp.type == ICMPType.ECHO_MSG:
                                out_pkt = IPv4Packet(in_pkt.dst, in_pkt.src, IPProto.ICMP, make_ICMP_Echo_Reply(in_pkt.payload))
                                next_hop_int = find_forward_interface(out_pkt.dst, self.node.routing_table, self.node.if_configs)
                                if next_hop_int is not None:
                                    self.node.ports[next_hop_int.name].tx_output(out_pkt)
                        elif in_pkt.payload_proto == IPProto.UDP:
                            assert isinstance(in_pkt.payload, UDPPacket)
                            udp = in_pkt.payload
                            if udp.dst_port == UDPPort.QOTD: # Maybe this shouldn't be a "Passive Router" functionality but a server service
                                quote = "Variables won't; constants aren't. (Osborn's Law)"
                                quote_length = len(quote) + 1
                                out_pkt = IPv4Packet(in_pkt.dst, in_pkt.src, IPProto.UDP, payload = UDPPacket(source_port=udp.dst_port, dst_port=udp.source_port, payload=quote, payload_length=quote_length), payload_length = UDP_HEADER_LENGTH+quote_length)
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
        ip = IPv4Packet(ipaddress.IPv4Address("192.168.0.1"), ipaddress.IPv4Address("192.168.0.129"), IPProto.ICMP, icmp, icmp.total_length)
        tx_rx_port.tx_output(ip)

        yield self.await_port_input(tx_rx_port)
        end_time = ns.sim_time()
        print(f"[{end_time:.2f} ns] {self.node.name} received ACK. Round trip: {end_time - start_time:.2f} ns")

class AskQOTDProtocol(NodeProtocol):
    def __init__(self, node):
        super().__init__(node)
        if not isinstance(self.node, IPNetworkNode):
            raise TypeError("Node is not IP")

    def run(self):
        if not isinstance(self.node, IPNetworkNode):
            raise TypeError("Node is not IP")
        start_time = ns.sim_time()
        print(f"[{start_time:.2f} ns] {self.node.name} sending QOTD req")
        
        tx_rx_port = self.node.ports[self.node.interface_names[0]]
        udp = UDPPacket(1234, UDPPort.QOTD, None, 0)
        ip = IPv4Packet(ipaddress.IPv4Address("192.168.0.17"), ipaddress.IPv4Address("192.168.0.133"), IPProto.UDP, udp, UDP_HEADER_LENGTH+udp.payload_length)
        tx_rx_port.tx_output(ip)

        yield self.await_port_input(tx_rx_port)
        in_pkt = tx_rx_port.rx_input().items[0]
        assert isinstance(in_pkt, IPv4Packet)
        assert isinstance(in_pkt.payload, UDPPacket)
        
        end_time = ns.sim_time()
        print(f"[{end_time:.2f} ns] {self.node.name} received QOTD \"{in_pkt.payload.payload}\". Round trip: {end_time - start_time:.2f} ns")

def connect_nodes(node_a: IPNetworkNode, node_b: IPNetworkNode, port_index_a, port_index_b, distance: int, quantum_noise_model: None | QuantumErrorModel = None):
    """Utility to bridge two nodes with a standard classical connection."""
    delay_model = FibreDelayModel()
    
    c_ab = ClassicalChannel(f"c_ch_{node_a.name}_{node_b.name}", length=distance, models={"delay_model": delay_model})
    c_ba = ClassicalChannel(f"c_ch_{node_b.name}_{node_a.name}", length=distance, models={"delay_model": delay_model})
    conn = DirectConnection(f"c_conn_{node_a.name}_{node_b.name}", channel_AtoB=c_ab, channel_BtoA=c_ba)
    node_a.connect_to(remote_node=node_b, connection=conn, local_port_name=node_a.interface_names[port_index_a], remote_port_name=node_b.interface_names[port_index_b])

    q_ab = QuantumChannel(f"q_ch_{node_a.name}_{node_b.name}", length=distance, models={"delay_model": delay_model, "quantum_noise_model": quantum_noise_model})
    q_ba = QuantumChannel(f"q_ch_{node_b.name}_{node_a.name}", length=distance, models={"delay_model": delay_model, "quantum_noise_model": quantum_noise_model})
    q_conn = DirectConnection(f"q_conn_{node_a.name}_{node_b.name}", channel_AtoB=q_ab, channel_BtoA=q_ba)
    node_a.connect_to(remote_node=node_b, connection=q_conn, local_port_name=qport(node_a.interface_names[port_index_a]), remote_port_name=qport(node_b.interface_names[port_index_b]))

if __name__ == "__main__":
    c1_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.0.1/30"))]
    c1_rt = [RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.0.2"))]

    c2_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.0.17/30"))]
    c2_rt = [RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.0.18"))]

    s1_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.0.129/30"))]
    s1_rt = [RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.0.130"))]

    s2_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.0.133/30"))]
    s2_rt = [RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.0.134"))]


    r1_ifs = [
        InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.0.2/30")), 
        InterfaceConfig("eht1", ipaddress.IPv4Interface("192.168.0.5/30"))
        ]
    r1_rt = [
        RoutingTableEntry(ipaddress.IPv4Network("192.168.0.0/30"), ipaddress.IPv4Address("192.168.0.1")),
        RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.0.6"))
        ]

    r2_ifs = [
        InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.0.18/30")), 
        InterfaceConfig("eht1", ipaddress.IPv4Interface("192.168.0.21/30"))
        ]
    r2_rt = [
        RoutingTableEntry(ipaddress.IPv4Network("192.168.0.16/30"), ipaddress.IPv4Address("192.168.0.17")),
        RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.0.22"))
        ]
    
    r3_ifs = [
        InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.0.6/30")), 
        InterfaceConfig("eht1", ipaddress.IPv4Interface("192.168.0.22/30")),
        InterfaceConfig("eht2", ipaddress.IPv4Interface("192.168.0.65/30"))
        ]
    r3_rt = [
        RoutingTableEntry(ipaddress.IPv4Network("192.168.0.0/28"), ipaddress.IPv4Address("192.168.0.5")),
        RoutingTableEntry(ipaddress.IPv4Network("192.168.0.16/28"), ipaddress.IPv4Address("192.168.0.21")),
        RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.0.66"))
        ]
    
    r4_ifs = [
        InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.0.130/30")), 
        InterfaceConfig("eht1", ipaddress.IPv4Interface("192.168.0.134/30")),
        InterfaceConfig("eht2", ipaddress.IPv4Interface("192.168.0.66/30"))
        ]
    r4_rt = [
        RoutingTableEntry(ipaddress.IPv4Network("192.168.0.128/30"), ipaddress.IPv4Address("192.168.0.129")),
        RoutingTableEntry(ipaddress.IPv4Network("192.168.0.132/30"), ipaddress.IPv4Address("192.168.0.133")),
        RoutingTableEntry(ipaddress.IPv4Network("0.0.0.0/0"), ipaddress.IPv4Address("192.168.0.65"))
        ]

    c1_node = IPNetworkNode("c1", c1_ifs, c1_rt)
    c2_node = IPNetworkNode("c2", c2_ifs, c2_rt)
    s1_node = IPNetworkNode("s1", s1_ifs, s1_rt)
    s2_node = IPNetworkNode("s2", s2_ifs, s2_rt)
    r1_node = IPNetworkNode("r1", r1_ifs, r1_rt)
    r2_node = IPNetworkNode("r2", r2_ifs, r2_rt)
    r3_node = IPNetworkNode("r3", r3_ifs, r3_rt)
    r4_node = IPNetworkNode("r4", r4_ifs, r4_rt)

    distance = 5 # km
    loss_model = DepolarNoiseModel(depolar_rate=0.01)

    connect_nodes(c1_node, r1_node, 0, 0, distance, loss_model)
    connect_nodes(r1_node, r3_node, 1, 0, distance)
    connect_nodes(c2_node, r2_node, 0, 0, distance+1)
    connect_nodes(r2_node, r3_node, 1, 1, distance)
    connect_nodes(s1_node, r4_node, 0, 0, distance)
    connect_nodes(s2_node, r4_node, 0, 1, distance)
    connect_nodes(r3_node, r4_node, 2, 2, distance)

    c1_pr = SendPingProtocol(c1_node)
    c2_pr = AskQOTDProtocol(c2_node)
    s1_pr = PassiveRouterProtocol(s1_node)
    s2_pr = PassiveRouterProtocol(s2_node)
    r1_pr = PassiveRouterProtocol(r1_node)
    r2_pr = PassiveRouterProtocol(r2_node)
    r3_pr = PassiveRouterProtocol(r3_node)
    r4_pr = PassiveRouterProtocol(r4_node)

    c1_pr.start()
    c2_pr.start()
    s1_pr.start()
    s2_pr.start()
    r1_pr.start()
    r2_pr.start()
    r3_pr.start()
    r4_pr.start()

    run_stats = ns.sim_run()

    print(run_stats)