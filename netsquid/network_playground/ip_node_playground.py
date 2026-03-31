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
    payload: any = None
    payload_length: int = 0
    payload_proto: IPProto
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
    payload: dict
    total_length: int

def make_ICMP_Echo_Request(identifier: int = 0, sequence: int = 0, data: bytes | None = None) -> ICMPPacket:
    if data is None:
        data = b = b'\x00' * 56
    return ICMPPacket(type=ICMPType.ECHO_MSG, code=0, payload={"identifier": identifier, "sequence": sequence, "data": data}, total_length = 4 + len(data))

def make_ICMP_Echo_Reply(echo_request: ICMPPacket) -> ICMPPacket:
    reply = copy.deepcopy(echo_request)
    reply.type = ICMPType.ECHO_REP
    return reply

if __name__ == "__main__":
    node_a_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.1.1/30"))]
    node_a_routing_table = [RoutingTableEntry(ipaddress.IPv4Network("192.168.1.0/30"), ipaddress.IPv4Address("192.168.1.2"))]

    node_b_ifs = [InterfaceConfig("eht0", ipaddress.IPv4Interface("192.168.1.2/30"))]
    node_b_routing_table = [RoutingTableEntry(ipaddress.IPv4Network("192.168.1.0/30"), ipaddress.IPv4Address("192.168.1.1"))]

    print(find_forward_interface(ipaddress.IPv4Address("192.168.1.2"), node_b_routing_table, node_b_ifs))
