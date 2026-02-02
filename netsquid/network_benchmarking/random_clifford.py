import netsquid as ns
from netsquid.nodes import Node
from netsquid.components.models import DelayModel
from netsquid.components import QuantumChannel
from netsquid.nodes import DirectConnection
from netsquid.protocols import NodeProtocol
import netsquid.qubits.operators as op
import numpy as np


clifford_rng = None

# identity
I = op.Operator("I", [[1, 0],
                      [0, 1]])

Sd = op.Operator("Sd", [[1, 0],
                        [0, -1j]])

# Operator composition uses matrix multiplication: A * B
# (first apply B, then A)

clifford_gates = [
    I,
    op.X,
    op.Y,
    op.Z,

    op.H,
    op.H * op.X,
    op.H * op.Y,
    op.H * op.Z,

    op.S,
    op.S * op.X,
    op.S * op.Y,
    op.S * op.Z,

    Sd,
    Sd * op.X,
    Sd * op.Y,
    Sd * op.Z,

    op.H * op.S,
    op.H * op.S * op.X,
    op.H * op.S * op.Y,
    op.H * op.S * op.Z,

    op.H * Sd,
    op.H * Sd * op.X,
    op.H * Sd * op.Y,
    op.H * Sd * op.Z
]

def get_clifford_gate(index):
    assert(index >= 0)
    assert(index < len(clifford_gates))
    return clifford_gates[index]

def random_clifford_set_seed(s):
    global clifford_rng
    clifford_rng = np.random.default_rng(s)

def get_random_clifford():
    if(clifford_rng == None):
        random_clifford_set_seed(None)
    return get_clifford_gate(clifford_rng.integers(0, len(clifford_gates)))



if __name__ == "__main__":
    print("Ordered Clifford gates:")
    for i in range(len(clifford_gates)):
        print(f"{i}: {get_clifford_gate(i)}")


    print("--------------------------------------------")
    print("Random Clifford gates:")
    random_clifford_set_seed(None)
    for i in range(10):
        gate = get_random_clifford()
        print(f"{i+1}: {gate}")



__all__ = [
    "get_clifford_gate",
    "random_clifford_set_seed",
    "get_random_clifford"
]