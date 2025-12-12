import netsquid.qubits.cliffords as cl
import netsquid.qubits.operators as op
import numpy as np

# According to the documentation (Page 3), all 24 local cliffords 
# are available in this pre-defined list.
clifford_gates = cl.local_cliffords


clifford_rng = None

def get_clifford_gate(index):
    """Returns the Clifford gate at the specified index."""
    assert 0 <= index < len(clifford_gates)
    return clifford_gates[index]

def random_clifford_set_seed(s):
    """Sets the seed for the random number generator."""
    global clifford_rng
    clifford_rng = np.random.default_rng(s)

def get_random_clifford():
    """Returns a random Clifford gate from the list."""
    if clifford_rng is None:
        random_clifford_set_seed(None)
    # Use rng.choice to pick a random gate directly from the list
    return clifford_rng.choice(clifford_gates)

def get_operator_from_clifford(clifford_obj):
    """
    Converts a LocalClifford object into a standard NetSquid Operator
    using the .arr attribute.
    """
    # 1. Extract the name (e.g., "Identity", "Hadamard")
    name = clifford_obj.name
    
    # 2. Extract the unitary matrix from the .arr attribute
    matrix_data = clifford_obj.arr
    
    # 3. Create and return the Operator object
    return op.Operator(name, matrix_data)

if __name__ == "__main__":
    print(f"Total defined Clifford gates: {len(clifford_gates)}")
    print("Ordered Clifford gates:")
    for i, gate in enumerate(clifford_gates):
        print(f"{i}: {gate}")

    print("--------------------------------------------")
    print("Random Clifford gates:")
    random_clifford_set_seed(None)
    for i in range(10):
        gate = get_random_clifford()
        print(f"{i+1}: {gate}")

__all__ = [
    "get_clifford_gate",
    "get_operator_from_clifford",
    "random_clifford_set_seed",
    "get_random_clifford"
]