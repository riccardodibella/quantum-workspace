import netsquid as ns
qubits = ns.qubits.create_qubits(1)
print(qubits)
qubit = qubits[0]
# To check the state is |0> we check its density matrix using reduced_dm():
print("zero (initial):")
print(ns.qubits.reduced_dm(qubit))


ns.qubits.operate(qubit, ns.X)
print("one:")
print(ns.qubits.reduced_dm(qubit))


measurement_result, prob = ns.qubits.measure(qubit) # Z measurement by default
if measurement_result == 0:
    state = "|0>"
else:
    state = "|1>"
print(f"Measured {state} with probability {prob:.1f}")


print("------")
print("dm before X measurement:")
print(ns.qubits.reduced_dm(qubit))
measurement_result, prob = ns.qubits.measure(qubit, observable=ns.X) # Measurement in X basis
if measurement_result == 0:
    state = "|+>"
else:
    state = "|->"
print(f"Measured {state} with probability {prob:.1f}")
print("dm after X measurement:")
print(ns.qubits.reduced_dm(qubit))
