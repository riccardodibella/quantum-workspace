import matplotlib.pyplot as plt

def calc_epsilon(V, lambda_nm, p):
    return 3.912 / V * (lambda_nm / 550.0)**(-p)

def p_kim(V):
    if V > 50:
        return 1.6
    if V > 6:
        return 1.3
    if V > 1:
        return 0.16 * V + 0.34
    if V > 0.5:
        return V - 0.5
    return 0

def p_kruse(V):
    if V > 50:
        return 1.6
    if V > 6:
        return 1.3
    return 0.585 * (V ** (1/3))

def epsilon_kim(V, lambda_nm):
    return calc_epsilon(V, lambda_nm, p_kim(V))

def epsilon_kruse(V, lambda_nm):
    return calc_epsilon(V, lambda_nm, p_kruse(V))

lambda_1 = 850
lambda_2 = 1650
V_vals = [0.1*i for i in range(1, 600)]

kim_eps_1 = [epsilon_kim(V, lambda_1) for V in V_vals]
kruse_eps_1 = [epsilon_kruse(V, lambda_1) for V in V_vals]
kim_eps_2 = [epsilon_kim(V, lambda_2) for V in V_vals]
kruse_eps_2 = [epsilon_kruse(V, lambda_2) for V in V_vals]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

ax1.loglog(V_vals, kim_eps_1,   label="Kim")
ax1.loglog(V_vals, kruse_eps_1, label="Kruse")
ax1.set_title(f"λ = {lambda_1} nm")
ax1.set_xlabel("V (km)")
ax1.set_ylabel("ε (km⁻¹)")
ax1.legend()

ax2.loglog(V_vals, kim_eps_2,   label="Kim")
ax2.loglog(V_vals, kruse_eps_2, label="Kruse")
ax2.set_title(f"λ = {lambda_2} nm")
ax2.set_xlabel("V (km)")
ax2.legend()

plt.tight_layout()
plt.show()