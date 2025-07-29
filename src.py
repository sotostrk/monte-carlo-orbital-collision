import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")


# Constants

mu = 398600.4418  
collision_threshold_km = 1.0  # 1 km
a_nom = 6771
e_nom = 0.0
i_nom = 51.6
RAAN_nom = 40.0
argp_nom = 0.0
nu_nom = 0.0


def orbital_elements_to_state_vector(a, e, i_deg, RAAN_deg, argp_deg, nu_deg, mu=398600.4418):
    i = np.radians(i_deg)
    RAAN = np.radians(RAAN_deg)
    argp = np.radians(argp_deg)
    nu = np.radians(nu_deg)
    p = a * (1 - e**2)

    r_pqw = np.array([
        p * np.cos(nu) / (1 + e * np.cos(nu)),
        p * np.sin(nu) / (1 + e * np.cos(nu)),
        0])
    v_pqw = np.array([
        -np.sqrt(mu / p) * np.sin(nu),
        np.sqrt(mu / p) * (e + np.cos(nu)),
        0])

    R3_Omega = np.array([
        [np.cos(-RAAN), np.sin(-RAAN), 0],
        [-np.sin(-RAAN), np.cos(-RAAN), 0],
        [0, 0, 1]])
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(-i), np.sin(-i)],
        [0, -np.sin(-i), np.cos(-i)]])
    R3_argp = np.array([
        [np.cos(-argp), np.sin(-argp), 0],
        [-np.sin(-argp), np.cos(-argp), 0],
        [0, 0, 1]])

    R = R3_Omega @ R1_i @ R3_argp
    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    return r_eci, v_eci

# 


def two_body_dynamics(t, y, mu=398600.4418):
    r = y[:3]
    v = y[3:]
    norm_r = np.linalg.norm(r)
    a = -mu * r / norm_r**3
    return np.concatenate((v, a))


# Randomizer second satellite elements

def randomize_satB():
    a = a_nom + np.random.uniform(-5, 5)
    e = e_nom + np.random.uniform(-0.0005, 0.0005)
    i = i_nom + np.random.uniform(-0.05, 0.05)
    RAAN = RAAN_nom + np.random.uniform(-0.05, 0.05)
    argp = argp_nom + np.random.uniform(-0.2, 0.2)
    nu = nu_nom + np.random.uniform(-0.1, 0.1)
    return orbital_elements_to_state_vector(a, e, i, RAAN, argp, nu)


# Monte Carlo Sim

num_trials = 500
collision_count = 0
results = []
all_distances = []
close_r_A = None
close_r_B = None

orbital_period = 2 * np.pi * np.sqrt(a_nom**3 / mu)
t_end = 2 * orbital_period
t_eval = np.linspace(0, t_end, 2000)

r_A0, v_A0 = orbital_elements_to_state_vector(a_nom, e_nom, i_nom, RAAN_nom, argp_nom, nu_nom)
y0_A = np.concatenate((r_A0, v_A0))

for trial in range(num_trials):
    r_B0, v_B0 = randomize_satB()
    y0_B = np.concatenate((r_B0, v_B0))

    sol_A = solve_ivp(two_body_dynamics, [0, t_end], y0_A, t_eval=t_eval)
    sol_B = solve_ivp(two_body_dynamics, [0, t_end], y0_B, t_eval=t_eval)

    r_A = sol_A.y[:3, :].T
    r_B = sol_B.y[:3, :].T
    distance = np.linalg.norm(r_A - r_B, axis=1)
    min_dist_km = np.min(distance)
    all_distances.append(min_dist_km * 1000)  # in meters

    if min_dist_km < collision_threshold_km:
        results.append(1)
        collision_count += 1
        if close_r_A is None:
            close_r_A = r_A
            close_r_B = r_B
    else:
        results.append(0)


# Results 

print(f"Total close approaches: {collision_count} out of {num_trials}")
print(f"Estimated probability of collision: {collision_count / num_trials:.4f}")

#
# Bar Plot 

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["red" if r == 1 else "blue" for r in results]
ax.bar(range(num_trials), results, color=colors)
ax.set_xlabel("Monte Carlo Trial")
ax.set_ylabel("Outcome (0 = safe, 1 = close approach)")
ax.set_title(f"Monte Carlo Collision Check: {collision_count}/{num_trials} close approaches")
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.grid(True)
plt.tight_layout()
plt.show()


# Histogram of Distances

plt.figure(figsize=(10, 5))
plt.hist(all_distances, bins=40, color='purple', edgecolor='black')
plt.axvline(100, color='red', linestyle='--', label='100m threshold')
plt.title("Distribution of Minimum Separation Distances")
plt.xlabel("Minimum Distance [m]")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 3D Plot of Closest Pass

if close_r_A is not None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(close_r_A[:, 0], close_r_A[:, 1], close_r_A[:, 2], label="Satellite A")
    ax.plot(close_r_B[:, 0], close_r_B[:, 1], close_r_B[:, 2], label="Satellite B")
    ax.set_title("Sample Close Encounter Trajectory")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend()
    plt.tight_layout()
    plt.show()