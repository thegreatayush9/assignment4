import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Physical parameters
# --------------------------------------------------
L = 1.0                 # slab length (m)
alpha = 1.0             # thermal diffusivity (m^2/s)

T_initial = 350.0       # initial temperature (K)
T_left = 300.0          # left boundary temperature (K)
T_right = 400.0         # right boundary temperature (K)

# Times at which solution is required (s)
output_times = [1, 5, 10, 50, 100]

# --------------------------------------------------
# Spatial discretization
# --------------------------------------------------
N = 51                          # number of spatial nodes
dx = L / (N - 1)
x = np.linspace(0, L, N)

# --------------------------------------------------
# Time step from stability condition (Explicit method)
# Fo <= 0.5
# --------------------------------------------------
Fo = 0.4                        # safe Fourier number
dt = Fo * dx**2 / alpha         # time step

print(f"dx = {dx}")
print(f"dt = {dt}")
print(f"Number of steps to reach 100 s ≈ {int(100/dt)}")

# --------------------------------------------------
# Initial condition
# --------------------------------------------------
T_old = np.ones(N) * T_initial
T_old[0] = T_left
T_old[-1] = T_right

T_new = T_old.copy()

# --------------------------------------------------
# Storage setup (STEP-BASED — KEY FIX)
# --------------------------------------------------
stored_results = {}

# Convert required times into step numbers (rounding to nearest step)
target_steps = {t: int(round(t / dt)) for t in output_times}

# --------------------------------------------------
# Explicit finite difference time-marching loop
# --------------------------------------------------
max_step = max(target_steps.values())

for step in range(1, max_step + 1):

    # ----------------------------------------------
    # Explicit update (vectorized, interior nodes)
    # ----------------------------------------------
    T_new[1:-1] = (
        T_old[1:-1]
        + Fo * (T_old[2:] - 2*T_old[1:-1] + T_old[:-2])
    )

    # ----------------------------------------------
    # Boundary conditions
    # ----------------------------------------------
    T_new[0] = T_left
    T_new[-1] = T_right

    # ----------------------------------------------
    # Store results at exact step indices
    # (current_time ≈ step * dt)
    # ----------------------------------------------
    for t_req, step_req in target_steps.items():
        if step == step_req and t_req not in stored_results:
            stored_results[t_req] = T_new.copy()

    # ----------------------------------------------
    # Prepare for next step
    # ----------------------------------------------
    T_old[:] = T_new[:]

    # Stop early if all required times are captured
    if len(stored_results) == len(output_times):
        break

# --------------------------------------------------
# Plot results
# --------------------------------------------------
plt.figure(figsize=(8, 5))

for t in output_times:
    if t in stored_results:
        plt.plot(x, stored_results[t], label=f"t = {t} s")
    else:
        print(f"Warning: no data stored for t = {t} s")

plt.xlabel("Position x (m)")
plt.ylabel("Temperature T (K)")
plt.title(f"Explicit Finite Difference Solution (α = {alpha})")
plt.legend()
plt.grid(True)
plt.show()
