import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Parameters from Assignment [cite: 6, 8, 9]
# --------------------------------------------------
L = 1.0                # m
T_initial = 350.0      # K
T_left = 300.0         # K
T_right = 400.0        # K
alphas = [1, 10, 100]  # m^2/s
output_times = [1, 5, 10, 50, 100] # s

# Spatial setup
N = 51
x = np.linspace(0, L, N)
dx = L / (N - 1)

for alpha in alphas:
    # Stability: dt <= dx^2 / (2*alpha) [cite: 48]
    dt = 0.4 * (dx**2) / alpha 
    Fo = (alpha * dt) / (dx**2)
    
    # Create a figure for each Alpha with subplots for each time
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    fig.suptitle(f'Temperature Distribution for Alpha = {alpha} m²/s', fontsize=16)
    
    T = np.full(N, T_initial)
    T[0], T[-1] = T_left, T_right
    
    current_time = 0
    plot_idx = 0
    
    # Simulation Loop
    while plot_idx < len(output_times):
        T_new = T.copy()
        # Explicit discretization formula [cite: 45]
        T_new[1:-1] = T[1:-1] + Fo * (T[2:] - 2*T[1:-1] + T[:-2])
        T = T_new
        current_time += dt
        
        # Check if we reached a target time
        if current_time >= output_times[plot_idx]:
            ax = axes[plot_idx]
            ax.plot(x, T, color='red', linewidth=2)
            ax.set_title(f"Time = {output_times[plot_idx]}s")
            ax.set_xlabel("x (m)")
            ax.grid(True)
            if plot_idx == 0:
                ax.set_ylabel("Temp (K)")
            plot_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()