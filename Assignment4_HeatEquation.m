% Assignment 4 - Heat Equation Solver
% Solves 1D unsteady-state heat conduction equation
% dT/dt = alpha * d^2T/dx^2

clc; clear; close all;

% --- Parameters ---
L = 1.0;            % Length (m)
T_initial = 350.0;  % Initial Temp (K)
T_left = 300.0;     % BC at x=0 (K)
T_right = 400.0;    % BC at x=L (K)
dx = 0.01;          % Spatial step (m)
alpha_values = [0.0001, 0.001, 0.01]; % Thermal Diffusivity
check_times = [1, 5, 10, 50, 100];    % Times to plot (s)

x = 0:dx:L;
Nx = length(x);

for i = 1:length(alpha_values)
    alpha = alpha_values(i);
    fprintf('Processing Alpha = %.4f\n', alpha);
    
    % Determine Time Step
    % Explicit stability: dt <= 0.5 * dx^2 / alpha
    dt_stable = 0.5 * dx^2 / alpha;
    dt = dt_stable * 0.9; % Safety factor
    
    % For Implicit/CN we could use larger dt, but let's keep it same for comparison
    % unless it's too small (like for alpha=100, but here max alpha=0.01)
    % For alpha=0.01, dt_stable = 0.5*1e-4/0.01 = 0.005. Reasonable.
    
    T_max = max(check_times);
    
    % --- Run Solvers ---
    
    % Explicit
    tic;
    [T_exp, t_exp_grid] = solveExplicit(alpha, L, dx, dt, T_max, T_initial, T_left, T_right, check_times);
    t_explicit = toc;
    fprintf('  Explicit done in %.4f s\n', t_explicit);
    
    % Implicit
    tic;
    [T_imp, ~] = solveImplicit(alpha, L, dx, dt, T_max, T_initial, T_left, T_right, check_times);
    t_implicit = toc;
    fprintf('  Implicit done in %.4f s\n', t_implicit);
    
    % Crank-Nicolson
    tic;
    [T_cn, ~] = solveCN(alpha, L, dx, dt, T_max, T_initial, T_left, T_right, check_times);
    t_cn = toc;
    fprintf('  Crank-Nicolson done in %.4f s\n', t_cn);
    
    % --- Plotting ---
    figure('Name', sprintf('Alpha = %.4f', alpha), 'Position', [100, 100, 1500, 300]);
    
    for j = 1:length(check_times)
        t_target = check_times(j);
        
        subplot(1, 5, j);
        hold on;
        
        % Analytical Solution
        T_ana = analyticalSol(x, t_target, alpha, L, T_left, T_right, T_initial);
        
        % Plot Lines
        plot(x, T_exp(j, :), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Explicit');
        plot(x, T_imp(j, :), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Implicit');
        plot(x, T_cn(j, :), 'g:', 'LineWidth', 2, 'DisplayName', 'Crank-Nicolson');
        plot(x, T_ana, 'ks', 'MarkerSize', 3, 'DisplayName', 'Analytical');
        
        title(sprintf('Time = %d s', t_target));
        xlabel('Position (m)');
        if j == 1
            ylabel('Temperature (K)');
            legend('Location', 'best');
        end
        grid on;
        box on;
        hold off;
    end
    
    sgtitle(sprintf('1D Heat Equation - Alpha = %.4f m^2/s', alpha));
end

% --- Local Functions ---

function [T_out, t_grid] = solveExplicit(alpha, L, dx, dt, T_max, T_init, T_L, T_R, times)
    x = 0:dx:L;
    Nx = length(x);
    Nt = ceil(T_max / dt);
    r = alpha * dt / dx^2;
    
    T = ones(1, Nx) * T_init;
    T(1) = T_L;
    T(end) = T_R;
    
    T_out = zeros(length(times), Nx);
    current_t_idx = 1;
    
    % Loop
    time = 0;
    while time < T_max
        if current_t_idx <= length(times) && time >= times(current_t_idx)
             T_out(current_t_idx, :) = T;
             current_t_idx = current_t_idx + 1;
        end
        
        T_new = T;
        T_new(2:end-1) = r*T(1:end-2) + (1-2*r)*T(2:end-1) + r*T(3:end);
        
        T = T_new;
        time = time + dt;
    end
    
    % Capture last time if needed (simple approximation)
    if current_t_idx <= length(times)
        T_out(current_t_idx:end, :) = repmat(T, length(times)-current_t_idx+1, 1);
    end
    t_grid = times;
end

function [T_out, t_grid] = solveImplicit(alpha, L, dx, dt, T_max, T_init, T_L, T_R, times)
    x = 0:dx:L;
    Nx = length(x);
    Nt = ceil(T_max / dt);
    r = alpha * dt / dx^2;
    
    T = ones(Nx, 1) * T_init;
    T(1) = T_L;
    T(end) = T_R;
    
    % Matrix A setup
    % -r*T(i-1) + (1+2r)*T(i) - r*T(i+1) = T_old(i)
    M = Nx - 2;
    main_diag = (1 + 2*r) * ones(M, 1);
    off_diag = -r * ones(M, 1);
    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], M, M);
    
    T_out = zeros(length(times), Nx);
    current_t_idx = 1;
    
    time = 0;
    while time < T_max
        if current_t_idx <= length(times) && time >= times(current_t_idx)
             T_out(current_t_idx, :) = T';
             current_t_idx = current_t_idx + 1;
        end
        
        b = T(2:end-1);
        b(1) = b(1) + r*T_L;
        b(end) = b(end) + r*T_R;
        
        T_internal = A \ b;
        T(2:end-1) = T_internal;
        
        time = time + dt;
    end
     if current_t_idx <= length(times)
        T_out(current_t_idx:end, :) = repmat(T', length(times)-current_t_idx+1, 1);
    end
    t_grid = times;
end

function [T_out, t_grid] = solveCN(alpha, L, dx, dt, T_max, T_init, T_L, T_R, times)
    x = 0:dx:L;
    Nx = length(x);
    Nt = ceil(T_max / dt);
    r = alpha * dt / dx^2;
    
    T = ones(Nx, 1) * T_init;
    T(1) = T_L;
    T(end) = T_R;
    
    M = Nx - 2;
    % LHS Matrix: -r/2, 1+r, -r/2
    main_diag = (1 + r) * ones(M, 1);
    off_diag = -0.5 * r * ones(M, 1);
    A = spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], M, M);
    
    T_out = zeros(length(times), Nx);
    current_t_idx = 1;
    
    time = 0;
    while time < T_max
        if current_t_idx <= length(times) && time >= times(current_t_idx)
             T_out(current_t_idx, :) = T';
             current_t_idx = current_t_idx + 1;
        end
        
        % RHS Calculation
        % b = r/2*T(i-1) + (1-r)*T(i) + r/2*T(i+1)
        T_vec = T; 
        b = 0.5*r*T_vec(1:end-2) + (1-r)*T_vec(2:end-1) + 0.5*r*T_vec(3:end);
        
        % Boundary contributions
        % LHS term -r/2*T0 moves to RHS as +r/2*T0
        b(1) = b(1) + 0.5*r*T_L;
        b(end) = b(end) + 0.5*r*T_R;
        
        T_internal = A \ b;
        T(2:end-1) = T_internal;
        
        time = time + dt;
    end
    if current_t_idx <= length(times)
        T_out(current_t_idx:end, :) = repmat(T', length(times)-current_t_idx+1, 1);
    end
    t_grid = times;
end

function T = analyticalSol(x_grid, t, alpha, L, T_L, T_R, T_init)
    T_ss = T_L + (T_R - T_L) * x_grid / L;
    
    v = zeros(size(x_grid));
    for n = 1:100
        kn = n * pi / L;
        
        % Bn calculation (same logic as Python)
        % Bn = (2/L) * integral
        % Integral terms derived:
        term1 = (-50.0 / kn) * (cos(n * pi) - 1.0);
        term2 = -100.0 * ( (-L / kn * cos(n * pi)) );
        Bn = (2.0 / L) * (term1 + term2);
        
        v = v + Bn * sin(kn * x_grid) * exp(-alpha * kn^2 * t);
    end
    T = T_ss + v;
end
