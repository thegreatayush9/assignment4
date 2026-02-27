function HeatConduction_Fixed()
    clc; clear; close all;

    % --- Parameters ---
    L = 1;                  % Length (m)
    T_init = 350;           % Initial Temp (K)
    T_left = 300;           % BC at x=0
    T_right = 400;          % BC at x=L
    alphas = [0.0001, 0.001, 0.01]; % Thermal diffusivities
    target_times = [1, 5, 10, 50, 100]; 
    
    % Grid Generation
    Nx = 50;                
    dx = L / Nx;
    x = linspace(0, L, Nx+1);

    % Loop through each Alpha value
    for k = 1:length(alphas)
        alpha = alphas(k);
        
        % Time step for Manual Method (Stability)
        dt_stable = 0.5 * dx^2 / alpha; 
        dt = min(dt_stable * 0.9, 0.1); % Cap dt at 0.1s for speed
        
        figure('Name', sprintf('Alpha = %.4f', alpha));
        hold on; grid on;
        title(sprintf('Temp vs Position (Alpha = %.4f m^2/s)', alpha));
        xlabel('Position (m)'); ylabel('Temperature (K)');
        ylim([290 410]);
        
        colors = lines(length(target_times));
        
        % Loop through target times
        for j = 1:length(target_times)
            t_final = target_times(j);
            
            % 1. Analytical Solution
            T_ana = solve_analytical(x, t_final, alpha, L, T_left, T_right);
            
            % 2. Crank-Nicholson (Manual Implementation)
            T_cn = solve_crank_nicholson(x, dx, dt, t_final, alpha, T_left, T_right, T_init);
            
            % 3. PDEPE (Built-in Solver) - FIXED LINE BELOW
            % We create a time vector with 20 points to satisfy the "at least 3 entries" rule
            t_span_pde = linspace(0, t_final, 20); 
            
            m = 0; % Slab geometry
            sol_pde = pdepe(m, @(x,t,u,DuDx) pdefun(x,t,u,DuDx,alpha), ...
                           @(x) icfun(x, T_init), ...
                           @(xl,ul,xr,ur,t) bcfun(xl,ul,xr,ur,t, T_left, T_right), ...
                           x, t_span_pde);
            
            % Extract the result at the FINAL time step (end of the vector)
            T_pdepe = sol_pde(end, :);
            
            % Plotting
            plot(x, T_ana, '-', 'Color', colors(j,:), 'LineWidth', 1.5, ...
                'DisplayName', sprintf('t=%ds (Ana)', t_final));
            plot(x, T_cn, 'o', 'Color', colors(j,:), 'MarkerSize', 4, ...
                'DisplayName', sprintf('t=%ds (CN)', t_final));
            plot(x, T_pdepe, '--', 'Color', 'k', 'LineWidth', 1, ...
                'HandleVisibility', 'off'); % Dashed black line for PDEPE check
        end
        legend('show', 'Location', 'best');
        hold off;
    end
end

% --- Manual Numerical Scheme (Crank-Nicholson) ---
function T = solve_crank_nicholson(x, dx, dt, t_final, alpha, T1, T2, T_init)
    Nx = length(x) - 1;
    r = alpha * dt / (dx^2);
    Steps = round(t_final / dt);
    
    % Initialize Temp Vector
    T = ones(Nx+1, 1) * T_init;
    T(1) = T1;
    T(end) = T2;
    
    % Matrices for Linear System: A * T_new = B * T_old + BCs
    M = Nx - 1; % Internal nodes
    
    % LHS Matrix A (Implicit): Diag = 1+r, Off-diag = -r/2
    A = diag((1+r)*ones(1,M)) + diag(-r/2*ones(1,M-1), 1) + diag(-r/2*ones(1,M-1), -1);
    
    % RHS Matrix B (Explicit): Diag = 1-r, Off-diag = r/2
    B = diag((1-r)*ones(1,M)) + diag(r/2*ones(1,M-1), 1) + diag(r/2*ones(1,M-1), -1);
    
    for n = 1:Steps
        T_inner = T(2:end-1);
        
        % Calculate explicit side
        RHS = B * T_inner;
        
        % Add Boundary Conditions
        RHS(1) = RHS(1) + (r/2)*T(1) + (r/2)*T(1);     % From Left BC
        RHS(end) = RHS(end) + (r/2)*T(end) + (r/2)*T(end); % From Right BC
        
        % Solve System
        T_new_inner = A \ RHS;
        T(2:end-1) = T_new_inner;
    end
end

% --- Analytical Solution Function ---
function T = solve_analytical(x, t, alpha, L, T1, T2)
    % Steady State Part
    T_steady = T1 + (T2 - T1) .* (x ./ L);
    
    % Transient Part (Infinite Series)
    sum_val = zeros(size(x));
    for n = 1:50 
        lambda = n * pi / L;
        % Integral of Initial Condition f(x)=350
        integral_term = (350 / lambda) * (1 - cos(n*pi));
        
        brace_term = (T1 - T2*cos(n*pi))/lambda - integral_term;
        
        term = exp(-alpha * lambda^2 * t) .* sin(lambda .* x) .* brace_term;
        sum_val = sum_val + term;
    end
    
    T = T_steady - (2/L) * sum_val;
end

% --- PDEPE Helper Functions ---
function [c,f,s] = pdefun(x,t,u,DuDx, alpha)
    c = 1/alpha; 
    f = DuDx; 
    s = 0;
end

function u0 = icfun(x, T_init)
    u0 = 350; 
end

function [pl,ql,pr,qr] = bcfun(xl,ul,xr,ur,t, T_left, T_right)
    pl = ul - 300; ql = 0;
    pr = ur - 400; qr = 0;
end