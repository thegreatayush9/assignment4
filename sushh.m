% --- CSTR Dynamic Simulation (Question 1b) ---
clear; clc; close all;

% 1. Define Parameters
p.F = 1.0;
p.V = 1.0;
p.k0 = 36e6;
p.neg_dH = 6500.0;
p.E = 12000.0;
p.rhocp = 500.0;
p.Tf = 298.0;
p.CAf = 10.0;
p.UA = 150.0;
p.Tj0 = 298.0;
p.rhojcj = 600.0;
p.Fj = 1.25;
p.Vj = 0.25;
p.R = 1.987;

% 2. Simulation Setup
t_span = [0 50]; % Simulate for 50 seconds

% Initial Conditions (Using State 1 from the Python output, slightly perturbed)
% State 1 roughly: CA=8.96, T=308.7, Tj=299.8
x0 = [8.968 * 1.01; 308.727 * 1.01; 299.788 * 1.01]; 

% 3. Solve ODE using ode45
[t, x] = ode45(@(t,x) cstr_ode(t, x, p), t_span, x0);

% 4. Plotting
figure('Name', 'CSTR Dynamics');

subplot(3,1,1);
plot(t, x(:,1), 'LineWidth', 1.5);
ylabel('C_A (mol/L)');
title('Concentration vs Time');
grid on;

subplot(3,1,2);
plot(t, x(:,2), 'r', 'LineWidth', 1.5);
ylabel('T (K)');
title('Reactor Temperature vs Time');
grid on;

subplot(3,1,3);
plot(t, x(:,3), 'g', 'LineWidth', 1.5);
ylabel('T_j (K)');
xlabel('Time (s)');
title('Jacket Temperature vs Time');
grid on;

% --- ODE Function ---
function dxdt = cstr_ode(~, x, p)
    CA = x(1);
    T  = x(2);
    Tj = x(3);
    
    % Reaction Rate: r = k0 * CA * exp(-E/RT)
    r = CA * p.k0 * exp(-p.E / (p.R * T));
    
    % Mass Balance: dCA/dt
    dCA = (p.F * (p.CAf - CA) - p.V * r) / p.V;
    
    % Energy Balance: dT/dt
    dT = (p.F * p.rhocp * (p.Tf - T) + p.neg_dH * p.V * r - p.UA * (T - Tj)) / (p.rhocp * p.V);
    
    % Jacket Balance: dTj/dt
    dTj = (p.Fj * p.rhojcj * (p.Tj0 - Tj) + p.UA * (T - Tj)) / (p.rhojcj * p.Vj);
    
    dxdt = [dCA; dT; dTj];
end