% --- Fin Heat Transfer Shooting Method (Question 2c) ---
clear; clc; close all;

% 1. Define Parameters
global p
p.T0 = 300;      % Boundary at x=0
p.TL = 400;      % Boundary at x=L
p.Tinf = 200;    % Ambient Temp
p.L = 10;        % Length
p.alpha = 0.05;  % Convection coeff
p.beta = 2.7e-9; % Radiation coeff

% 2. Shooting Method Logic
% We need to find the initial slope (dT/dx at x=0) such that T(L) = 400.
% We define a function 'objective' that returns the error at x=L.

% Guess a range for the slope [-50, 50] to search within
slope_guess_range = [-50, 50]; 

% Use fzero to find the correct slope
initial_slope = fzero(@shooting_objective, slope_guess_range);

fprintf('Converged Initial Slope (dT/dx): %.4f\n', initial_slope);

% 3. Final Simulation with Correct Slope
[x_vals, y_vals] = ode45(@fin_ode, [0 p.L], [p.T0; initial_slope]);

% 4. Plotting
figure('Name', 'Fin Temperature Profile');
plot(x_vals, y_vals(:,1), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Length x');
ylabel('Temperature T(x)');
title('Temperature Distribution along Fin (Shooting Method)');
grid on;
hold on;
yline(p.TL, 'r--', 'Target T_L');
legend('Temperature Profile', 'Target Boundary Condition');


% --- Helper Functions ---

% The objective function: Error = T_calc(L) - T_target
function error = shooting_objective(slope_guess)
    global p
    [~, Y] = ode45(@fin_ode, [0 p.L], [p.T0; slope_guess]);
    T_L_calculated = Y(end, 1); % Temperature at x=L
    error = T_L_calculated - p.TL;
end

% The System of ODEs (Reduced from 2nd order to two 1st order)
function dydx = fin_ode(~, y)
    global p
    T = y(1);
    dT = y(2); % This is dT/dx
    
    % d2T/dx2 = alpha*(T - Tinf) + beta*(T^4 - Tinf^4)
    d2T = p.alpha * (T - p.Tinf) + p.beta * (T^4 - p.Tinf^4);
    
    dydx = [dT; d2T];
end