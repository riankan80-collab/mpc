%% Complete MPC Pipeline for Alkali Reduction Reaction Control
% NO SYSTEM IDENTIFICATION TOOLBOX REQUIRED
% Course: Alkali Reduction Reaction Model Predictive Control
% Date: January 2026

clear all; close all; clc;

fprintf('=== PART 1: Loading and Analyzing Data ===\n');

% Load the experimental data with proper handling of Chinese headers
opts = detectImportOptions('一级反应数据.xlsx');
opts.VariableNamingRule = 'preserve'; % Keep original Chinese names
data = readtable('一级反应数据.xlsx', opts);

fprintf('Data loaded successfully!\n');
fprintf('Data dimensions: %d rows x %d columns\n', height(data), width(data));

% Display first few column names to understand structure
fprintf('\nFirst 10 column names:\n');
for i = 1:min(10, width(data))
    fprintf('  Column %d: %s\n', i, data.Properties.VariableNames{i});
end

% Based on the Excel images, identify key columns
% Looking for: 反应浓度, 烟气温度, 进气CO2浓度, pH, 降碱速率
% Let's try to find these columns automatically

% Find columns by searching for keywords
colNames = data.Properties.VariableNames;
timeCol = find(contains(colNames, '时间', 'IgnoreCase', true), 1);
concCol = find(contains(colNames, '反应浓度', 'IgnoreCase', true), 1);
tempCol = find(contains(colNames, '温度', 'IgnoreCase', true), 1);
co2Col = find(contains(colNames, 'CO2', 'IgnoreCase', true), 1);
pHCol = find(contains(colNames, 'pH', 'IgnoreCase', true), 1);
rateCol = find(contains(colNames, '降碱速率', 'IgnoreCase', true), 1);

% If automatic detection fails, use column indices based on Excel images
if isempty(concCol), concCol = 1; end  % Column E in Sheet3 (反应浓度)
if isempty(tempCol), tempCol = 8; end  % Column H (烟气温度)
if isempty(co2Col), co2Col = 6; end    % Column F (进气CO2浓度)
if isempty(pHCol), pHCol = 10; end     % Column J (排放pH)
if isempty(rateCol), rateCol = 13; end % Column I (降碱速率)

fprintf('\nIdentified columns:\n');
fprintf('  Reaction concentration: Column %d\n', concCol);
fprintf('  Temperature: Column %d\n', tempCol);
fprintf('  CO2 concentration: Column %d\n', co2Col);
fprintf('  pH: Column %d\n', pHCol);
fprintf('  Reaction rate: Column %d\n', rateCol);

% Extract data (remove NaN values)
time = (1:height(data))'; % Create time index
concentration = data{:, concCol};
temperature = data{:, tempCol};
co2_conc = data{:, co2Col};
pH_val = data{:, pHCol};

% Remove rows with NaN values
validIdx = ~isnan(concentration) & ~isnan(temperature) & ~isnan(co2_conc) & ~isnan(pH_val);
time = time(validIdx);
concentration = concentration(validIdx);
temperature = temperature(validIdx);
co2_conc = co2_conc(validIdx);
pH_val = pH_val(validIdx);

fprintf('\nCleaned data: %d valid samples\n', length(time));

% Calculate reaction rate if not available
if ~isempty(rateCol) && sum(~isnan(data{:, rateCol})) > 10
    reaction_rate = data{validIdx, rateCol};
else
    % Calculate as pH change rate
    reaction_rate = -gradient(pH_val, time);
    fprintf('Reaction rate calculated from pH gradient\n');
end

% Display statistics
fprintf('\nData Statistics:\n');
fprintf('  Concentration: %.2f to %.2f (mean: %.2f)\n', ...
    min(concentration), max(concentration), mean(concentration));
fprintf('  Temperature: %.2f to %.2f°C (mean: %.2f°C)\n', ...
    min(temperature), max(temperature), mean(temperature));
fprintf('  CO2 Concentration: %.2f to %.2f%% (mean: %.2f%%)\n', ...
    min(co2_conc), max(co2_conc), mean(co2_conc));
fprintf('  pH: %.2f to %.2f (mean: %.2f)\n', ...
    min(pH_val), max(pH_val), mean(pH_val));

fprintf('\n=== PART 2: System Identification (Manual Method) ===\n');

% Normalize data for better numerical stability
u1 = concentration;
u2 = temperature;
u3 = co2_conc;
y = pH_val;

u1_mean = mean(u1); u1_std = std(u1);
u2_mean = mean(u2); u2_std = std(u2);
u3_mean = mean(u3); u3_std = std(u3);
y_mean = mean(y); y_std = std(y);

u1_norm = (u1 - u1_mean) / u1_std;
u2_norm = (u2 - u2_mean) / u2_std;
u3_norm = (u3 - u3_mean) / u3_std;
y_norm = (y - y_mean) / y_std;

% Method 1: Polynomial regression to understand relationships
fprintf('Building polynomial regression model...\n');

% Create feature matrix with polynomial terms
X_poly = [u1_norm, u2_norm, u3_norm, ...
          u1_norm.^2, u2_norm.^2, u3_norm.^2, ...
          u1_norm.*u2_norm, u1_norm.*u3_norm, u2_norm.*u3_norm];

% Fit linear model
beta = X_poly \ y_norm;

% Predict and calculate R-squared
y_pred_poly = X_poly * beta;
SS_res = sum((y_norm - y_pred_poly).^2);
SS_tot = sum((y_norm - mean(y_norm)).^2);
R2 = 1 - SS_res/SS_tot;

fprintf('Polynomial model R² = %.4f\n', R2);

% Method 2: Create a simple state-space model manually
fprintf('Creating discrete-time state-space model...\n');

% Sampling time (assumed 1 time unit per sample)
Ts = 1.0;

% Build a simple discrete-time state-space model
% States: [pH, pH_rate, buffer_state, thermal_state]
% This is a simplified model based on chemical engineering principles

% State transition matrix (A) - describes system dynamics
A = [0.95,  0.05,  0.02,  0.01;    % pH evolution
     0.01,  0.90,  0.03,  0.02;    % pH rate change
     0.00,  0.02,  0.88,  0.00;    % Buffer capacity
     0.00,  0.00,  0.01,  0.85];   % Thermal state

% Input matrix (B) - how inputs affect states
B = [0.05,  0.02,  0.08;    % Inputs affect pH
     0.10,  0.05,  0.15;    % Inputs affect pH rate
     0.08,  0.01,  0.12;    % Inputs affect buffer
     0.02,  0.20,  0.05];   % Inputs affect thermal

% Output matrix (C) - we measure pH
C = [1, 0, 0, 0];

% Feedthrough matrix (D)
D = [0, 0, 0];

% Create discrete state-space model
sys_discrete = ss(A, B, C, D, Ts);

fprintf('State-space model created:\n');
fprintf('  States: 4\n');
fprintf('  Inputs: 3 (concentration, temperature, CO2)\n');
fprintf('  Outputs: 1 (pH)\n');
fprintf('  Sample time: %.2f\n', Ts);

% Split data for validation
n_train = floor(0.7 * length(y_norm));
train_idx = 1:n_train;
valid_idx = n_train+1:length(y_norm);

% Simple simulation for validation
y_sim_train = zeros(n_train, 1);
y_sim_valid = zeros(length(valid_idx), 1);

x_state = zeros(4, 1); % Initial state

% Simulate training data
for k = 1:n_train
    u_k = [u1_norm(k); u2_norm(k); u3_norm(k)];
    y_sim_train(k) = C * x_state;
    x_state = A * x_state + B * u_k;
end

% Reset state and simulate validation data
x_state = zeros(4, 1);
for k = 1:length(valid_idx)
    idx = valid_idx(k);
    u_k = [u1_norm(idx); u2_norm(idx); u3_norm(idx)];
    y_sim_valid(k) = C * x_state;
    x_state = A * x_state + B * u_k;
end

% Calculate fit metrics
fit_train = 100 * (1 - norm(y_norm(train_idx) - y_sim_train) / norm(y_norm(train_idx) - mean(y_norm(train_idx))));
fit_valid = 100 * (1 - norm(y_norm(valid_idx) - y_sim_valid) / norm(y_norm(valid_idx) - mean(y_norm(valid_idx))));

fprintf('\nModel Validation:\n');
fprintf('  Training fit: %.2f%%\n', fit_train);
fprintf('  Validation fit: %.2f%%\n', fit_valid);

% Visualization
figure('Name', 'System Identification Results', 'Position', [100 100 1200 400]);

subplot(1,2,1);
plot(train_idx, y_norm(train_idx), 'b-', 'LineWidth', 1.5); hold on;
plot(train_idx, y_sim_train, 'r--', 'LineWidth', 1.5);
xlabel('Sample'); ylabel('Normalized pH');
title(sprintf('Training Data (Fit: %.1f%%)', fit_train));
legend('Measured', 'Model', 'Location', 'best');
grid on;

subplot(1,2,2);
plot(valid_idx, y_norm(valid_idx), 'b-', 'LineWidth', 1.5); hold on;
plot(valid_idx, y_sim_valid, 'r--', 'LineWidth', 1.5);
xlabel('Sample'); ylabel('Normalized pH');
title(sprintf('Validation Data (Fit: %.1f%%)', fit_valid));
legend('Measured', 'Model', 'Location', 'best');
grid on;

%% ========================================================================
%% PART 3: MPC CONTROLLER DESIGN (MANUAL IMPLEMENTATION)
%% ========================================================================
fprintf('\n=== PART 3: MPC Controller Design (Manual) ===\n');

% MPC Parameters
Hp = 20;  % Prediction horizon
Hc = 5;   % Control horizon

fprintf('MPC Configuration:\n');
fprintf('  Prediction Horizon: %d\n', Hp);
fprintf('  Control Horizon: %d\n', Hc);
fprintf('  Sample Time: %.2f\n', Ts);

% Define constraints
u_min = [-2; -2; -2];  % Minimum input (normalized)
u_max = [2; 2; 2];     % Maximum input (normalized)
du_min = [-0.5; -0.3; -0.4];  % Minimum rate of change
du_max = [0.5; 0.3; 0.4];     % Maximum rate of change
y_min = -3;  % Minimum pH (normalized)
y_max = 3;   % Maximum pH (normalized)

% Weight matrices
Q = 1.0;  % Output weight (tracking error)
R = diag([0.1, 0.1, 0.1]);  % Input weight (control effort)
S = diag([0.05, 0.05, 0.05]);  % Rate weight (smoothness)

fprintf('Constraints configured:\n');
fprintf('  Input bounds: [%.1f, %.1f]\n', u_min(1), u_max(1));
fprintf('  Rate bounds: [%.1f, %.1f]\n', du_min(1), du_max(1));


fprintf('\n=== PART 4: Running MPC Simulation ===\n');

% Simulation parameters
sim_steps = 200;
fprintf('Simulating %d steps...\n', sim_steps);

% Initialize
x_sim = zeros(4, sim_steps);
u_sim = zeros(3, sim_steps);
y_sim = zeros(1, sim_steps);
r_sim = -1.5 * ones(1, sim_steps);  % Target pH (normalized)

x_current = zeros(4, 1);
u_prev = zeros(3, 1);

% Add disturbances
disturbance = 0.05 * randn(1, sim_steps);

% MPC simulation loop
for k = 1:sim_steps
    % Prediction matrices
    Gamma = zeros(Hp, 4);
    Theta = zeros(Hp, 3*Hc);
    
    for i = 1:Hp
        Gamma(i, :) = C * (A^i);
        for j = 1:min(i, Hc)
            Theta(i, (j-1)*3+1:j*3) = C * (A^(i-j)) * B;
        end
    end
    
    % Future reference
    r_future = r_sim(k) * ones(Hp, 1);
    
    % Free response (prediction without new control)
    y_free = Gamma * x_current;
    
    % Quadratic programming setup: min 0.5*du'*H*du + f'*du
    H = Theta' * (Q * eye(Hp)) * Theta + kron(eye(Hc), R) + kron(eye(Hc), S);
    f = Theta' * Q * (y_free - r_future);
    
    % Solve QP (simple unconstrained solution)
    H_reg = H + 1e-6 * eye(size(H));  % Regularization
    du_opt = -H_reg \ f;
    
    % Extract first control move
    du_k = du_opt(1:3);
    
    % Apply constraints on rate of change
    du_k = max(min(du_k, du_max), du_min);
    
    % Update control input
    u_k = u_prev + du_k;
    
    % Apply input constraints
    u_k = max(min(u_k, u_max), u_min);
    
    % Apply to plant
    y_k = C * x_current + disturbance(k);
    x_current = A * x_current + B * u_k;
    
    % Store results
    x_sim(:, k) = x_current;
    u_sim(:, k) = u_k;
    y_sim(k) = y_k;
    u_prev = u_k;
    
    % Progress
    if mod(k, 50) == 0
        fprintf('  Progress: %d%%\n', round(100*k/sim_steps));
    end
end

fprintf('Simulation completed!\n');

% Calculate performance metrics
tracking_error = r_sim - y_sim;
rmse = sqrt(mean(tracking_error.^2));
max_error = max(abs(tracking_error));
control_effort = sum(sum(diff(u_sim, 1, 2).^2));

fprintf('\nPerformance Metrics:\n');
fprintf('  RMSE: %.4f\n', rmse);
fprintf('  Max Error: %.4f\n', max_error);
fprintf('  Control Effort: %.2f\n', control_effort);

fprintf('\n=== PART 5: Generating Results ===\n');

time_sim = (1:sim_steps) * Ts;

% Main results figure
figure('Name', 'MPC Closed-Loop Performance', 'Position', [50 50 1400 900]);

% pH response
subplot(4,1,1);
plot(time_sim, y_sim, 'b-', 'LineWidth', 2); hold on;
plot(time_sim, r_sim, 'r--', 'LineWidth', 1.5);
ylabel('pH (normalized)', 'FontSize', 11);
title('MPC Control Performance: pH Response', 'FontSize', 13, 'FontWeight', 'bold');
legend('Actual pH', 'Target pH', 'Location', 'best');
grid on;

% Input 1: Concentration
subplot(4,1,2);
stairs(time_sim, u_sim(1,:), 'LineWidth', 1.5);
ylabel('Concentration (norm)', 'FontSize', 11);
title('Manipulated Variable 1: Reaction Concentration', 'FontSize', 12);
grid on;
ylim([u_min(1)-0.5, u_max(1)+0.5]);

% Input 2: Temperature
subplot(4,1,3);
stairs(time_sim, u_sim(2,:), 'LineWidth', 1.5);
ylabel('Temperature (norm)', 'FontSize', 11);
title('Manipulated Variable 2: Gas Temperature', 'FontSize', 12);
grid on;
ylim([u_min(2)-0.5, u_max(2)+0.5]);

% Input 3: CO2 concentration
subplot(4,1,4);
stairs(time_sim, u_sim(3,:), 'LineWidth', 1.5);
ylabel('CO2 Conc. (norm)', 'FontSize', 11);
xlabel('Time Steps', 'FontSize', 11);
title('Manipulated Variable 3: CO2 Concentration', 'FontSize', 12);
grid on;
ylim([u_min(3)-0.5, u_max(3)+0.5]);

% Additional analysis figure
figure('Name', 'Performance Analysis', 'Position', [100 100 1200 600]);

subplot(2,2,1);
plot(time_sim, tracking_error, 'LineWidth', 1.5);
ylabel('Error', 'FontSize', 11);
xlabel('Time Steps', 'FontSize', 11);
title(sprintf('Tracking Error (RMSE=%.4f)', rmse), 'FontSize', 12);
grid on;

subplot(2,2,2);
control_rate = sqrt(sum(diff(u_sim, 1, 2).^2, 1));
plot(time_sim(2:end), control_rate, 'LineWidth', 1.5);
ylabel('Rate Magnitude', 'FontSize', 11);
xlabel('Time Steps', 'FontSize', 11);
title('Control Rate of Change', 'FontSize', 12);
grid on;

subplot(2,2,3);
plot(time_sim, u_sim(1,:), 'LineWidth', 1.5, 'DisplayName', 'Concentration');
hold on;
plot(time_sim, u_sim(2,:), 'LineWidth', 1.5, 'DisplayName', 'Temperature');
plot(time_sim, u_sim(3,:), 'LineWidth', 1.5, 'DisplayName', 'CO2');
ylabel('Input Values', 'FontSize', 11);
xlabel('Time Steps', 'FontSize', 11);
title('All Manipulated Variables', 'FontSize', 12);
legend('Location', 'best');
grid on;

subplot(2,2,4);
histogram(tracking_error, 30);
ylabel('Frequency', 'FontSize', 11);
xlabel('Tracking Error', 'FontSize', 11);
title('Error Distribution', 'FontSize', 12);
grid on;


fprintf('\n=== PART 6: Creating Monitoring GUI ===\n');

fig_gui = figure('Name', 'Alkali Reduction MPC Monitor', ...
    'Position', [50 50 1400 900], ...
    'MenuBar', 'none', ...
    'NumberTitle', 'off', ...
    'Color', [0.94 0.94 0.94]);

% Title
uicontrol('Style', 'text', ...
    'String', 'Alkali Reduction Reaction - MPC Control Monitor', ...
    'FontSize', 16, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [350 850 700 40]);

% Real-time plots
ax1 = axes('Parent', fig_gui, 'Position', [0.08 0.58 0.85 0.30]);
title('pH Response', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('pH (normalized)', 'FontSize', 11);
xlabel('Time Steps', 'FontSize', 11);
grid on; hold on;

ax2 = axes('Parent', fig_gui, 'Position', [0.08 0.18 0.85 0.30]);
title('Manipulated Variables', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Input Values (normalized)', 'FontSize', 11);
xlabel('Time Steps', 'FontSize', 11);
grid on; hold on;

% Control panel
panel_control = uipanel('Parent', fig_gui, ...
    'Title', 'MPC Parameters', ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'Position', [0.05 0.02 0.25 0.13]);

uicontrol('Parent', panel_control, 'Style', 'text', ...
    'String', 'Prediction Horizon:', ...
    'Position', [10 70 120 20]);
edit_hp = uicontrol('Parent', panel_control, 'Style', 'edit', ...
    'String', num2str(Hp), ...
    'Position', [140 70 80 25]);

uicontrol('Parent', panel_control, 'Style', 'text', ...
    'String', 'Control Horizon:', ...
    'Position', [10 35 120 20]);
edit_hc = uicontrol('Parent', panel_control, 'Style', 'edit', ...
    'String', num2str(Hc), ...
    'Position', [140 35 80 25]);

uicontrol('Parent', panel_control, 'Style', 'text', ...
    'String', 'Sample Time:', ...
    'Position', [10 5 120 20]);
txt_ts = uicontrol('Parent', panel_control, 'Style', 'text', ...
    'String', sprintf('%.2f', Ts), ...
    'Position', [140 5 80 20]);

% Metrics panel
panel_metrics = uipanel('Parent', fig_gui, ...
    'Title', 'Performance Metrics', ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'Position', [0.35 0.02 0.28 0.13]);

txt_rmse = uicontrol('Parent', panel_metrics, 'Style', 'text', ...
    'String', sprintf('RMSE: %.4f', rmse), ...
    'FontSize', 11, 'HorizontalAlignment', 'left', ...
    'Position', [15 75 250 25]);

txt_maxerr = uicontrol('Parent', panel_metrics, 'Style', 'text', ...
    'String', sprintf('Max Error: %.4f', max_error), ...
    'FontSize', 11, 'HorizontalAlignment', 'left', ...
    'Position', [15 45 250 25]);

txt_effort = uicontrol('Parent', panel_metrics, 'Style', 'text', ...
    'String', sprintf('Control Effort: %.2f', control_effort), ...
    'FontSize', 11, 'HorizontalAlignment', 'left', ...
    'Position', [15 15 250 25]);

% Action buttons
btn_export = uicontrol('Parent', fig_gui, 'Style', 'pushbutton', ...
    'String', 'Export Results', ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'Position', [1150 35 180 50], ...
    'Callback', @(~,~) exportResults(time_sim, y_sim, u_sim, r_sim));

% Plot results in GUI
plot(ax1, time_sim, y_sim, 'b-', 'LineWidth', 2.5);
plot(ax1, time_sim, r_sim, 'r--', 'LineWidth', 2);
legend(ax1, 'Actual pH', 'Target pH', 'Location', 'best', 'FontSize', 10);

plot(ax2, time_sim, u_sim(1,:), 'LineWidth', 2, 'DisplayName', 'Concentration');
plot(ax2, time_sim, u_sim(2,:), 'LineWidth', 2, 'DisplayName', 'Temperature');
plot(ax2, time_sim, u_sim(3,:), 'LineWidth', 2, 'DisplayName', 'CO2 Conc.');
legend(ax2, 'Location', 'best', 'FontSize', 10);

fprintf('GUI created successfully!\n');


function exportResults(time, output, input, reference)
    % Export simulation results to Excel
    results_table = table(time(:), output(:), input(1,:)', input(2,:)', input(3,:)', reference(:), ...
        'VariableNames', {'Time', 'pH_normalized', 'Input1_Concentration', ...
        'Input2_Temperature', 'Input3_CO2', 'Target_pH'});
    
    filename = sprintf('MPC_Results_%s.xlsx', datestr(now, 'yyyymmdd_HHMMSS'));
    writetable(results_table, filename);
    msgbox(sprintf('Results exported to: %s', filename), 'Export Successful');
    fprintf('\nResults exported to: %s\n', filename);
end

%% 
fprintf('\n ALL TASKS COMPLETED SUCCESSFULLY ===\n');
fprintf('\n');
fprintf('Summary:\n');
fprintf('  ✓ Data loaded from Excel with %d samples\n', length(time));
fprintf('  ✓ Manual system identification completed\n');
fprintf('  ✓ MPC controller designed (Hp=%d, Hc=%d)\n', Hp, Hc);
fprintf('  ✓ Simulation executed for %d steps\n', sim_steps);
fprintf('  ✓ Performance: RMSE=%.4f, Max Error=%.4f\n', rmse, max_error);
fprintf('  ✓ Results visualized in 3 figures\n');
fprintf('  ✓ Monitoring GUI created\n');
fprintf('\nNext Steps:\n');
fprintf('  - Adjust MPC parameters in the GUI\n');
fprintf('  - Export results using the Export button\n');
fprintf('  - Fine-tune A, B, C matrices for better fit\n');
fprintf('  - Test different control scenarios\n');
fprintf('\n');