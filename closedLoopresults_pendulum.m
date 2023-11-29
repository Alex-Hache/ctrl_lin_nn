
addpath(genpath('data/pendulum'));
%% Load training and test data
dataTrain = load('data_train.mat');
dataTest = load('data_test.mat');
%% Simulation parameters
nx = 2;

theta0 = 0;

m = 2;
l = 1;
f=3;

g = 9.81;
dt = 0.1;

%% Convert continuous-time to discrete time
% sys_c = ss(A,[B,G],C,0);
% sys_d = c2d(sys_c, dt, 'zoh');
% 
% A = sys_d.a;
% B = sys_d.b(:,1:nu);
% G = sys_d.b(:,nu+1:nu+np);
% C = sys_d.c;
% 
% 
% sys_c_bla = ss(A_bla, [B_bla G_bla], C, 0);
% sys_d_bla = c2d(sys_c_bla, dt, 'zoh');
% 
% A_bla = sys_d_bla.a;
% B_bla = sys_d_bla.b(:,1:nu);
% G_bla = sys_d_bla.b(:,nu+1:nu+np);
% C_bla = sys_d_bla.c;

%% Closed-loop pole placement
e_ol = -abs(eig(A));
perf_cl = linspace(1.1,1.15 ,nx)';

e_cl = e_ol.*perf_cl;
% e_cl = [-1.6602; -1.7356]; % Poles of the baseline model for comparison
K = place(A,B,e_cl); 
% K = zeros(size(K));
eig_pp = eig(A-B*K);

eAo = e_cl*2;

L = place(A',C',eAo)';

sys_cl = ss(A-B*K,B,C,0);
dcg = dcgain(sys_cl);

eig_obs  = eig(A-L*C);


%% Comparaison avec le modèle BLA
% e_ol_bla = -abs(eig(A_bla));
% e_cl_bla = e_ol_bla.*perf_cl;
K_bla = place(A_bla, B_bla, e_cl);
% K_bla = zeros(size(K_bla));
eib_bla_pp = eig(A_bla - B_bla*K_bla);
sys_bla_cl = ss(A_bla-B_bla*K_bla, B_bla, C_bla,0);
dcg_bla = dcgain(sys_bla_cl);

eAo_bla = e_cl*2;

L_bla = place(A_bla',C_bla',eAo_bla)';

eig_bla_obs = eig(A_bla-L_bla*C_bla);

%% Comparaison normes hinf entre les 2 modèles
h_inf_model = hinfnorm(ss(A-B*K,G,C,0));
h_inf_bla = hinfnorm(ss(A_bla-B_bla*K_bla,G_bla,C_bla,0));

h_inf_model_no_pp = hinfnorm(ss(A,G,C,0));
h_inf_bla_no_pp = hinfnorm(ss(A_bla,G_bla,C_bla,0));


%% Figures
set(0, 'DefaultLineLinewidth', 2.5);
set(0, 'DefaultConstantLineLinewidth', 2.5)
set(0, 'DefaultConstantLineFontSize', 20)
set(0, 'DefaultLegendLocation', 'best')
set(0, 'DefaultAxesFontSize', 24)
set(0, 'DefaultConstantLineHandleVisibility', 'off')
% set(0, 'DefaultTitleFontweight', 'normal')

sim_out = sim('stateFeedback_closedLoop');
% sim_out = sim('stateFeedback_closedLoop_scaled');

% Have a very small dataset (on purpose), the validation is done on steps
% so the bounds we consider here is then the one of the first half which
% does not include the sines.

% Max
max_u = max(dataTrain.uTot(:,1:2000), [], 2);
max_p = max(dataTrain.pTot(:,1:2000), [], 2);
max_y = max(dataTrain.yTot(1:2000,:)', [], 2);

% Min
min_u = min(dataTrain.uTot(:,1:2000), [], 2);
min_p = min(dataTrain.pTot(:,1:2000), [], 2);
min_y = min(dataTrain.yTot(1:2000,:)', [], 2);

fig1 = figure;
h1 = subplot(1,2,1);
hold on
title('Inputs', 'FontWeight', 'normal')
plot(sim_out.t,sim_out.u_lin, 'b','DisplayName', 'u_{lin}')
plot(sim_out.t,sim_out.d, 'r', 'DisplayName', 'd')
yline(min_u, 'b--','u_{min}')
yline(max_u, 'b--','u_{max}')
yline(min_p, 'r--','d_{min}')
yline(max_p, 'r--','d_{max}')
lgd = legend('show', 'Location', 'best');
lgd.FontSize = 20;
% lgd.FontWeight = 'bold';
xlabel('t (s)')
ylabel('Torque input (Nm)')

h2 = subplot(1,2,2);
title('Outputs', 'FontWeight', 'normal')
hold on
plot(sim_out.t,sim_out.y_sys, 'DisplayName', 'y_{sys}')
plot(sim_out.t,sim_out.y_mod, 'DisplayName', 'y_{lin}')
plot(sim_out.t,sim_out.y_obs, 'DisplayName', 'y_{obs}')
% plot(sim_out.t,sim_out.y_sys_bla, 'DisplayName', 'y_{sys_{bla}}')
plot(sim_out.t,sim_out.ref, 'k--', 'DisplayName', 'r')
yline(min_y*180/pi, '--', 'y_{min}')
yline(max_y*180/pi, '--', 'y_{max}')
lgd = legend('show','Location', 'best', 'NumColumns', 4);
lgd.FontSize = 20;
% lgd.FontWeight = 'bold';
ylabel('Angle (Deg)', 'FontSize',24)
xlabel('t (s)', 'FontSize',24)
ylim([(min_y*180/pi-10) (max_y*180/pi + 10)])
linkaxes([h1 h2], 'x');

savefig(fig1, name_fig1);
