clear all
close all
clc
restoredefaultpath;
addpath(genpath('~/Bureau/Recherche/Simulation/papier_l4DC/results/pendulum/seed_42/rnn/flnssm/hinfnn/gamma_0.001'))
addpath(genpath('data/pendulum'));
strMatFileWeights = 'model.mat';
nu = 1;
nx = 2;
np = 1;
dt = 0.1;
fig_name = 'Test';
assignin('base', 'name_fig1', fullfile(pwd, 'figures', [fig_name, '_1.fig']));
sprintf(" Path to figure 1 : %s", fullfile(pwd, 'figures', [fig_name, '_1.fig']));
assignin('base', 'name_fig2', fullfile(pwd, 'figures', [fig_name, '_2.fig']));
sprintf(" Path to figure 2 : %s", fullfile(pwd, 'figures', [fig_name, '_2.fig']));

extract_model_params_from_python(strMatFileWeights, nu, np);
closedLoopresults_pendulum;