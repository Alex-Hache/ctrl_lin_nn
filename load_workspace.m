function load_workspace(strMatFileWeights, net_dims, dt, fig_name)
%load_workspace Load variables used for closed_loop simulation

nu = net_dims(1);
np = net_dims(2);
nx = net_dims(3);
assignin('base', 'nu', nu);
assignin('base', 'np', np);
assignin('base', 'nx', nx);

extract_model_params_from_python(strMatFileWeights, nu, np);
assignin('base', 'dt', dt);
assignin('base', 'name_fig1', fullfile(pwd, [fig_name, '_1.fig']));
sprintf(" Path to figure 1 : %s", fullfile(pwd, [fig_name, '_1.fig']));
assignin('base', 'name_fig2', fullfile(pwd, [fig_name, '_2.fig']));
sprintf(" Path to figure 2 : %s", fullfile(pwd, [fig_name, '_2.fig']));


end

