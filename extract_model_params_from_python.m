function  extract_model_params_from_python(strMatFileWeights, nu, np)
sModelWeights = load(strMatFileWeights);


A = double(sModelWeights.A);
B = double(sModelWeights.Bu);
G = double(sModelWeights.Bd);
C = double(sModelWeights.C);

W_y_in = double(sModelWeights.alpha_in_y);
b_alpha_in = double(sModelWeights.alpha_in_y_bias)';
W_p_in = double(sModelWeights.alpha_in_d);
W_out  = double(sModelWeights.alpha_out);
b_alpha_out = double(sModelWeights.alpha_out_bias)';



sBLA = load('pend_BLA.mat');


A_bla = sBLA.A;
B_bla = sBLA.B(:, 1:nu);
C_bla = sBLA.C;
G_bla = sBLA.B(:,nu+1:nu+np);


%% Populate base workspace for simulation
assignin('base', 'A', A);
assignin('base', 'B', B);
assignin('base', 'C', C);
assignin('base', 'G', G);

assignin('base','W_y_in', W_y_in);
assignin('base','b_in', b_alpha_in);
assignin('base', 'W_p_in', W_p_in);
assignin('base', 'W_out', W_out);
assignin('base', 'b_out', b_alpha_out);


assignin('base', 'A_bla', A_bla);
assignin('base', 'B_bla', B_bla);
assignin('base', 'C_bla', C_bla);
assignin('base', 'G_bla', G_bla);

end
