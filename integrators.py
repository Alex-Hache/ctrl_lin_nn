import torch
import torch.nn as nn
import time
import numpy as np
import re
from postprocessing import extract_parameter_name

def getIntegrator(config, model, ts):
    integrators = {
            'RK4' : RK4Simulator,
            'discrete' : Sim_discrete,
        }[config.integrator]
    return integrators(model, ts)
    
class Simulator(nn.Module):
    def __init__(self, ss_model, ts) -> None:
        super(Simulator, self).__init__()

        self.ss_model = ss_model
        self.ts = ts

    def clone(self):
        copy_ss = self.ss_model.clone() # State-space model module must have a clone function
        copy = type(self)(copy_ss, self.ts)
        copy.load_state_dict(self.state_dict())
        return copy
    
    def simulate(self, u, x0):
        start_time = time.time()
        with torch.no_grad():
            x_sim, y_sim= self(u[None, :, :], x0[None, :]) # Appel à forward 
            x_sim.squeeze_(0)
            y_sim.squeeze_(0)
            sim_time = time.time() - start_time
            print(f"\nSim time: {sim_time:.2f}") 
        return x_sim, y_sim

    def extract_params(self):
        dict_param = {}
        # Extract the parametrized linear part if has one
        if hasattr(self.ss_model, 'linmod'): # If model has a linear part
            if hasattr(self.ss_model.linmod, "eval_") and callable(getattr(self.ss_model.linmod, 'eval_')):
                A,Bd,C = self.ss_model.linmod.eval_()
                dict_param['A'] = A
                dict_param['Bd'] = Bd
                dict_param['C'] = C

        for name_param, param_tensor in self.named_parameters():
            tensor = param_tensor.detach().numpy()
            name = extract_parameter_name(name_param)
            dict_param[name] = tensor

        return dict_param


    def extract_weights(self):
        weights = []
        biases = []
        for param_tensor in self.state_dict():
            tensor = self.state_dict()[param_tensor].detach().numpy()

            if 'weight' in param_tensor:
                weights.append(tensor)
            if 'bias' in param_tensor:
                biases.append(tensor)
        weights = np.array(weights, dtype= object)
        biases = np.array(biases, dtype = object)
        return weights, biases

    def write_flat_params(self, x):
        r""" Writes vector x to model parameters.."""
        index = 0
        theta = torch.Tensor(x)
        for name, p in self.named_parameters():
            p.data = theta[index:index + p.numel()].view_as(p.data)
            index = index + p.numel()

    def flatten_params(self):
        views = []
        for i,p in enumerate(self.parameters()):
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
class RK4Simulator(Simulator):
    def __init__(self, ss_model, ts):
        super(RK4Simulator, self).__init__(ss_model=ss_model, ts=ts)

    def forward(self, u_batch, x0_batch = torch.Tensor([[0., 0., 0., 0.]])):
        """ Multi-step simulation over (mini)batches

        Parameters
        ----------
        u_batch: Tensor. Size: (m, q, n_u)
            Input sequence for each subsequence in the minibatch

        x0_batch: Tensor. Size: (q, n_x)
            initial state for each sequence in the minibatch
        x_batch: Tensor. Size: (m, q, n_x)
            state sequence for each subsequence in the minibatch
        Returns
        -------
        Tensor. Size: (m, q, n_x)
            Simulated state for all subsequences in the minibatch

        """

      
        X_sim_list = []
        Y_sim_list= []
        x_step = x0_batch

        for u_step in u_batch.split(1, dim =1):#i in range(seq_len):

            u_step = u_step.squeeze(1)
            # x_step = x_step.squeeze(0)
           
            X_sim_list += [x_step]

            dt2 = self.ts / 2.0
            k1,y_step = self.ss_model(u_step, x_step)
            Y_sim_list += [y_step]

            k2_dx, _  = self.ss_model(u_step, x_step + dt2*k1)
            k3_dx, _= self.ss_model(u_step, x_step + dt2*k2_dx)
            k4_dx, _ = self.ss_model(u_step, x_step + self.ts*k3_dx)

            dx = self.ts / 6.0 * (k1 + 2.0 * k2_dx + 2.0 * k3_dx + k4_dx)
            x_step = x_step + dx


        X_sim = torch.stack(X_sim_list, 1) 
        Y_sim = torch.stack(Y_sim_list, 1)
        return X_sim, Y_sim
      
class Sim_discrete(Simulator):
    def __init__(self, ss_model, ts = 1):
        super(Sim_discrete, self).__init__(ss_model=ss_model, ts=ts)

    def forward(self, u_batch, x0_batch = torch.Tensor([[0., 0., 0., 0.]])):
        """ Multi-step simulation over (mini)batches

        Parameters
        ----------
        u_batch: Tensor. Size: (N_batch, seq_len, n_u)
            Input sequence for each subsequence in the minibatch

        x0_batch: Tensor. Size: (N_batch, n_x)
            initial state for each sequence in the minibatch
        x_batch: Tensor. Size: (N_batch, seq_len, n_x)
            state sequence for each subsequence in the minibatch
        Returns
        -------
        Tensor. Size: (N_batch, seq_len, n_x)
            Simulated state and outputs for all subsequences in the minibatch
        """

        X_sim_list = []
        Y_sim_list= []
        x_step = x0_batch
        for u_step in u_batch.split(1, dim = 1):#i in range(seq_len):

            u_step = u_step.squeeze(1)           
            X_sim_list += [x_step]

            x_step,y_step = self.ss_model(u_step, x_step)
            Y_sim_list += [y_step]

        X_sim = torch.stack(X_sim_list, 1) 
        Y_sim = torch.stack(Y_sim_list, 1)
        return X_sim, Y_sim