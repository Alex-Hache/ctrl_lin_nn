import math
import os
import sys

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
from torch.autograd import Function

import geotorch as geo

from preprocessing import *
import cvxpy as cp

import matplotlib.pyplot as plt
from collections import OrderedDict

def getModel(config, data):
    if config.model == 'linear':
        if config.lin != '':
            config.model = config.lin
    models = {
        'linear': NNLinear,
        'alphaLNN' : AlphaLNN,
        'hinfnn' : HinfNN,
        'h2nn' : H2NN,
        'flnssm' : FLNSSM,
        'GRNSSM' : GRNSSM
    }[config.model]

    u_train, y_train, x_train = data
    nu = u_train.shape[1]
    ny = y_train.shape[1]
    return models(nu, ny, config)


class NNLinear(nn.Module):
    """
        neural network module corresponding to a linear state-space model
            x^+ = Ax + Bu
            y = Cx
    """
    def __init__(self, input_dim : int, output_dim: int, config) -> None:
        super(NNLinear, self).__init__()

        self.input_dim = input_dim
        self.state_dim = config.nx
        self.output_dim = output_dim

        self.A = nn.Linear(self.state_dim, self.state_dim, bias = False)
        self.B = nn.Linear(self.input_dim, self.state_dim, bias = False)
        self.C = nn.Linear(self.state_dim, self.output_dim, bias = False)
        self.config = config

    def forward(self, u,x):
        dx = self.A(x) + self.B(u)
        y = self.C(x)

        return dx,y

    def init_model_(self, A0, B0, C0, is_grad = True):
        self.A.weight = nn.parameter.Parameter(A0)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self): # Method called by the simulator 
        copy = type(self)(self.input_dim,self.output_dim, self.config)
        copy.load_state_dict(self.state_dict())
        return copy


class StableA(nn.Module):
    '''
        This module produces a direct parametrization of the A matrix to be alpha-stable
    '''

    def __init__(self, config) -> None:
        super(StableA, self).__init__()
        self.alpha = config.alpha
        self.nx = config.nx
        #Q, P = self.solve_lmi()

        self.Q = Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.Pinv =  Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.S =  Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))

        # register P,Q,S variables to be on specified manifolds
        geo.positive_definite(self, 'Pinv')
        geo.positive_definite(self, 'Q')
        geo.skew(self, 'S')

        # TO-DO : how to enforce specific values from a first LMI resolution ?

    def solve_lmi(self, epsilon = 1e-6, solver = "MOSEK"):

        print(" Initializing Lyap LMI \n")

        A = self.A.detach().numpy()
        P = cp.Variable((self.nx, self.nx), 'P', PSD=True)
        nx = self.nx
        
        M = A.T@P + P@A + 2*self.alpha*P
        constraints = [M << -epsilon*np.eye(nx), P -(epsilon)*np.eye(nx)>> 0 ] 
        objective = cp.Minimize(0) # Feasibility problem

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")


        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        return torch.Tensor(-M.value), torch.Tensor(P.value)

    def forward(self,x):
        return x@((self.Pinv@(-0.5*self.Q + self.S) - self.alpha*torch.eye(self.nx)).T)

    def eval(self):
        return (self.Pinv@(-0.5*self.Q + self.S) - self.alpha*torch.eye(self.nx))
class AlphaLNN(nn.Module):
    """
        Neural network module corresponding to a linear state-space model but with A alpha-stable
            x^+ = Ax + Bu
            y = Cx
    """
    def __init__(self, input_dim : int, output_dim: int, config) -> None:
        super(AlphaLNN, self).__init__()

        self.input_dim = input_dim
        self.state_dim = config.nx
        self.output_dim = output_dim
        self.alpha = config.alpha
        self.A = StableA(config)
        self.B = nn.Linear(self.input_dim, self.state_dim, bias = False)
        self.C = nn.Linear(self.state_dim, self.output_dim, bias = False)
        self.config = config

    def forward(self, u,x):
        dx = self.A(x) + self.B(u)
        y = self.C(x)

        return dx,y

    def init_model_(self, A0, B0, C0, alpha : float, is_grad = True):
        self.A = StableA(A0, alpha=alpha)
        self.B.weight = nn.parameter.Parameter(B0)
        self.C.weight = nn.parameter.Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self): # Method called by the simulator 
        copy = type(self)(self.input_dim, self.output_dim, self.config)
        copy.load_state_dict(self.state_dict())
        return copy
    def eval_(self):
        print(f"A eigenvalues : {torch.linalg.eigvals(self.A.eval())}")

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input
sqrtm = MatrixSquareRoot.apply


class HinfNN(nn.Module):
    def __init__(self, input_dim : int, output_dim: int, config) -> None:
        super(HinfNN, self).__init__()
        self.nu = input_dim
        self.nx = config.nx
        self.ny = output_dim
        self.gamma = config.gamma
        self.config = config

        
        self.Q = Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.P =  Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.S =  Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.G = Parameter(nn.init.normal_(torch.empty((self.ny, self.nx))))
        self.H = Parameter(nn.init.normal_(torch.empty((self.nx, self.nu)))) # H is restriction to input space
        self.eps = torch.Tensor([1e-4])
        # register P,Q,S variables to be on specified manifolds
        geo.positive_definite(self, 'P')
        geo.positive_definite(self, 'Q')
        geo.skew(self, 'S')

    def forward(self, u, x):

        # Building linear matrix system
        Htilde = self.H/(1.01*torch.sqrt(torch.norm(self.H@self.H.T, 2)))
        Asym =  -0.5*(self.Q + self.G.T@ self.G + self.eps* torch.eye(self.nx))
        A = (Asym + self.S)@self.P
        B = self.gamma*sqrtm(self.Q)@Htilde
        C = self.G@ self.P

        dx = x@A.T + u@B.T
        y = x@C.T
        return dx, y

    def eval_(self, abs_tol = 1e-8, solver = "MOSEK", espilon = 1e-8):
        Htilde = self.H/(1.01*torch.sqrt(torch.norm(self.H@self.H.T, 2)))
        Asym =  -0.5*(self.Q + self.G.T@ self.G + self.eps* torch.eye(self.nx))
        A = (Asym + self.S)@self.P
        B = self.gamma*sqrtm(self.Q)@Htilde
        C = self.G@ self.P

        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()
        print(" Initializing HInf LMI \n")

        nu = B.shape[1]
        nx = A.shape[0]
        ny = C.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)
        gamma = cp.Variable()
        D = np.zeros((ny,nu))


        M = cp.bmat([[A.T@P + P@A , P@B, C.T], [B.T@P, -gamma *np.eye(nu), D.T], [ C, D, -gamma*np.eye(ny)]])


        constraints = [M << -np.eye(nu+nx+ny)*espilon, P -(abs_tol)*np.eye(nx)>> 0, gamma - abs_tol >=0] 
        objective = cp.Minimize(gamma) # Find the L2 gain of the BLA

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print("Gamma = {} \n".format(prob.value))
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        
        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        return A,B,C

    def clone(self): # Method called by the simulator 
        copy = type(self)(self.input_dim, self.output_dim, self.config)
        copy.load_state_dict(self.state_dict())
        return copy



class H2NN(nn.Module):
    def __init__(self, input_dim : int, output_dim: int, config) -> None:
        super(H2NN, self).__init__()
        self.nu = input_dim
        self.nx = config.nx
        self.ny = output_dim
        self.gamma = config.nu
        self.config = config

        
        self.Q = Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.Wcinv =  Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.S =  Parameter(nn.init.normal_(torch.empty(self.nx, self.nx)))
        self.H = nn.init.eye(torch.empty((self.nx, self.nu))) # H is restriction to input space
        # register P,Q,S variables to be on specified manifolds
        geo.positive_definite(self, 'Wcinv')
        geo.positive_definite(self, 'Q')
        geo.skew(self, 'S')
        self.C = Parameter(nn.init.normal_(torch.empty(self.ny, self.nx)))

    def forward(self, u, x):

        # Building linear matrix systems 
        A = (-0.5*self.Q + self.S)@self.Wcinv
        B = sqrtm(self.Q)@self.H
        C = self.C

        dx = x@A.T + u@B.T
        y = x@C.T
        return dx, y

    def eval_(self, abs_tol = 1e-6, solver = "MOSEK"):
 
        A = (-0.5*self.Q + self.S)@self.Wcinv
        B = self.gamma*sqrtm(self.Q)@self.H
        C = self.C

        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()
        print(" Initializing H2 LMI \n")

        nx = A.shape[0]

        P = cp.Variable((nx,nx), 'P', PSD=True)

        M = A@P + P@A.T + B@B.T


        constraints = [M == 0 , P -(abs_tol)*np.eye(nx)>> 0] 
        objective = cp.Minimize(cp.trace(C@P@C.T)) # Find the H2 gain

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print("Nu = {} \n".format(prob.value))
            print("P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")


    def clone(self): # Method called by the simulator 
        copy = type(self)(self.input_dim, self.output_dim, self.config)
        copy.load_state_dict(self.state_dict())
        return copy

 
class FLNSSM(nn.Module):

    def __init__(self, input_dim : int ,  output_dim : int, 
                config):
        """
        Constructor FLNSSM where in is a generalized input ex : in = [u, y, d]:
            z^+ = Az + B(u + alpha(y, d)) + Gd
            y = Cz

        params : 
            * input_dim : size of control input
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space (z state-space is assumed to be of the size of x)
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(FLNSSM, self).__init__()

        # Set network dimensions
        self.input_dim = input_dim - config.nd
        self.state_dim = config.nx
        self.hidden_dim = config.nh
        self.output_dim = output_dim
        self.n_hid_layers= config.n_layers
        self.dist_dim = config.nd
        self.config = config

        # Activation functions
        self.actF = config.actF
        # Linear part
        if config.lin == 'alphaLNN':
            self.linmod = AlphaLNN(self.input_dim + self.dist_dim, self.output_dim, self.config)
        elif config.lin == 'hinfnn':
            self.linmod = HinfNN(self.dist_dim, self.output_dim, self.config)
            self.Bu = nn.Linear(self.input_dim, self.state_dim, bias = False)
        elif config.lin == 'h2nn':
            self.linmod = H2NN(self.input_dim + self.dist_dim, self.output_dim, self.config)
        else:
            self.linmod = NNLinear(self.input_dim + self.dist_dim, self.output_dim, self.config)

        
        # Alpha layer
        self.alpha_in_y = nn.Linear(self.output_dim, self.hidden_dim, bias = True)
        self.alpha_in_d = nn.Linear(self.dist_dim, self.hidden_dim, bias = False)

        if self.n_hid_layers>1:
            paramsNLHidA = []
            for k in range(self.n_hid_layers-1):
                tup = ('dense{}'.format(k), nn.Linear(self.hidden_dim, self.hidden_dim))
                paramsNLHidA.append(tup)
                tupact = ('actF{}'.format(k), self.actF)
                paramsNLHidA.append(tupact)
            self.alpha_hid = nn.Sequential(OrderedDict(paramsNLHidA))
        self.alpha_out = nn.Linear(self.hidden_dim, self.input_dim)



    def forward(self, input, state):
        '''
            params :
                - input = [u, d]
                - state =  state vector
        '''
        u = input[:, 0:self.input_dim]
        d = input[:, self.input_dim: self.input_dim+ self.dist_dim]

        z = state
        if hasattr(self, 'Bu'):
            _, y = self.linmod(d,z)
        
            alpha_y = self.alpha_in_y(y)
            alpha_d = self.alpha_in_d(d)
            alpha = self.actF(alpha_y + alpha_d)
            if self.n_hid_layers>1: # Only if there exists more than one hidden layer
                alpha = self.alpha_hid(alpha)
            alpha = self.alpha_out(alpha)
            
            dz, y = self.linmod(d, z) # Z_dot = Az + Gd
            dz = dz + self.Bu(u+ alpha) #z_dot  = Az + Gd + B(u+alpha)
        else:
            y = self.linmod.C(z)
            
            alpha_y = self.alpha_in_y(y)
            alpha_d = self.alpha_in_d(d)
            alpha = self.actF(alpha_y + alpha_d)
            if self.n_hid_layers>1: # Only if there exists more than one hidden layer
                alpha = self.alpha_hid(alpha)
            alpha = self.alpha_out(alpha)
            
            dz, y = self.linmod(torch.cat((u+ alpha, d),dim = 1), z)
        
        return dz, y 

    def init_weights(self, A0, B0, C0, isLinTrainable = True):
        # Initializing linear weights
        self.linmod.init_model_(A0, B0, C0, is_grad=isLinTrainable)

        # Initializing nonlinear output weights to 0
        nn.init.zeros_(self.alpha_out.weight)
        if self.alpha_out.bias is not None:
            nn.init.zeros_(self.alpha_out.bias)

        # TO-DO initialize inner weights to a specific distribution

    def load_weights(self, params_dict):
        """
            strWeigthsFile : MAT file containing wieghts matrices
        """
        # Linear part
        A0 = torch.Tensor(params_dict['A'])
        B0 = torch.Tensor(params_dict['B'])
        C0 = torch.Tensor(params_dict['C'])
        self.linmod.init_model_(A0, B0, C0, True)

        # Nonlinear part
        self.alpha_in_y.weight = nn.parameter.Parameter(torch.Tensor(params_dict['alpha_in_y']))
        self.alpha_in_y.bias = nn.parameter.Parameter(torch.Tensor(params_dict['b_in']).squeeze())
        self.alpha_in_d.weight = nn.parameter.Parameter(torch.Tensor(params_dict['alpha_in_d']))
        self.alpha_out.weight = nn.parameter.Parameter(torch.Tensor(params_dict['alpha_out']))
        self.alpha_out.bias = nn.parameter.Parameter(torch.Tensor(params_dict['b_out']).squeeze())


    def clone(self):
        copy = type(self)(self.input_dim, self.hidden_dim, self.state_dim, 
                          self.output_dim, self.n_hid_layers, self.dist_dim, self.actF)
        copy.load_state_dict(self.state_dict())
        return copy

    def eval_(self):
        self.linmod.eval_()


class LMI_networkLin(nn.Module):

    def __init__(self, X0, L0, nx, nu, nh, ny) -> None:
        super(LMI_networkLin, self).__init__()

        #self.X = nn.parameter.Parameter(X0)
        self.X = X0 # XX^T = P
        self.L = L0 #LL^T =M

        self.nx = nx 
        self.nu = nu
        self.ny = ny

    def solve_LMI(self, A,B,C, abs_tol = 1e-4, solver = "MOSEK", gamma = None):
        
        print(" Initializing HInf LMI \n")

        P = cp.Variable((self.nx, self.nx), 'P', PSD=True)
        if gamma is None:
            gamma = cp.Variable((1,1), PSD = True)
        
        
        
        nw = self.nu
        nx = self.nx
        ny = self.ny
        

        M = cp.bmat([[A.T@P + P@A, P@B, C.T], [B.T@P, -gamma *np.eye(nw), np.zeros((nw,ny))], [C, np.zeros((ny, nw)), -gamma*np.eye(ny)]])


        constraints = [M << -abs_tol*np.eye(nw+nx+ny), P -(abs_tol)*np.eye(nx)>> 0 ] 
        objective = cp.Minimize(gamma) # Find the L2 gain of the BLA

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print("Gamma = {} \n".format(prob.value))
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        self.P = torch.Tensor(P.value)
        self.gamma = torch.Tensor(gamma.value).to(dtype=torch.float32)

        # Evaluate if it closed to the boundary of the LMI
        #X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        return -M.value, P.value

    def evalLMI(self, A, B, C, P,gamma):
        lyap = torch.matmul(A.T, P) + torch.matmul(P,A)
        M11 = lyap + 1/gamma * torch.matmul(C.T,C)
        M12 =  torch.matmul(P,B)
        nw = self.nu

        M1 = torch.cat((M11, M12),1)
        M2 = torch.cat((M12.T, -gamma *torch.eye(nw)),1)
        M = torch.cat((M1,M2),0)
        return torch.Tensor(M)

    def getSystemMatrices(self):
        P = torch.matmul(self.X, self.X.T)
        P_inv = torch.linalg.inv(P)
        
        M = -torch.matmul(self.L, self.L.T)
        X11 = M[0:self.nx, 0:self.nx] # A^TP + PA
        X12 = M[0:self.nx, self.nx:self.nx+self.nu] # PB
        X31 = M[self.nx+self.nu:self.nx+self.nu+ self.ny,0: self.nx] #C
        X22 = M[self.nx:self.nx+self.nu, self.nx:self.nx+self.nu] # -gamma*I

        '''
        Pourquoi on utilise jamais le terme X22 celui qui contient gamma :
            * En utilisant cette méthode on évolue dans un espace non-contraint pour l'optimisation
            possiblement assez restreint si la BLA/initialisation a un meilleur gamma que le système linéaire d'origine
            En revanche on garantit de cette façon que la LMI est toujours faisable.
            Reste à voir si l'optimisation devient dégeu.
            Test à faire : 
                * Pour un gamma > true_sys vérifier que l'on converge bien vers le vraie système (optim non-contrainte)
                * Pour un gamma de plus en plus petit vérifier l'influence sur la MSE.
        '''
        A = torch.matmul(P_inv, X11/2 + self.S)
        B = torch.matmul(P_inv, X12)
        C = X31
        return A,B,C,P

        
    def getBLA(self, data_In, data_Out, ts = 1):
        nx = self.nx
        A,B,C, _ = findBLA(data_In, data_Out, nx, save = False, ts =ts)
        # Memorize initialisation weights
        self.A0 = A
        self.B0 = B
        self.C0 = C
        return A, B, C

    
    def init_weights(self, M, P,A):
        '''
            M : Semidefinite matrix POSITIVE for Hinf LMI 
            P : Lyapunov certificate
        
        '''
        Q = -torch.Tensor(M)
        P = torch.Tensor(P)
        L = torch.linalg.cholesky(torch.tensor(M)).to(dtype=torch.float32)
        X = torch.linalg.cholesky(torch.tensor(P)).to(dtype=torch.float32)
        self.X = nn.parameter.Parameter(torch.Tensor(X))
        self.L = nn.parameter.Parameter(torch.Tensor(L))
        A_forward = torch.matmul(torch.linalg.inv(P), Q[0:2, 0:2])/2
        self.S = torch.matmul(torch.Tensor(P), A - A_forward)


    def set_gamma(self, gamma):
        M = torch.matmul(self.L, self.L.T)
        M[2,2] = gamma
        M[3,3] = gamma

    def forward(self, u, x0):

        # A mettre dans une layer custom qui prend P en sortie et renvoie A,B,C
        P = torch.matmul(self.X, self.X.T)
        P_inv = torch.linalg.inv(P)
        
        M = -torch.matmul(self.L, self.L.T)
        X11 = M[0:self.nx, 0:self.nx] # A^TP + PA
        X12 = M[0:self.nx, self.nx:self.nx+self.nu] # PB
        X31 = M[self.nx+self.nu:self.nx+self.nu+ self.ny,0: self.nx] #C
        X22 = M[self.nx:self.nx+self.nu, self.nx:self.nx+self.nu] # -gamma*I

        '''
        Pourquoi on utilise jamais le terme X22 celui qui contient gamma :
            * En utilisant cette méthode on évolue dans un espace non-contraint pour l'optimisation
            possiblement assez restreint si la BLA/initialisation a un meilleur gamma que le système linéaire d'origine
            En revanche on garantit de cette façon que la LMI est toujours faisable.
            Reste à voir si l'optimisation devient dégeu.
            Test à faire : 
                * Pour un gamma > true_sys vérifier que l'on converge bien vers le vraie système (optim non-contrainte)
                * Pour un gamma de plus en plus petit vérifier l'influence sur la MSE.
        '''
        A = torch.matmul(P_inv, X11/2 + self.S)
        B = torch.matmul(P_inv, X12)
        C = X31

        # Fin de la custom layer

        y = torch.matmul(x0, C.T)# torch.bmm(C,x0)
        dx = torch.matmul(x0,A.T) + torch.matmul(u, B.T)
        return dx,y

class GRNSSM(nn.Module):
    def __init__(self, input_dim : int , hidden_dim : int , state_dim : int,  output_dim : int, 
                n_hid_layers : int, actF = nn.Tanh()):
        """
        Constructor grNSSM u is a generalized input ex : [control, distrubance]:
            x^+ = Ax + Bu + f(x,u)
            y = Cx + h(x)

        params : 
            * input_dim : size of input layer
            * hidden_dim : size of hidden layers
            * state_dim : size of the state-space
            * n_hid_layers : number of hidden layers
            * output_dim : size of the output layer
            * actF : activation function for nonlienar residuals
        """
        super(GRNSSM, self).__init__()

        # Set network dimensions
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_hid_layers= n_hid_layers
        # Activation functions
        self.actF = actF

        # Linear part
        self.linmod = NNLinear(self.input_dim, self.state_dim, self.output_dim)
        # Nonlinear part for the state f(x,u) = W_l \sigma(W_{l-1}...)
        # Input Layer
        self.Wfx = nn.Linear(self.state_dim, self.hidden_dim, bias = False)
        self.Wfu = nn.Linear(self.input_dim, self.hidden_dim, bias = True)
        self.actFinF = actF
        if self.n_hid_layers>1:
            paramsNLHidF = []
            for k in range(n_hid_layers-1):
                tup = ('dense{}'.format(k), nn.Linear(hidden_dim, hidden_dim, bias = False))
                paramsNLHidF.append(tup)
                tupact = ('actF{}'.format(k), actF)
                paramsNLHidF.append(tupact)
            self.f_hid = nn.Sequential(OrderedDict(paramsNLHidF))
        self.Wf = nn.Linear(self.hidden_dim, self.state_dim, bias = True)

        '''
        # Nonlinear part for the output y = cx + h(x)
        self.Whx = nn.Linear(self.state_dim, self.hidden_dim, bias = False)
        self.actFinH = actF
        if self.n_hid_layers>1:
            paramsNLHidH = []
            for k in range(n_hid_layers-1):
                tup = ('dense{}'.format(k), nn.Linear(hidden_dim, hidden_dim, bias= False))
                paramsNLHidH.append(tup)
                tupact = ('actF{}'.format(k), actF)
                paramsNLHidH.append(tupact)
            self.h_hid = nn.Sequential(OrderedDict(paramsNLHidH))
        self.Wh = nn.Linear(self.hidden_dim, self.output_dim, bias = False)
        '''

    def forward(self, u, x):
        # Forward pass -- prediction of the output at time k : y_k

        x_lin, y_lin = self.linmod(u,x) # Linear part

        # Nonlinear part fx
        fx = self.Wfx(x) + self.Wfu(u)
        fx = self.actFinF(fx)
        if self.n_hid_layers>1:
            fx = self.f_hid(fx)
        fx = self.Wf(fx)

        dx = x_lin + fx

        '''
        # Nonlinear part hx
        hx = self.Whx(x)
        hx = self.actFinH(hx)

        if self.n_hid_layers>1:
            hx = self.h_hid(hx)
        hx = self.Wh(hx)
        '''
        y = y_lin  # + hx
        
        return dx, y

    def init_weights(self, A0, B0, C0, isLinTrainable = True):
        # Initializing linear weights
        self.linmod.init_model_(A0, B0, C0, is_grad=isLinTrainable)

        # Initializing nonlinear output weights to 0
        nn.init.zeros_(self.Wf.weight)
        if self.Wf.bias is not None:
            nn.init.zeros_(self.Wf.bias)
        #nn.init.zeros_(self.Wh.weight)

        # TO-DO initialize inner weights to a specific distribution
    
    def load_weights(self, params_dict):
        """
            strWeigthsFile : MAT file containing wieghts matrices
        """
        # Linear part
        A0 = torch.Tensor(params_dict['A'])
        B0 = torch.Tensor(params_dict['B'])
        C0 = torch.Tensor(params_dict['C'])
        self.linmod.init_model_(A0, B0, C0, True)

        # Nonlinear part
        self.Wfu.weight = nn.parameter.Parameter(torch.Tensor(params_dict['Wfu']))
        self.Wfu.bias = nn.parameter.Parameter(torch.Tensor(params_dict['bi']).squeeze())
        self.Wfx.weight = nn.parameter.Parameter(torch.Tensor(params_dict['Wfx']))
        self.Wf.weight = nn.parameter.Parameter(torch.Tensor(params_dict['Wf']))
        self.Wf.bias = nn.parameter.Parameter(torch.Tensor(params_dict['bo']).squeeze())

    def clone(self): # Method called by the simulator 
        copy = type(self)(self.input_dim, self.hidden_dim, self.state_dim, 
                          self.output_dim, self.n_hid_layers, self.actF)
        copy.load_state_dict(self.state_dict())
        return copy

import torch
from torch import nn
from torch.nn import Parameter
from torch import Tensor

import cvxpy as cp
import numpy as np
import scipy as sp

import utils 
# A modifier pour appeler matlab SIPPY

class RobustRnn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, res_size, method="Neuron",
                 nl=None, nBatches=1, learn_init_state=True, alpha=0, beta=1, supply_rate="stable"):
        super(RobustRnn, self).__init__()

        self.type = "iqcRNN"

        self.nx = hidden_size
        self.nu = input_size
        self.ny = output_size
        self.nw = res_size

        self.nBatches = nBatches
        # self.h0 = torch.nn.Parameter(torch.rand(nBatches, hidden_size))
        # self.h0 = torch.nn.Parameter(torch.zeros(nBatches, hidden_size)) if learn_init_state else torch.zeros(nBatches, hidden_size)
        self.h0 = torch.nn.Parameter(torch.zeros(nBatches, hidden_size))
        self.h0.requires_grad = learn_init_state

        self.criterion = torch.nn.MSELoss()

        #  nonlinearity
        self.nl = torch.nn.ReLU() if nl is None else nl

        # metric
        E0 = torch.eye(hidden_size)
        P0 = torch.eye(hidden_size)

        # Metric
        self.E = nn.Parameter(E0)
        self.P = nn.Parameter(P0)

        # dynamics
        self.F = nn.Linear(hidden_size, hidden_size)
        self.F.bias.data = torch.zeros_like(self.F.bias.data)

        self.B1 = nn.Linear(res_size, hidden_size, bias=False)
        self.B2 = nn.Linear(input_size, hidden_size, bias=False)

        # output for v
        self.C2tild = Parameter(torch.randn((self.nw, self.nx)) / np.sqrt(self.nx))
        self.bv = Parameter(torch.rand(self.nw) - 0.5)
        self.Dtild = Parameter(torch.randn((self.nw, self.nu)) / np.sqrt(self.nu))

        # y ouputs for model
        self.C1 = nn.Linear(hidden_size, output_size, bias=False)
        self.D11 = nn.Linear(res_size, output_size, bias=False)
        self.D12 = nn.Linear(input_size, output_size, bias=False)
        self.by = Parameter(0*(torch.rand(self.ny) - 0.5))

        # Create parameters for the iqc multipliers
        self.alpha = alpha
        self.beta = beta
        self.method = method
        if method == "Layer":
            self.IQC_multipliers = torch.nn.Parameter(torch.zeros(1))

        elif method == "Neuron":
            self.IQC_multipliers = torch.nn.Parameter(1E-4*torch.ones(self.nw))

        elif method == "Network":
            # Same number of variables as in lower Triangular matrix?
            self.IQC_multipliers = torch.nn.Parameter(torch.zeros(((self.nw + 1) * self.nw) // 2))
        else:
            print("Do Nothing")

        self.supply_rate = supply_rate

    def forward(self, u, h0=None, c0=None):

        inputs = u.permute(0, 2, 1)
        seq_len = inputs.size(1)

        #  Initial state
        b = inputs.size(0)
        if h0 is None:
            ht = torch.zeros(b, self.nx)
        else:
            ht = h0

        # First calculate the inverse for E for each layer
        Einv = self.E.inverse()

        # Tensor to store the states in
        states = torch.zeros(b, seq_len, self.nx)
        yest = torch.zeros(b, seq_len, self.ny)
        index = 1
        states[:, 0, :] = ht

        # Construct C2 from Ctild.
        Cv = torch.diag(1 / self.IQC_multipliers) @ self.C2tild
        Dv = torch.diag(1 / self.IQC_multipliers) @ self.Dtild

        for tt in range(seq_len - 1):
            # Update state
            vt = ht @ Cv.T + inputs[:, tt, :] @ Dv.T + self.bv[None, :]

            wt = self.nl(vt)
            eh = self.F(ht) + self.B1(wt) + self.B2(inputs[:, tt, :])
            ht = eh @ Einv.T

            # Store state
            states[:, index, :] = ht
            index += 1

        # Output function
        W = self.nl(states @ Cv.T + inputs @ Dv.T + self.bv)
        yest = self.C1(states) + self.D11(W) + self.D12(inputs) + self.by

        return yest.permute(0, 2, 1)

    def construct_T(self):
        r'Returns a conic combination of IQC multipliers coupling different sets of neurons together.\
        Methods are listed in order of most scalable to most accurate.'
        # Lip SDP neuron
        if self.method == "Layer":
            # Tv = torch.cat([torch.ones(ni) * self.IQC_multipliers[idx] for (idx, ni) in enumerate(self.N[:-1])])
            T = torch.eye(self.nx) * self.IQC_multipliers

        elif self.method == "Neuron":
            T = torch.diag(self.IQC_multipliers)

        elif self.method == "Network":
            # return the (ii,jj)'th multiplier from the mulitplier vector
            get_multi = lambda ii, jj: self.IQC_multipliers[ii * (ii + 1) // 2 + jj]

            # Get the structured matrix in T
            Id = torch.eye(self.nx)
            e = lambda ii: Id[:, ii:ii + 1]
            Tij = lambda ii, jj: e(ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj) for ii in range(0, self.nx) for jj in range(0, ii + 1))

        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        return T

    #  Barrier function to ensure multipliers are positive
    def multiplier_barrier(self):
        return self.IQC_multipliers

    def lipschitz_LMI(self, gamma=10.0, eps=1E-4):
        def l2gb_lmi():
            T = self.construct_T()
            M = utils.bmat([[-2 * self.alpha * self.beta * T, (self.alpha + self.beta) * T],
                            [(self.alpha + self.beta) * T, - 2 * T]])

            L_sq = gamma ** 2

            # Construct LMIs
            P = self.P
            E = self.E
            F = self.F.weight
            B1 = self.B1.weight
            B2 = self.B2.weight

            # y output
            C1 = self.C1.weight
            D11 = self.D11.weight
            D12 = self.D12.weight

            # v output
            Ctild = self.C2tild
            Dtild = self.Dtild

            zxu = torch.zeros((self.nx, self.nu))
            L_sq = gamma ** 2

            # Mat11 = utils.bmat([E + E.T - P, z1, L_sq * np.eye(self.nu)]) - Gamma_v.T @ M @ Gamma_v
            Mat11 = utils.bmat([[E + E.T - P, -self.beta * Ctild.T, zxu],
                                [-self.beta * Ctild, 2 * T, -self.beta * Dtild],
                                [zxu.T, -self.beta * Dtild.T, L_sq * torch.eye(self.nu)]])

            Mat21 = utils.bmat([[F, B1, B2], [C1, D11, D12]])
            Mat22 = utils.bmat([[P, torch.zeros((self.nx, self.ny))],
                                [torch.zeros((self.ny, self.nx)), torch.eye(self.ny)]])

            Mat = utils.bmat([[Mat11, Mat21.T],
                              [Mat21, Mat22]])

            return 0.5 * (Mat + Mat.T)

        def E_pd():
            return 0.5 * (self.E + self.E.T) - eps * torch.eye(self.nx)

        def P_pd():
            return 0.5 * (self.P + self.P.T) - eps * torch.eye(self.nx)

        return [l2gb_lmi, E_pd, P_pd]

    def initialize_lipschitz_LMI(self, gamma=10.0, eps=1E-4, init_var=1.5, solver="SCS"):
        solver_tol = 1E-4
        print("Initializing Lipschitz LMI ...")
        # Lip SDP multiplier
        if self.method == "Layer":
            multis = cp.Variable((1), 'lambdas', nonneg=True)
            T = multis * np.eye(self.nx)

        elif self.method == "Neuron":
            multis = cp.Variable((self.nx), 'lambdas', nonneg=True)
            T = cp.diag(multis)

        elif self.method == "Network":
            # Variables can be mapped to tril matrix => (n+1) x n // 2 variables
            multis = cp.Variable((self.nx + 1) * self.nx // 2, 'lambdas', nonneg=True)

            # return the (ii,jj)'th multiplier
            get_multi = lambda ii, jj: multis[(ii * (ii + 1)) // 2 + jj]

            # Get the structured matrix in T
            Id = np.eye(self.nx)
            e = lambda ii: Id[:, ii:ii + 1]
            Tij = lambda ii, jj: e(ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj) for ii in range(0, self.nx) for jj in range(0, ii + 1))
        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        # square of L2 gain
        # L_sq = cp.Variable((1, 1), "rho")
        L_sq = gamma ** 2

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        Bu = cp.Variable((self.nx, self.nu), 'Bu')
        C = cp.Variable((self.ny, self.nx), 'C')
        Dw = cp.Variable((self.ny, self.nx), 'Dw')
        Du = cp.Variable((self.ny, self.nu), 'Du')

        Bw = cp.Variable((self.nx, self.nw), 'Bw')

        Cv = np.random.normal(0, init_var / np.sqrt(self.nx), (self.nw, self.nx))

        Gamma_1 = sp.linalg.block_diag(Cv, np.eye(self.nw))
        Gamma_v = np.concatenate([Gamma_1, np.zeros((2 * self.nw, self.nu))], axis=1)

        M = cp.bmat([[-2 * self.alpha * self.beta * T, (self.alpha + self.beta) * T], [(self.alpha + self.beta) * T.T, - 2 * T]])

        # Construct final LMI.
        zxw = np.zeros((self.nx, self.nw))
        zxu = np.zeros((self.nx, self.nu))
        zwu = np.zeros((self.nw, self.nu))
        zww = np.zeros((self.nw, self.nw))

        # Mat11 = utils.bmat([E + E.T - P, z1, L_sq * np.eye(self.nu)]) - Gamma_v.T @ M @ Gamma_v
        Mat11 = cp.bmat([[E + E.T - P, zxw, zxu], [zxw.T, zww, zwu],
                         [zxu.T, zwu.T, L_sq * np.eye(self.nu)]]) - Gamma_v.T @ M @ Gamma_v

        Mat21 = cp.bmat([[F, Bw, Bu], [C, Dw, Du]])
        Mat22 = cp.bmat([[P, np.zeros((self.nx, self.ny))], [np.zeros((self.ny, self.nx)), np.eye(self.ny)]])

        Mat = cp.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> solver_tol * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        # Just find a feasible point
        A = np.random.normal(0, init_var / np.sqrt(self.nx), (self.nx, self.nx))

        # Just find a feasible point
        objective = cp.Minimize(cp.norm(E @ A - Bw))

        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        elif solver == "SCS":
            prob.solve(solver=cp.SCS)
        else:
            print("Select valid sovler")

        print("Initilization Status: ", prob.status)

        # self.L_squared = torch.nn.Parameter(torch.Tensor(L_sq.value))
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.Bw.weight = Parameter(Tensor(Bw.value))
        self.Bu.weight = Parameter(Tensor(Bu.value))
        self.C.weight = Parameter(Tensor(C.value))
        self.Dw.weight = Parameter(Tensor(Dw.value))
        self.Du.weight = Parameter(Tensor(Du.value))
        self.Cv.weight = Parameter(Tensor(Cv))

    def init_lipschitz_ss(self, loader, gamma=10.0, eps=1E-4, init_var=1.2, solver="SCS"):

        print("RUNNING N4SID for intialization of A, Bu, C, Du")

        data = [(u, y) for (idx, u, y) in loader]
        U = data[0][0][0].numpy()
        Y = data[0][1][0].numpy()
        sys_id = sippy.system_identification(Y, U, 'N4SID', SS_fixed_order=self.nx)

        Ass = sys_id.A
        Bss = sys_id.B
        Css = sys_id.C
        Dss = sys_id.D

        # Calculate the trajectory.
        Xtraj = np.zeros((self.nx, Y.shape[1]))
        for tt in range(1, Y.shape[1]):
            Xtraj[:, tt:tt+1] = Ass @ Xtraj[:, tt - 1:tt] + Bss @ U[:, tt - 1:tt]

        # Sample points, calulate next state
        samples = 5000
        xtild = 3 * np.random.randn(self.nx, samples)
        utild = 3 * np.random.randn(self.nu, samples)
        xtild_next = Ass @ xtild + Bss @ utild

        print("Initializing using LREE")

        solver_tol = 1E-3
        print("Initializing stable LMI ...")

        # Lip SDP multiplier
        if self.method == "Layer":
            multis = cp.Variable((1, 1), 'lambdas', nonneg=True)
            T = multis * np.eye(self.nw)

        elif self.method == "Neuron":
            multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
            T = cp.diag(multis)

        elif self.method == "Network":
            print('YOU ARE USING THE NETWORK IQC MULTIPLIER. THIS DOES NOT WORK. PLEASE CHANGE TO NEURON OR LAYER')
            # Variables can be mapped to tril matrix => (n+1) x n // 2 variables
            multis = cp.Variable((self.nx + 1) * self.nx // 2, 'lambdas', nonneg=True)

            # Used for mapping vector to tril matrix
            indices = list(range((self.nx + 1) * self.nx // 2))
            Tril_Indices = np.zeros((self.nx, self.nx), dtype=int)
            Tril_Indices[np.tril_indices(self.nx)] = indices

            # return the (ii,jj)'th multiplier
            get_multi = lambda ii, jj: multis[Tril_Indices[ii, jj]]

            # Get the structured matrix in T
            Id = np.eye(self.nx)
            e = lambda ii: Id[:, ii:ii + 1]
            Tij = lambda ii, jj: e(ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj) for ii in range(self.nx) for jj in range(ii + 1))
        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        B1 = cp.Variable((self.nx, self.nw), 'Bw')
        B2 = cp.Variable((self.nx, self.nu), 'Bu')

        # Output matrices
        C1 = cp.Variable((self.ny, self.nx), 'C1')
        D11 = cp.Variable((self.ny, self.nw), 'D11')
        D12 = cp.Variable((self.ny, self.nu), 'D12')


        # Randomly initialize C2
        C2 = np.random.normal(0, init_var / np.sqrt(self.nw), (self.nw, self.nx))
        D22 = np.random.normal(0, init_var / np.sqrt(self.nw), (self.nw, self.nu))

        Ctild = T @ C2
        Dtild = T @ D22

        # lmi for dl2 gain bound.
        zxu = np.zeros((self.nx, self.nu))
        L_sq = gamma ** 2

        # Mat11 = utils.bmat([E + E.T - P, z1, L_sq * np.eye(self.nu)]) - Gamma_v.T @ M @ Gamma_v
        Mat11 = cp.bmat([[E + E.T - P, -self.beta * Ctild.T, zxu],
                         [-self.beta * Ctild, 2 * T, -self.beta * Dtild],
                         [zxu.T, -self.beta * Dtild.T, L_sq * np.eye(self.nu)]])

        Mat21 = cp.bmat([[F, B1, B2], [C1, D11, D12]])
        Mat22 = cp.bmat([[P, np.zeros((self.nx, self.ny))],
                         [np.zeros((self.ny, self.nx)), np.eye(self.ny)]])

        Mat = cp.bmat([[Mat11, Mat21.T],
                       [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> solver_tol * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        # Find the closest l2 gain bounded model
        bv = self.bv.detach().numpy()[:, None]

        if type(self.nl) is torch.nn.ReLU:
            wt = np.maximum(C2 @ xtild + D22 @ utild + bv, 0)
            wtraj = np.maximum(C2 @ Xtraj + D22 @ U + bv, 0)
        else:
            wt = np.tanh(C2 @ xtild + D22 @ utild + bv)
            wtraj = np.tanh(C2 @ Xtraj + D22 @ U + bv, 0)

        zt = np.concatenate([xtild_next, xtild, wt, utild], 0)

        EFBB = cp.bmat([[E, -F, -B1, -B2]])

        # empirical covariance matrix PHI
        Phi = zt @ zt.T
        R = cp.Variable((2 * self.nx + self.nw + self.nu, 2 * self.nx + self.nw + self.nu))
        Q = cp.bmat([[R, EFBB.T], [EFBB, E + E.T - np.eye(self.nx)]])

        # Add additional term for output errors

        eta = Y - C1 @ Xtraj - D11 @ wtraj - D12 @ U

        objective = cp.Minimize(cp.trace(R@Phi) + cp.norm(eta, p="fro")**2)
        constraints.append(Q >> 0)

        prob = cp.Problem(objective, constraints)
        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve(solver=cp.SCS)

        print("Initilization Status: ", prob.status)

        # Assign results to model
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.B1.weight = Parameter(Tensor(B1.value))
        self.B2.weight = Parameter(Tensor(B2.value))

        # Output mappings
        self.C1.weight = Parameter(Tensor(C1.value))
        self.D12.weight = Parameter(Tensor(D12.value))
        self.D11.weight = Parameter(Tensor(D11.value))
        self.by = Parameter(Tensor([0.0]))

        # Store Ctild and Dtild, C2 and D22 are extracted from
        #  T^{-1} \tilde{C} and T^{-1} \tilde{Dtild}
        self.C2tild = Parameter(Tensor(Ctild.value))
        self.Dtild = Parameter(Tensor(Dtild.value))

        print("Init Complete")

    def stable_LMI(self, eps=1E-4):
        def stable_lmi():
            T = self.construct_T()
            M = utils.bmat([[-2 * self.alpha * self.beta * T, (self.alpha + self.beta) * T],
                            [(self.alpha + self.beta) * T, - 2 * T]])

            # Construct LMIs
            P = self.P
            E = self.E
            F = self.F.weight
            B1 = self.B1.weight
            # Cv = self.Cv.weight

            S = utils.bmat([[torch.zeros(self.nx, self.nx), self.C2tild.T],
                            [self.C2tild, -2 * T]])

            # Construct final LMI.
            Mat11 = utils.block_diag([E + E.T - P, torch.zeros(self.nw, self.nw)]) - S
            Mat21 = utils.bmat([[F, B1]])
            Mat22 = P

            Mat = utils.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])
            return 0.5 * (Mat + Mat.T)

        def E_pd():
            return 0.5 * (self.E + self.E.T) - eps * torch.eye(self.nx)

        def P_pd():
            return 0.5 * (self.P + self.P.T) - eps * torch.eye(self.nx)

        return [stable_lmi, E_pd, P_pd]

    def initialize_stable_LMI(self, eps=1E-4, init_var=1.5, obj='B', solver="SCS"):

        solver_tol = 1E-4
        print("Initializing stable LMI ...")
        # Lip SDP multiplier
        if self.method == "Layer":
            multis = cp.Variable((1, 1), 'lambdas', nonneg=True)
            T = multis * np.eye(self.nw)

        elif self.method == "Neuron":
            multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
            T = cp.diag(multis)

        elif self.method == "Network":
            print('YOU ARE USING THE NETWORK IQC MULTIPLIER. THIS DOES NOT WORK. PLEASE CHANGE TO NEURON OR LAYER')
            # Variables can be mapped to tril matrix => (n+1) x n // 2 variables
            multis = cp.Variable((self.nx + 1) * self.nx // 2, 'lambdas', nonneg=True)

            # Used for mapping vector to tril matrix
            indices = list(range((self.nx + 1) * self.nx // 2))
            Tril_Indices = np.zeros((self.nx, self.nx), dtype=int)
            Tril_Indices[np.tril_indices(self.nx)] = indices

            # return the (ii,jj)'th multiplier
            get_multi = lambda ii, jj: multis[Tril_Indices[ii, jj]]

            # Get the structured matrix in T
            Id = np.eye(self.nx)
            e = lambda ii: Id[:, ii:ii + 1]
            Tij = lambda ii, jj: e(ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj) for ii in range(self.nx) for jj in range(ii + 1))
        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        Bw = cp.Variable((self.nx, self.nw), 'Bw')

        Cv = np.random.normal(0, init_var / np.sqrt(self.nx), (self.nw, self.nx))
        Gamma_v = sp.linalg.block_diag(Cv, np.eye(self.nw))
        M = cp.bmat([[-2 * self.alpha * self.beta * T, (self.alpha + self.beta) * T], [(self.alpha + self.beta) * T, - 2 * T]])

        # Construct final LMI.
        z1 = np.zeros((self.nx, self.nw))
        z2 = np.zeros((self.nw, self.nw))

        Mat11 = cp.bmat([[E + E.T - P, z1], [z1.T, z2]]) - Gamma_v.T @ M @ Gamma_v

        Mat21 = cp.bmat([[F, Bw]])
        Mat22 = P

        Mat = cp.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> (eps + solver_tol) * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        A = np.random.normal(0, init_var / np.sqrt(self.nx), (self.nx, self.nw))
        # Ass = np.eye(self.nx)

        # ensure wide distribution of eigenvalues for Bw
        objective = cp.Minimize(cp.norm(E @ A - Bw))

        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve(solver=cp.SCS)

        print("Initilization Status: ", prob.status)

        # self.L_squared = torch.nn.Parameter(torch.Tensor(L_sq.value))
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.Bw.weight = Parameter(Tensor(Bw.value))
        self.Bu.weight = Parameter(Tensor(0.1 * self.Bu.weight.data))
        self.C.weight = Parameter(Tensor(0.1 * self.C.weight.data))
        self.Du.weight = Parameter(Tensor(0.0 * self.Du.weight.data))

        self.Ctild = Parameter(Tensor(Cv))

        print("Init Complete")

    def init_stable_ss(self, loader, eps=1E-4, init_var=1.2, solver="SCS"):

        print("RUNNING N4SID for intialization of A, Bu, C, Du")

        data = [(u, y) for (idx, u, y) in loader]
        U = data[0][0][0].numpy()
        Y = data[0][1][0].numpy()
        sys_id = sippy.system_identification(Y, U, 'N4SID', SS_fixed_order=self.nx)

        Ass = sys_id.A
        Bss = sys_id.B
        Css = sys_id.C
        Dss = sys_id.D

        # Sample points, calulate next state
        samples = 5000
        xtild = 3 * np.random.randn(self.nx, samples)
        utild = 3 * np.random.randn(self.nu, samples)
        xtild_next = Ass @ xtild + Bss @ utild

        print("Initializing using LREE")

        solver_tol = 1E-4
        print("Initializing stable LMI ...")

        # Lip SDP multiplier
        if self.method == "Layer":
            multis = cp.Variable((1, 1), 'lambdas', nonneg=True)
            T = multis * np.eye(self.nw)

        elif self.method == "Neuron":
            multis = cp.Variable((self.nw), 'lambdas', nonneg=True)
            T = cp.diag(multis)

        elif self.method == "Network":
            print('YOU ARE USING THE NETWORK IQC MULTIPLIER. THIS DOES NOT WORK. PLEASE CHANGE TO NEURON OR LAYER')
            # Variables can be mapped to tril matrix => (n+1) x n // 2 variables
            multis = cp.Variable((self.nx + 1) * self.nx // 2, 'lambdas', nonneg=True)

            # Used for mapping vector to tril matrix
            indices = list(range((self.nx + 1) * self.nx // 2))
            Tril_Indices = np.zeros((self.nx, self.nx), dtype=int)
            Tril_Indices[np.tril_indices(self.nx)] = indices

            # return the (ii,jj)'th multiplier
            get_multi = lambda ii, jj: multis[Tril_Indices[ii, jj]]

            # Get the structured matrix in T
            Id = np.eye(self.nx)
            e = lambda ii: Id[:, ii:ii + 1]
            Tij = lambda ii, jj: e(ii) @ e(ii).T if ii == jj else (e(ii) - e(jj)) @ (e(ii) - e(jj)).T

            # Construct the full conic comibation of IQC's
            T = sum(Tij(ii, jj) * get_multi(ii, jj) for ii in range(self.nx) for jj in range(ii + 1))
        else:
            print("Invalid method selected. Try Neuron, Layer or Network")

        # Construct LMIs
        P = cp.Variable((self.nx, self.nx), 'P', symmetric=True)
        E = cp.Variable((self.nx, self.nx), 'E')
        F = cp.Variable((self.nx, self.nx), 'F')
        B1 = cp.Variable((self.nx, self.nw), 'Bw')
        B2 = cp.Variable((self.nx, self.nu), 'Bu')

        # Randomly initialize C2
        C2 = np.random.normal(0, init_var / np.sqrt(self.nw), (self.nw, self.nx))
        D22 = np.random.normal(0, init_var / np.sqrt(self.nw), (self.nw, self.nu))

        Ctild = T @ C2
        Dtild = T @ D22

        # Stability LMI
        S = cp.bmat([[np.zeros((self.nx, self.nx)), Ctild.T], [Ctild, -2 * T]])
        z1 = np.zeros((self.nx, self.nw))
        z2 = np.zeros((self.nw, self.nw))
        Mat11 = cp.bmat([[E + E.T - P, z1], [z1.T, z2]]) - S
        Mat21 = cp.bmat([[F, B1]])
        Mat22 = P
        Mat = cp.bmat([[Mat11, Mat21.T], [Mat21, Mat22]])

        # epsilon ensures strict feasability
        constraints = [Mat >> (eps + solver_tol) * np.eye(Mat.shape[0]),
                       P >> (eps + solver_tol) * np.eye(self.nx),
                       E + E.T >> (eps + solver_tol) * np.eye(self.nx),
                       multis >= 1E-6]

        # ensure wide distribution of eigenvalues for Bw
        bv = self.bv.detach().numpy()[:, None]

        if type(self.nl) is torch.nn.ReLU:
            wt = np.maximum(C2 @ xtild + D22 @ utild + bv, 0)
        else:
            wt = np.tanh(C2 @ xtild + D22 @ utild + bv)

        zt = np.concatenate([xtild_next, xtild, wt, utild], 0)

        EFBB = cp.bmat([[E, -F, -B1, -B2]])

        # empirical covariance matrix PHI
        Phi = zt @ zt.T
        R = cp.Variable((2*self.nx + self.nw + self.nu, 2*self.nx + self.nw + self.nu))
        Q = cp.bmat([[R, EFBB.T], [EFBB, E + E.T - np.eye(self.nx)]])

        objective = cp.Minimize(cp.trace(R@Phi))
        constraints.append(Q >> 0)

        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve(solver=cp.SCS)
        print("Initilization Status: ", prob.status)

        # Solve for output mapping from (W, X, U) -> Y
        # using linear least squares
        X = np.zeros((self.nx, U.shape[1]))
        X[:, 0:1] = sys_id.x0

        Einv = np.linalg.inv(E.value)
        Ahat = Einv @ F.value
        Bwhat = Einv @ B1.value
        Buhat = Einv @ B2.value
        for t in range(1, U.shape[1]):
            w = np.maximum(C2 @ X[:, t-1:t] + D22 @ U[:, t-1:t] + bv, 0)
            X[:, t:t+1] = Ahat @ X[:, t-1:t] + Bwhat @ w + Buhat @ U[:, t-1:t]

        if type(self.nl) is torch.nn.ReLU:
            W = np.maximum(C2 @ X + D22 @ U + bv, 0)
        else:
            W = np.tanh(C2 @ X + D22 @ U + bv, 0)

        Z = np.concatenate([X, W, U], 0)
        output_mats = Y @ np.linalg.pinv(Z)

        C1 = output_mats[:, :self.nx]
        D11 = output_mats[:, self.nx:self.nx+self.nw]
        D12 = output_mats[:, self.nx+self.nw:]

        # Assign results to model
        self.IQC_multipliers = Parameter(Tensor(multis.value))
        self.E = Parameter(Tensor(E.value))
        self.P = Parameter(Tensor(P.value))
        self.F.weight = Parameter(Tensor(F.value))
        self.B1.weight = Parameter(Tensor(B1.value))
        self.B2.weight = Parameter(Tensor(B2.value))

        # Output mappings
        self.C1.weight = Parameter(Tensor(C1))
        self.D12.weight = Parameter(Tensor(D12))
        self.D11.weight = Parameter(Tensor(D11))
        self.by = Parameter(Tensor([0.0]))

        # Store Ctild, C2 is extracted from T^{-1} \tilde{C}
        self.C2tild = Parameter(Tensor(Ctild.value))
        self.Dtild = Parameter(Tensor(Dtild.value))

        print("Init Complete")

    def clone(self):
        copy = type(self)(self.nu, self.nx, self.ny, self.nw, nBatches=self.nBatches, nl=self.nl, method=self.method)
        copy.load_state_dict(self.state_dict())

        return copy

    def flatten_params(self):
        r"""Return paramter vector as a vector x."""
        views = []
        for p in self.parameters():
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.reshape(-1)
            views.append(view)
        return torch.cat(views, 0)

    def write_flat_params(self, x):
        r""" Writes vector x to model parameters.."""
        index = 0
        theta = torch.Tensor(x)
        for p in self.parameters():
            p.data = theta[index:index + p.numel()].view_as(p.data)
            index = index + p.numel()

    def flatten_grad(self):
        r""" Returns vector of all gradients."""
        views = []
        for p in self.parameters():
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()