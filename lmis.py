import torch.nn as nn
from torch.nn.parameter import Parameter
import geotorch as geo
import cvxpy as cp
import numpy as np
import torch
from torch.nn import Module
from torch import Tensor
from models import NNLinear
from utils import find_module

'''
    Utility functions
'''

def block_diag(arr_list):
    '''create a block diagonal matrix from a list of cvxpy matrices'''

    # rows and cols of block diagonal matrix
    m = np.sum([arr.shape[0] for arr in arr_list])
    n = np.sum([arr.shape[1] for arr in arr_list])

    # loop to create the list for the bmat function
    block_list = []  # list for bmat function
    ind = np.array([0,0])
    for arr in arr_list:
        # index of the end of arr in the block diagonal matrix
        ind += arr.shape

        # list of one row of blocks
        horz_list = [arr]

        # block of zeros to the left of arr
        zblock_l = np.zeros((arr.shape[0], ind[1]-arr.shape[1]))
        if zblock_l.shape[1] > 0:
            horz_list.insert(0, zblock_l)

        # block of zeros to the right of arr
        zblock_r = np.zeros((arr.shape[0], n-ind[1]))
        if zblock_r.shape[1] > 0:
            horz_list.append(zblock_r)

        block_list.append(horz_list)

    B = cp.bmat(block_list)

    return B

'''
    Continuous time LMI versions
'''

class LMI_Lyap(Module):
    '''
        A base class for all LMI based on Lyapunov stability for a continuous time linear system
        M = A^TP + PA 
        To create a new LMI make it child of this class and override init_lmi and forward methods.
        If more than 1 matrix for Lyapunov certificate add a method like symP_
    '''
    def __init__(self, A : torch.Tensor, epsilon : float = 1e-6) -> None:
        super(LMI_Lyap, self).__init__()

        self.P = Parameter(torch.empty_like(A))
        #self.P = Parameter(torch.empty_like(A))
        geo.positive_definite(self, "P")
        M, P = self.init_lmi(A,epsilon = epsilon)
        self.P = Parameter(P.requires_grad_(True))
        self.A = A
        self.nm = M.shape[0]

    def init_lmi(self, A : torch.Tensor, epsilon = 1e-6, solver = "MOSEK"):
        A = A.detach().numpy()
        print(" Initializing Lyapunov LMI \n")
        nx = A.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)

        M = A.T@P + P@A


        constraints = [M << -np.eye(nx)*epsilon, P -(epsilon)*np.eye(nx)>> 0] 
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

        P = torch.Tensor(P.value).to(dtype=torch.float32)
        M = torch.Tensor(M.value).to(dtype=torch.float32)
        return M, P

    def symP_(self):
        self.P = Parameter(0.5*(self.P + self.P.T))
    
    def forward(self):
        self.symP_()
        eig_val, _ = torch.linalg.eig(self.P)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("P is not SDP")

        M = -(self.A.T@self.P + self.P@self.A)
        return M # M must be SDP 

class LMI_HInf(Module):
    def __init__(self, model, config) -> None:
        super(LMI_HInf, self).__init__()

        linmod = find_module(model, NNLinear)
        epsilon = config.epsilon
        gamma = config.gamma
        nd = config.nd
        if linmod is not None:
            self.A = linmod.A.weight
            nu = linmod.B.weight.shape[1]
            self.B = linmod.B.weight[:,nu-nd:]
            self.C = linmod.C.weight
        else:
            raise AttributeError("Model has no linear module")

        self.P = nn.Linear(self.A.shape[0], self.A.shape[0], bias = False)

        P, gamma = self.eval_(abs_tol=1e-5, solver ="MOSEK", epsilon= epsilon)
        self.P.weight = Parameter(P)
        #geo.positive_definite(self.P, "weight")

        self.gamma = Parameter(gamma)
        print(f'Gamma prescrit = {float(self.gamma):.4e}')


    def forward(self):
        '''
         M = [A^TPA-P  A^T*P*B C^T
                * B^T*P*B -gammaI 0(nu,ny)
                *  *  -\gamma I_{nu}] <0
        ** returns : -M 
        
        '''
        self.symetrizeP_()
        eig_val, _ = torch.linalg.eig(self.P.weight)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("P is not SDP")
        nu = self.B.shape[1]
        M11 = torch.matmul(self.A.T, self.P.weight) + torch.matmul(self.P.weight, self.A) + 1/self.gamma*torch.matmul(self.C.T, self.C)
        M12 = torch.matmul(self.P.weight, self.B)
        
        M22 = -self.gamma*torch.eye(nu)

        M = torch.cat((torch.cat((M11,M12), 1), torch.cat((M12.T,M22), 1)), 0)
        eig_val, _ = torch.linalg.eig(-M)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("M is not SDP")
        return -M # Always have to return an positive semidefinite matrix

    def eval_(self, abs_tol = 1e-6, solver = "MOSEK", epsilon = 1e-6):
        
        A = self.A.detach().numpy()
        B = self.B.detach().numpy()
        C = self.C.detach().numpy()
        print(" Initializing HInf LMI \n")

        nu = B.shape[1]
        nx = A.shape[0]
        ny = C.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)
        gamma = cp.Variable()
        D = np.zeros((ny,nu))


        M = cp.bmat([[A.T@P + P@A , P@B, C.T], [B.T@P, -gamma *np.eye(nu), D.T], [ C, D, -gamma*np.eye(ny)]])


        constraints = [M << -np.eye(nu+nx+ny)*epsilon, P -(abs_tol)*np.eye(nx)>> 0, gamma - abs_tol >=0] 
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
        return torch.Tensor(P.value).to(dtype=torch.float32), torch.Tensor(gamma.value).to(dtype=torch.float32)

    def symetrizeP_(self):
        self.P.weight = Parameter(0.5*(self.P.weight + self.P.weight.T))

class LMI_H2(Module): # TO-DO
    '''
        This module will compute for a linear system with a specified state-space representaiton
        x_+ = Ax + Bu
        y = Cx
        nu = trace(CPC^T) with P solution A^TP + PA + BB^T = 0
        The H2 norm of a system can be interpreted as an average gain over all frequecies
    '''
    def __init__(self, model, gamma : float, epsilon = 1e-6) -> None:
        super(LMI_HInf, self).__init__()

        linmod = find_module(model, NNLinear)
        if linmod is not None:
            self.A = linmod.A
            self.B = linmod.B
            self.C = linmod.C
        else:
            raise AttributeError("Model has no linear module")

        self.P = nn.Linear(self.A.shape[0], self.A.shape[0], bias = False)

        P, nu = self.solve_lmi(self.A,self.B, self.C, abs_tol=1e-5, solver ="MOSEK", epsilon= epsilon)
        self.P.weight = Parameter(P.requires_grad_(True))
        #geo.positive_definite(self.P, "weight")

        self.nu = Parameter(nu)
        print(f'Nu prescrit = {float(self.nu):.4e}')


    def forward(self):
        ''' H2 = trace(CPC^T) with P solution A^TP + PA + BB^T = 0
         M = [A^TPA-P  A^T*P*B C^T
                * B^T*P*B -gammaI 0(nu,ny)
                *  *  -\gamma I_{nu}] <0
        ** returns : -M 
        
        '''
        self.symetrizeP_()
        eig_val, _ = torch.linalg.eig(self.P.weight)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("P is not SDP")
        nu = self.B.shape[1]
        M11 = torch.matmul(self.A.T, self.P.weight) + torch.matmul(self.P.weight, self.A) + 1/self.gamma*torch.matmul(self.C.T, self.C)
        M12 = torch.matmul(self.P.weight, self.B)
        
        M22 = -self.nu*torch.eye(nu)

        M = torch.cat((torch.cat((M11,M12), 1), torch.cat((M12.T,M22), 1)), 0)
        return -M # Always have to return an positive semidefinite matrix

    def solve_lmi(self, A: torch.Tensor,B: torch.Tensor,C: torch.Tensor, abs_tol = 1e-6, solver = "MOSEK", espilon = 0.1):
        
        A = A.detach().numpy()
        B = B.detach().numpy()
        C = C.detach().numpy()
        print(" Initializing H2 LMI \n")

        nu = B.shape[1]
        nx = A.shape[0]
        ny = C.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)
        D = np.zeros((ny,nu))

        M = A.T@P + P@A + B@B.T


        constraints = [M == 0, P -(abs_tol)*np.eye(nx)>> 0] 
        nu = cp.trace(C@P@C.T)

        objective = cp.Minimize(nu) # Find the H2 gain

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print("nu = {} \n".format(prob.value))
            print(" P eigenvalues : \n")
            eigVal = np.linalg.eig(P.value)[0]
            print(eigVal)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        
        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        return torch.Tensor(P.value).to(dtype=torch.float32), torch.Tensor(nu.value).to(dtype=torch.float32)

    def symetrizeP_(self):
        self.P.weight = Parameter(0.5*(self.P.weight + self.P.weight.T))
        


class LMI_decay_rate(Module):
    def __init__(self, model, alpha : float, epsilon = 1e-6) -> None:
        super(LMI_decay_rate, self).__init__()

        self.alpha = alpha
        linmod = find_module(model, NNLinear)
        if linmod is not None:
            self.A = linmod.A
        else:
            raise AttributeError("Model has no linear module")

        self.P = Parameter(torch.empty_like(self.A))
        geo.positive_semidefinite(self, "P")
        M,P = self.solve_lmi(epsilon = epsilon)
        self.P = Parameter(P)
        self.nm = M.shape[0]

    def forward(self):
        eig_val, _ = torch.linalg.eig(self.P)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("P is not SDP")

        M = -(self.A.T@self.P + self.P@self.A + 2*self.alpha*self.P)
        return M # M must be SDP

    def solve_lmi(self, epsilon = 1e-6, solver = "MOSEK"):
        
        A = self.A.detach().numpy()
        print(" Initializing Lyapunov LMI \n")
        alpha = self.alpha
        nx = A.shape[0]
        P = cp.Variable((nx,nx), 'P', PSD=True)

        M = A.T@P + P@A + 2*alpha*P


        constraints = [M << -np.eye(nx)*epsilon, P -(epsilon)*np.eye(nx)>> 0] 
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
        P = torch.Tensor(P.value).to(dtype=torch.float32)
        M = torch.Tensor(M.value).to(dtype=torch.float32)
        return M, P

    def symP_(self):
        self.P = Parameter(0.5*(self.P + self.P.T))

class Lipschitz1Layer(Module):
    def __init__(self, upper_slope :float, W_out : Tensor, W_in : Tensor, init_margin : float = 1e-6) -> None:
        super(Lipschitz1Layer,self).__init__()
        self.alpha =  0.0
        self.beta = upper_slope
        self.w_out = W_out
        self.w_in = W_in

        M,T = self.solve_lmi(epsilon = init_margin)
        self.T = Parameter(T)
        self.nM = M.shape[0]

    def solve_lmi(self, epsilon = 1e-6, solver = "MOSEK"):

        print(" Initializing Lipschitz LMI \n")
        W_out = self.w_out.detach().numpy()
        W_in = self.w_in.detach().numpy()

        beta = self.beta

        n_in = self.w_in.shape[1]
        n_hid = self.w_in.shape[0]
        n_out = self.w_out.shape[0]

        self.n_in = n_in
        self.n_out = n_out
        self.n_hid = n_hid

        T = cp.Variable((n_hid,n_hid), 'T', PSD=True)
        lip = cp.Variable() # square of the true lipschitz constant for solving the problem
        M11 = -lip*np.eye(n_in)
        M12 = beta*W_in.T@T
        M23 = W_out.T
        M = cp.bmat([[M11, M12, np.zeros((n_in,n_out))], 
                    [M12.T, -2*T, M23], [np.zeros((n_out,n_in)), M23.T, -np.eye(n_out, n_out)]])

        nM = n_in+n_hid+n_out
        constraints = [M << -np.eye(nM)*epsilon, T -(epsilon)*np.eye(nM)>> 0, lip-epsilon > 0] 
        objective = cp.Minimize(lip) # Feasibility problem

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print(" Lipschitz Constant upper bound : \n")
            
            print(lip.value)
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        self.lip = torch.Tensor(np.sqrt(lip.value)).to(dtype=torch.float32)
        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        M = torch.Tensor(M.value).to(dtype=torch.float32)
        T = torch.Tensor(T.value).to(dtype=torch.float32)
        return -M, T

    def forward(self):
        eig_val, _ = torch.linalg.eig(self.T)
        if not torch.all(torch.real(eig_val)>0):
            raise AssertionError("T is not SDP")

        M11 = self.lip*torch.eye(self.n_in)
        M12 = self.beta*self.w_out.T@self.T
        M13 = torch.zeros((self.n_in, self.n_out))
        M23 = self.w_out.T

        M = torch.cat((torch.cat((M11,M12,M13), 1), 
                       torch.cat((M12.T,-2*self.T, M23), 1),
                       torch.cat((M13.T, M23.T, -torch.eye(self.n_out)))), 0)
        return -M # M must be SDP

    def symT(self):
        self.T = Parameter(0.5*(self.T + self.T.T))

class Lipschitz(Module):
    def __init__(self, model, upper_slope : float, epsilon : float = 1e-6, L_presc = 1e7) -> None:
        super(Lipschitz, self).__init__()

        layer_dims = []
        idx_weights = []
        idx = 0
        for layer in model.layers:
            if hasattr(layer, 'weight'):
                layer_dims.append(tuple(layer.weight.shape))
                idx_weights.append(idx)
            idx = idx +1
        self.idx_weights = idx_weights
        self.beta = upper_slope
        self.dims = model.dims
        self.layers = model.layers # Model must be pure feedforward
        T, L = self.solve_lmi(epsilon = epsilon)
        # Implementation to do with n independent variables it won't be necessary to re diag the tensor at each step
        self.T = Parameter(T.requires_grad_(True)) 
        if L< L_presc:
            self.L = L_presc
        else:
            self.L = L
        print(f'Prescribed Lipschitz constant = {float(self.L):.4e}')



    def solve_lmi(self, epsilon = 1e-6, solver = "MOSEK"):
        print("Initializing Lipschitz LMI \n")

        Ts = [cp.Variable((self.dims[i], self.dims[i]), diag = True) for i in range(1,len(self.dims)-1)]
        T = block_diag(Ts)
        Ft = cp.bmat([[np.zeros(T.shape), self.beta*T],[self.beta*T, -2*T]])
        dims_A = self.idx_weights[:-1]
        Ws = [self.layers[i].weight.detach().numpy() for i in dims_A]
        W = block_diag(Ws)
        print(W.shape)
        A = cp.hstack([W,np.zeros((W.shape[0],self.dims[-2]))])
        print(A.shape)
        dims_B = self.dims[1:-1]
        
        I_B = np.eye(sum(dims_B))
        B = cp.hstack([np.zeros((I_B.shape[0],self.dims[0])), I_B])
        print(B.shape)
        AB = cp.vstack([A,B])
        LMI = AB.T @ Ft @ AB

        LMI_schur = block_diag([LMI,np.zeros((self.dims[-1],self.dims[-1]))])

        # 2eme partie LMI
        lip = cp.Variable()

        L = -lip*np.eye(self.dims[0])

        # Block schur last layer
        b11 = np.zeros((self.dims[-2],self.dims[-2]))
        b12 = self.layers[self.idx_weights[-1]].weight.T.detach().numpy()
        b21 = self.layers[self.idx_weights[-1]].weight.detach().numpy()
        b22 = -np.eye(self.dims[-1])
        bf = cp.bmat([[b11, b12], [b21,b22]])

        dim_inter = sum(self.dims[1:-2]) 
        if dim_inter > 0:
            inter = np.zeros((dim_inter, dim_inter))
            part2 = block_diag([L,inter, bf])
        else:
            part2 = block_diag([L,bf])

        M = LMI_schur + part2 

        '''      
        Dn = -np.eye(self.dims[-1])


        lDiag = [D0, Ts, Dn]

        lSubDiag = [self.beta*Ts[i] @ self.layers[i].weight().detach().numpy() for i in range(0,len(self.layers)-1)] 
        lSubDiag.append(self.layers[-1].weight().detach().numpy())
        lUpDiag = [subDiagTensor.T for subDiagTensor in lSubDiag]

        M_diag = block_diag(lDiag)
        M_subdiag = block_diag(lSubDiag, k=-1)
        M_updiag = block_diag(lUpDiag, k=1) 

        M = M_diag + M_subdiag + M_updiag
        '''
        nM = M.shape[0]
        nT = T.shape[0]
        constraints = [M << -np.eye(nM)*epsilon, T -(epsilon)*np.eye(nT)>> 0, lip-epsilon >= 0] 
        objective = cp.Minimize(lip) # Find lowest lipschitz constant

        prob = cp.Problem(objective, constraints= constraints)
        prob.solve(solver)
        if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
            print(" Lipschitz Constant upper bound : \n")
            
            print(np.sqrt(lip.value))
            
        else:
            raise ValueError("SDP problem is infeasible or unbounded")

        lip = torch.Tensor(np.array(np.sqrt(lip.value))).to(dtype=torch.float32)
        # Evaluate if it closed to the boundary of the LMI
        # X = np.linalg.inv(np.matmul(A.T ,P.value) + np.matmul(P.value,A))
        # t = np.matmul(-A, X) -np.matmul(X,A.T) + 2*alpha*X
        # If it is close to zero it is at the center
        #Ts = [torch.Tensor(tens.value.todense()).to(dtype=torch.float32) for tens in Ts]
        #T = torch.block_diag()
        T = torch.Tensor(T.value).to(dtype=torch.float32)
        M = torch.Tensor(M.value).to(dtype=torch.float32)
        return T, lip

    def forward(self):
        
        T = self.diagT_()
        Ft = torch.vstack([torch.hstack([torch.zeros(T.shape), self.beta*T]),torch.hstack([self.beta*T, -2*T])])
        dims_A = self.idx_weights[:-1]
        Ws = [self.layers[i].weight for i in dims_A]
        W = torch.block_diag(*tuple(Ws))
        A = torch.hstack([W,torch.zeros((W.shape[0],self.dims[-2]))])
        dims_B = self.dims[1:-1]
        I_B = torch.eye(sum(dims_B))
        B = torch.hstack([torch.zeros((I_B.shape[0],self.dims[0])), I_B])
        AB = torch.cat([A,B])
        LMI = AB.T @ Ft @ AB

        LMI_schur = torch.block_diag(LMI,torch.zeros((self.dims[-1],self.dims[-1])))

        # 2eme partie LMI
        lip = self.L**2

        L = -lip*torch.eye(self.dims[0])

        # Block schur last layer
        b11 = torch.zeros((self.dims[-2],self.dims[-2]))
        b12 = self.layers[self.idx_weights[-1]].weight.T
        b1 = torch.hstack([b11,b12])
        b21 = self.layers[self.idx_weights[-1]].weight
        b22 = -torch.eye(self.dims[-1])
        b2 = torch.hstack([b21,b22])
        bf = torch.vstack([b1,b2])

        dim_inter = sum(self.dims[1:-2]) 
        inter = torch.zeros((dim_inter, dim_inter))
        part2 = torch.block_diag(L,inter, bf)

        M = LMI_schur + part2 
        #torch.block_diag
        if torch.any(torch.isnan(M)):
            print(M)
        if torch.any(torch.isnan(T)):
            print(T)
        
        return -M, T 

    def diagT_(self):
        if torch.any(torch.isnan(self.T)):
            print(torch.diag(torch.diag(self.T)))
        return torch.diag(torch.diag(self.T))
    
'''
    Discrete time LMI versions
'''

class LMI_HInf_Dist_discrete(Module):
    def __init__(self, A,B,C,gamma) -> None:
        super(LMI_HInf_Dist_discrete, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.P = nn.Linear(A.shape[0],A.shape[0])
        geo.PSD(self.P) # Positive semi definite
        self.gamma = gamma


    def forward(self):
        '''
         M = [A^TPA-P  A^T*P*B C^T
                * B^T*P*B -gammaI 0(nu,ny)
                *  *  -\gamma I_{nu}] <0
        ** returns : -M 
        
        '''
        M11 = torch.matmul(torch.matmul(self.A.T, self.P), self.A) - self.P
        M12 = torch.matmul(torch.matmul(self.A.T, self.P), self.B)
        M13 = self.C.T
        M22 = torch.matmul(torch.matmul(self.B.T, self.P), self.B) - self.gamma*torch.eye(nu)
        nu = self.B.shape[1]
        ny = self.C.shape[0]
        M23 = torch.zeros((nu,ny))
        M33 = self.gamma*torch.eye(ny)

        M = torch.Tensor([[M11, M12, M13],[M12.T, M22, M23],[M13.T, M23.T, M33]])
        return -M
