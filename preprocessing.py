import torch
import numpy as np
import matlab.engine

from scipy.io import savemat, loadmat
def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_data(config):
    data_train = loadmat(f"./data/{config.dataset}/{config.train_set}")
    data_test = loadmat(f"./data/{config.dataset}/{config.test_set}")
    return data_train, data_test


def preprocess_mat_file(dMat : dict, nx : int, idxMax : int = 100000):
        fs = dMat['fs']
        ts = 1/fs[0][0]
        u = dMat['uTot']
        y = dMat['yTot']

        if u.shape[0]< u.shape[1]: # number of samples is on dimension 1
            u = u.T
        if y.shape[0] < y.shape[1]:
            y = y.T

        u_torch = torch.from_numpy(u[:idxMax,:]).to(dtype= torch.float32)
        y_torch = torch.from_numpy(y[:idxMax,:]).to(dtype= torch.float32)
        
        try : # Do we have disturbances ?
            d = dMat['pTot']
            if d.shape[0] < d.shape[1]:
                d = d.T
            d_torch = torch.from_numpy(d[:idxMax,:]).to(dtype= torch.float32)
            u_torch = torch.cat((u_torch, d_torch), dim = 1)
        except :
            u_torch = torch.from_numpy(u[:idxMax,:]).to(dtype= torch.float32)


        try : # Do we have state measurements ?
            dMat['xTot']
            x = np.reshape(x, (max(x.shape), min(x.shape)))
            x_torch = torch.from_numpy(x[:idxMax,:]).to(dtype= torch.float32)
        except:
            x_torch = torch.zeros((y.shape[0], nx), dtype=torch.float32, requires_grad=True)

        y_torch_dot = (y_torch[1:,:]-y_torch[0:-1,:])/ts
        y_torch_dot = torch.cat([torch.zeros((1,y.shape[1])),y_torch_dot])

        return u_torch, y_torch, x_torch, y_torch_dot, ts
        


def findBLA(u : np.ndarray, y : np.ndarray, nx : int,
            ts : float, model_type : str = 'discrete',
            save : bool = False, strNameSave : str = "BLA.mat"):
    """
        Perform linear identification using scriptName Matlab function

        params : 
            * u : input data
            * y : output data
            * nx : dimension of state-space
            * ts : sample time
            * save : do we want to save the linMod structure into a mat file
        returns :
            A, B, C, D matrices of the identified state-space as pytorch Tensors
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(eng.pwd()))
    data_InMat = matlab.double(u.tolist()) # N_samples x N_channels matlab double
    data_OutMat = matlab.double(y.tolist()) # N_samples x N_channels matlab double
    nx = matlab.double([nx])
    mTs = matlab.double([ts])
    linMod = eng.initLin(data_InMat, data_OutMat, nx, mTs, model_type) # Call the initLin.m script  
    eng.quit()

    if save :
        linMod['A'] = np.asarray(linMod['A'])
        linMod['B'] = np.asarray(linMod['B'])
        linMod['C'] = np.asarray(linMod['C'])
        linMod['uTot'] = np.asarray(u)
        linMod['yTot'] = np.asarray(y)
        savemat(strNameSave, linMod)
    A = torch.from_numpy(np.asarray(linMod['A'])).to(torch.float32)
    B = torch.from_numpy(np.asarray(linMod['B'])).to(torch.float32)
    C = torch.from_numpy(np.asarray(linMod['C'])).to(torch.float32)
    D = torch.from_numpy(np.asarray(linMod['D'])).to(torch.float32).unsqueeze(dim=0)
    return A, B, C, D