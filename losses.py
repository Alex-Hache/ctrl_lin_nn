import torch
from torch.nn import MSELoss
from lmis import *
from utils import *
import torch.nn as nn


def getLoss(config, model = None):
    if config.loss == 'mse':
        if config.reg_lmi != 'trace':
            if config.lmi != '':
                if config.lmi== 'lyap':
                    lmi = LMI_decay_rate(model, config.alpha, config.epsilon)
                elif config.lmi == 'hinf':
                    lmi = LMI_HInf(model, config)
                elif config.lmi == 'h2':
                    lmi = LMI_H2(model, config.nu, config.epsilon)

                if config.reg_lmi == 'logdet':
                    crition = Mixed_MSELOSS_LMI(lmi, mu = config.mu)
                else:
                    raise ValueError("Please specify a reg term")
            else:
                crition = Mixed_MSELOSS()
        else:
            crition = Mixed_MSE_Trace(model, mu = config.mu)
    else:
        raise(NotImplementedError("Only mse loss is implemented so far"))
    return crition
    
class Mixed_MSELOSS(torch.nn.Module):
    """
        Introduced a convex mixed mse on the state and the ouput
    """
    def __init__(self, eta =0) -> None:
        super(Mixed_MSELOSS, self).__init__()

        self.crit = MSELoss()
        self.eta = eta

    def forward(self,y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        return self.eta*x_mse + (1- self.eta)*y_mse

class Mixed_MSELOSS_LMI(torch.nn.Module):
    def __init__(self, lmi, eta= 0, mu = 1) -> None:
        super(Mixed_MSELOSS_LMI, self).__init__()
        self.crit = MSELoss()
        self.lmi = lmi
        self.mu = mu
        self.eta = eta

    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        L1 = self.eta*x_mse + (1- self.eta)*y_mse
        lmi = self.lmi()
        L2 = 0.0
        '''
        for lmi in lmis:
            L2 = L2 -torch.logdet(lmi)'''
        L2 = -torch.logdet(lmi)
        if torch.isnan(L2):
            print('Test')
        return L1 + self.mu*L2

    def update_mu_(self, scale):
        self.mu = self.mu*scale


class Mixed_MSE_Trace(torch.nn.Module):
    def __init__(self, model, eta= 0, mu = 1) -> None:
        super(Mixed_MSE_Trace, self).__init__()
        self.crit = MSELoss()
        self.model = model
        self.mu = mu
        self.eta = eta

    def forward(self, y_true, y_sim, x_true, x_sim):
        x_mse = self.crit(x_true, x_sim)
        y_mse = self.crit(y_true, y_sim)
        L1 = self.eta*x_mse + (1- self.eta)*y_mse
        gamma_2 = torch.trace(self.model.C@torch.linalg.inv(self.model.Wcinv)@self.model.C.T)
        return L1 + self.mu*gamma_2

    def update_mu_(self, scale):
        self.mu = self.mu*scale



class Mixed_LOSS_LMI(torch.nn.Module):
    def __init__(self, lmi, mu = 1) -> None:
        super(Mixed_LOSS_LMI, self).__init__()
        self.crit = MSELoss()
        self.lmi = lmi
        self.mu = mu

    def forward(self, pred, true):

        L1 = self.crit(pred, true)
        lmis = self.lmi()
        #assert torch.all(torch.real(eig_val)>0)
        L2 = 0.0
        for lmi in lmis:
            L2 = L2 -torch.logdet(lmi)
        return L1, L2

    def update_mu_(self, scale):
        self.mu = self.mu*scale
