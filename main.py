import os
import sys
from argparse import ArgumentParser

import numpy as np

from models import *
from integrators import *
from lmis import *
from argparse import ArgumentParser
from train import *

def main(args):

    config = args
    config.test_freq = 200
    config.root_dir = '/results'

    # Choice of the dataset
    if config.dataset == 'toy_linear':
        config.in_channels = 1
        config.out_channels = 1
        config.train_batch_size = 64
        config.num_workers = 0
        config.train_set = 'toy_train.mat'
        config.test_set = 'toy_test.mat'
        config.nx = 2
    elif config.dataset == 'pendulum':
        config.train_batch_size = 512
        config.train_set = 'pend_train.mat'
        config.test_set = 'pend_test.mat'
        config.nx = 2 
        config.nd = 1

    # Choice of the model 
    if config.model == 'linear':
        config.loss == 'mse'
        if config.lin != '':
            config.lmi = ''
            config.reg_lmi = ''
        if config.lin =='h2nn':
            config.reg_lmi = 'trace' 
    elif config.model == 'flnssm':
        config.n_layers = 1
        config.actF = nn.Tanh()
        config.loss == 'mse'
        if config.lin != '':
            config.lmi == ''
            config.reg_lmi = ''
        if config.lin =='h2nn':
            config.reg_lmi = 'trace' 

        
    if config.lmi != '':
        config.reg_lmi = 'logdet'
        config.epsilon = 1e-4 # Inital margin to the feasibility set
        config.max_ls = 1000 # 1000
        config.alpha_ls  = 0.5 # 0.5
        config.mu_dec = 0.1 # Decrement of the barrier term to each plateau

        
    # Creation of the results directory
    if config.lin == 'alphaLNN' or config.lmi == 'lyap': # Alpha stable models
        config.train_dir = f"./{config.root_dir}/{config.dataset}/seed_{config.seed}/{config.struct}/{config.model}/{config.lin}/alpha_{config.alpha}"
    elif config.lin == 'hinfnn' or config.lmi == 'hinf': # L2-stable models
        config.train_dir = f"./{config.root_dir}/{config.dataset}/seed_{config.seed}/{config.struct}/{config.model}/{config.lin}/gamma_{config.gamma}"
    elif config.lin == 'h2nn' or config.lmi == 'h2': # H2-stable models
        config.train_dir = f"./{config.root_dir}/{config.dataset}/seed_{config.seed}/{config.struct}/{config.model}/{config.lin}/gamma2_{config.mu}"   
    else:
        config.train_dir = f"./{config.root_dir}/{config.dataset}/seed_{config.seed}/{config.struct}/{config.model}"   
    
    os.makedirs("./data", exist_ok=True)
    os.makedirs(config.train_dir, exist_ok=True)
    if config.mode == 'train':
        if config.struct == 'rnn':
            train_rnn(config)
    


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('-d', '--dataset', type=str, default='toy_linear',
                        help="dataset [toy_linear, pendulum]")
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--struct', type=str, default='rnn', help = "[fnn, rnn]")
    parser.add_argument('-m', '--model', type=str, default='linear',
                        help="[linear, flnssm]")
    parser.add_argument('--lin', type = str, default='hinfnn',
                        help = "[alphaLNN, hinfnn, h2nn]")
    parser.add_argument('-i', '--integrator', type = str, default = 'RK4')
    parser.add_argument('-a', '--alpha', type=float, default=3,
                        help="Network alpha stability bound")
    parser.add_argument('-g', '--gamma', type=float, default=0.5,
                        help="Network L2 gain")
    parser.add_argument('-n', '--nu', type=float, default=1.0,
                        help="Network H2 gain")
    parser.add_argument('--nh', '--hidden_size', type = int, default = 20)

    parser.add_argument('-e','--epochs', type=int, default=2000)
    parser.add_argument('--seq_len', type = int, default = 30)
    parser.add_argument('-p', '--patience', type = int, default=100)

    parser.add_argument('--tol_change', type = float, default = 0.01) # 1%
    parser.add_argument('--lr', type=float, default= 1e-2, help="learning rate")
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('-l', '--lmi', type = str, default='', help ="[lyap, hinf, h2]")
    parser.add_argument('--mu', type = float, default = 0.01)
    parser.add_argument('--bCertGrad', type = bool, default=True)
    parser.add_argument('-b', '--backtrack', type = bool, default = True)
    parser.add_argument('--epsilon', type = float, default = 0.1)

    
    args = parser.parse_args()

    main(args)
