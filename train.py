import torch
import numpy as np
from preprocessing import *
from models import *
from losses import *
from integrators import *
from postprocessing import *

from alive_progress import alive_bar
from scipy.io import savemat
import copy

def train_rnn(config):
    seed_everything(config.seed)
    train_set, test_set = get_data(config)
    nx = config.nx
    # Prepare data
    u_train, y_train, x_train, _, ts = preprocess_mat_file(train_set, nx)
    u_test, y_test, _, _, _ = preprocess_mat_file(test_set, nx)
    
    train_data = u_train, y_train, x_train
    test_data = u_test, y_test

    model = getModel(config, train_data)
    if config.lin =='': # No particular parameterization so we can use data to initialize linear part
                # Perform linear estimate from train data
        save_bla_path = os.path.join(os.getcwd(), 'pend_BLA.mat')
        '''
        A0, B0, C0, _ = findBLA(u_train, y_train, nx, ts= ts, model_type = 'continuous',
                                save= True, strNameSave = save_bla_path)'''
        bla = loadmat('./data/pendulum/pend_BLA.mat')
        A0 = torch.from_numpy(bla['A']).to(torch.float32)
        B0 = torch.from_numpy(bla['B']).to(torch.float32)
        C0 = torch.from_numpy(bla['C']).to(torch.float32)
        scale = 0.1 #Scaled down factor = 0.1 for both tries

        model.init_weights(A0, B0*scale, C0, isLinTrainable = True) 
    criterion = getLoss(config, model) # Call loss first to instantiate criterion on model variable and not
    sim_model = getIntegrator(config,model, ts)

    if not os.path.exists(config.train_dir):
        os.makedirs(config.train_dir)

    print(f"Set global seed to {config.seed:d}")
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    if nparams >= 1000000:
        print(f"name: {config.model}, num_params: {1e-6*nparams:.1f}M")
    else:
        print(f"name: {config.model}, num_params: {1e-3*nparams:.1f}K")


    # Choice of the training algorithm
    if hasattr(criterion, 'lmi'):
        best_model, dict_res = train_rec(sim_model, criterion, train_data, test_data, config)
    else:
        best_model, dict_res = train_rec(sim_model, criterion, train_data, test_data, config)

    # Saving config
    savemat(config.train_dir + '/config.mat', config.__dict__)
    # Saving best model
    savemat(config.train_dir + '/model.mat',  best_model.extract_params())

    savemat(config.train_dir + '/losses.mat', dict_res)

    if config.dataset  == 'pendulum':

        '''
        Closed-loop simulation
        '''

        # Give the name of the matfile to python function

        net_dims = [1, 1, nx]
        strSimFigName = f"{config.train_dir}/sim_closed_loop.fig"
        str_clr_sim_fig_name = strSimFigName[:-4]
        strDirSaveName = config.train_dir + '/model.mat'
        sim_closed_loop(strDirSaveName, net_dims, ts, str_clr_sim_fig_name)
    return best_model, dict_res 


def train_rec(model, criterion, train_data, test_data, config):

    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        print(f"name: {config.model}, num_params: {1e-6*nparams:.1f}M")
    else:
        print(f"name: {config.model}, num_params: {1e-3*nparams:.1f}K")
    
    # Prepare data
    u_train, y_train, x_train = train_data
    u_test, y_test = test_data

    nx = x_train.shape[1]

    # Batch extraction funtions
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = u_train.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch samples indices

        # Extract batch data
        
        batch_x0_hidden = x_train[batch_start, :]
        batch_x_hidden = x_train[[batch_idx]]
        batch_u = u_train[[batch_idx]]
        batch_y = y_train[[batch_idx]]

        return batch_x0_hidden, batch_u, batch_y, batch_x_hidden


    # Setup optimizer
    lr = config.lr
    params_net = list(model.parameters())
    params_hidden = [x_train]
    optimizer = torch.optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': lr},
    ], lr=lr)

    x0_val = torch.zeros((nx), dtype=torch.float32)
    u_torch_val = u_test.to(dtype= torch.float32)
    y_true_torch_val = y_test.to(dtype= torch.float32)
    
    _, y_sim_init = model.simulate(u_torch_val, x0_val)
    val_mse =  torch.mean((y_true_torch_val-y_sim_init)**2)
    if torch.isnan(val_mse):
        val_mse = torch.inf
    print("Initial val_MSE = {:.7f} \n".format(float(val_mse)))
    torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")

    vLoss = []
    vVal_mse = []
    vInfo = []

    start_time = time.time()
    # Training loop
    best_loss = val_mse
    #best_model = model.clone()
    no_decrease_counter = 0
    batch_size = config.train_batch_size
    seq_len = config.seq_len
    test_freq = config.test_freq
    tol_change = config.tol_change

    num_iter = config.epochs
    patience = config.patience

    with alive_bar(num_iter) as bar:
        epoch_loss = 0.0
        for itr in range(0, num_iter):

            optimizer.zero_grad()

            # Simulate
            #x0_torch = torch.zeros((nx))
            batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
            x_sim_torch_fit, y_sim_torch_fit = model(batch_u, batch_x0_hidden)


            # Compute fit loss
            loss = criterion(batch_y, y_sim_torch_fit, batch_x_hidden, x_sim_torch_fit)

            epoch_loss += float(loss.item())

            if itr % test_freq == 0 or itr == num_iter-1:
                # Statistics
                epoch_loss = epoch_loss/test_freq
                vLoss.append(epoch_loss)

                with torch.no_grad():
                    # Simulation perf on test data
                    _, y_sim_val = model.simulate(u_torch_val, x0_val)
                    val_mse =  torch.mean((y_true_torch_val-y_sim_val)**2)
                    vVal_mse.append(val_mse)

                if (best_loss - val_mse)/best_loss > tol_change:
                        no_decrease_counter = 0
                        best_loss = val_mse
                        torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")  

                        
                else:
                    no_decrease_counter += 1
                print(" Epoch loss = {:.7f} || Val_MSE = {:.7f} || Best loss = {:.7f} \n".format(float(epoch_loss),
                        float(val_mse), float(best_loss)))    
                epoch_loss = 0.0
                if no_decrease_counter> patience/5 and hasattr(criterion, 'mu'):
                    criterion.update_mu_(0.1)
                    print(f"Updating barrier term weight : mu = {criterion.mu}")
                    no_decrease_counter = 0
                if hasattr(criterion, 'mu') and criterion.mu<1e-8:
                    break
                if no_decrease_counter > patience: # early stopping
                    break
                
            if (math.isnan(loss)): 
                break
            # Optimize
            loss.backward()
            optimizer.step()

            
            bar()

    train_time = time.time() - start_time

    print("Total dentification runtime : {} \n Best loss : {} \n".format(train_time, best_loss))

    model_state = torch.load(f"{config.train_dir}/model.ckpt")
    model.load_state_dict(model_state)
    # Final simulation perf on test data
    x_sim_val, y_sim_val = model.simulate(u_torch_val, x0_val)
    val_mse =  torch.mean((y_true_torch_val-y_sim_val)**2)

    # Final simulation on train data
    x_sim_train, y_sim_train = model.simulate(u_train, torch.zeros(nx))
    train_mse = torch.mean((y_train - y_sim_train)**2)
    print(" Final MSE = {:.7f} || Val_MSE = {:.7f} \n".format(float(train_mse.detach()),float(val_mse)))

    try:
        model.ss_model.eval_() # Either the model
    except:
        if hasattr(criterion, 'lmi'):
            criterion.lmi.eval_() # or the criterion has an lmi to check
    strSaveName = str(config.model) + f'lr_{lr}' + f'_{num_iter}epch'

    fig = plt.figure()
    plt.plot(y_sim_train, label = 'model')
    plt.plot(y_train, label = 'data')
    plt.legend()
    plt.show()

    fig.savefig(f"{config.train_dir}/{strSaveName}_sim.png")

    fig = plt.figure()

    plt.plot(range(len(vLoss)),np.log10(np.array(vLoss)), label = 'Loss')
    plt.plot(range(len(vVal_mse)), np.log10(np.array(vVal_mse)), label = 'Test loss') 

    # Ajouter une légende
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE (log10)")
    plt.title(f"{config.model}")
    plt.show()


    fig.savefig(f"{config.train_dir}/{strSaveName}.png")

    dict_res = {'train_loss' : vLoss,
                'test_loss' : vVal_mse, 
                'y_sim' : y_sim_val.squeeze(0).numpy(),
                'y_sim_train' : y_sim_train.squeeze(0).numpy(),
                'train_mse' : train_mse.detach().numpy(),
                'val_mse' : val_mse.detach().numpy()}

    savemat(f"{config.train_dir}/results.mat", dict_res)

    return model, dict_res  

