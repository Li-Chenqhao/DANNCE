import numpy as np
from scipy import stats
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import setup

'''
Network training
'''
def network_training(detail = False):
    loss = nn.BCELoss()
    model_net = setup.network_FNN(d, params['nd'], params['nw'], params['output_scale']).to(device)
    net_optimizer = optim.Adam(model_net.parameters(), lr = params['lr'])
    scheduler = optim.lr_scheduler.StepLR(net_optimizer, step_size = params['lr_step'], gamma = params['lr_gamma'])
    best_loss = 100
    count = 0
    for epoch in range(params['iter_num']):
        model_net.zero_grad()

        real_data = setup.data_loader(params['bs'], training_data_real)
        real_label = torch.full((params['bs'],), real_lb, device=device).double()
        real_mod_pdf = model_net(torch.from_numpy(real_data).to(device).float()).reshape(params['bs'])
        real_ref_pdf = torch.from_numpy(ref_model.pdf(real_data)).to(device)
        real_prob = setup.get_prob(real_ref_pdf, real_mod_pdf, params['nu'])
        real_loss = loss(real_prob, real_label)
        real_loss.backward()

        fake_data = setup.data_loader(params['nu']*params['bs'], training_data_ref)
        fake_label = torch.full((params['nu']*params['bs'],), fake_lb, device=device).double()
        fake_mod_pdf = model_net(torch.from_numpy(fake_data).to(device).float()).reshape(params['nu']*params['bs'])
        fake_ref_pdf = torch.from_numpy(ref_model.pdf(fake_data)).to(device)
        fake_prob = setup.get_prob(fake_ref_pdf, fake_mod_pdf, params['nu'])
        fake_loss = params['nu']*loss(fake_prob, fake_label)
        fake_loss.backward()

        current_loss = (real_loss + fake_loss).detach().cpu()

        if current_loss < best_loss:
            best_loss = current_loss
            count = 0
        else:
            count += 1
        
        if count == params['stop_threshold']:
            break

        if detail and (epoch+1)%params['log_step'] == 0:
            print('epoch: {}, loss: {}'.format(epoch + 1, current_loss))

        net_optimizer.step()
        scheduler.step()
    return model_net

'''
Main estimation function
'''
if __name__ == '__main__':
    start_time = time.time()

    # Parameters
    d = 3 # dimension
    no_augmentation = False # Set as 'False' for data augmentation, and 'True' for estimation without data augmentation
    if no_augmentation:
        params = yaml.safe_load(open('configs/config_multi_dim_no_aug/config_' + str(d) + '.yaml', 'r'))
    else:    
        params = yaml.safe_load(open('configs/config_multi_dim/config_' + str(d) + '.yaml', 'r'))

    print('Estimating density of {}-dimensional \'indep_GMM\' model'.format(d))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    # Label Convention
    real_lb = 1.0
    fake_lb = 0.0

    # Random seed
    setup.setup_seed(params['seed'])

    '''
    Model specification and data generation
    '''
    # real model
    real_model = setup.indep_GMM(d)

    # reference model
    ref_cov = stats.Covariance.from_diagonal([params['var_ref']**2]*d)
    ref_model = stats.multivariate_normal(mean = [0]*d, cov = ref_cov)

    # sample training data
    original_data = real_model.rvs(params['sample_size']).reshape(params['sample_size'], d)
    if no_augmentation:
        training_data_real = original_data
        training_data_ref = ref_model.rvs(params['nu']*params['sample_size']).reshape(params['nu']*params['sample_size'], d)
    else:
        aug_data = ref_model.rvs(int(params['rho']*params['sample_size']))
        training_data_real = np.concatenate((original_data, aug_data), axis = 0)
        training_data_ref = ref_model.rvs(params['nu']*int((1 + params['rho'])*params['sample_size'])).reshape(params['nu']*int((1 + params['rho'])*params['sample_size']), d)
    # generate evaluation data
    eva_points = real_model.rvs(params['eva_size']).reshape(params['eva_size'], d)
    real_pdf = real_model.pdf(eva_points)
    ref_pdf = ref_model.pdf(eva_points)
    print('Data generated and training starts. Time: {}'.format(time.time()-start_time))
    
    '''
    Network training
    '''
    for repetition in range(params['repetition_num']):
        model_net= network_training(repetition==0)
        model_pdf = setup.to_flattened_numpy(model_net(torch.from_numpy(eva_points).to(device).float()))
        if no_augmentation:
            est_pdf = model_pdf
        else:
            est_pdf = (1 + params['rho'])*model_pdf - params['rho']*ref_pdf

        # evaluation
        spearman_stat = stats.spearmanr(est_pdf, real_pdf).statistic
        if repetition == 0:
            evaluation = [spearman_stat]
        else:
            evaluation.append(spearman_stat)

        print('Network {} trained, time: {}'.format(repetition+1, time.time()-start_time))

    np.savetxt(params['output_dir'], evaluation, delimiter=",")
    print('Evaluatation saved. The sumary is as follows:')
    print(np.mean(evaluation), np.std(evaluation))
    print('Simulation finished with total time: {}'.format(time.time()-start_time))