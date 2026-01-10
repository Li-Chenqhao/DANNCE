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
    model_net = setup.network_FNN(real_model.dim, params['nd'], params['nw'], 1, False).to(device)
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

        if detail and (epoch+1)%params['log_step'] == 0:
            print('epoch: {}, loss: {}'.format(epoch + 1, current_loss))

        if current_loss < best_loss:
            best_loss = current_loss
            count = 0
        else:
            count += 1
        
        if count == params['stop_threshold']:
            break

        net_optimizer.step()
        scheduler.step()
    return model_net

'''
Main estimation function
'''
if __name__ == '__main__':
    start_time = time.time()

    # Parameters
    dataset_name = 'Shuttle' # name of dataset
    no_augmentation = False # Set as 'False' for data augmentation, and 'True' for estimation without data augmentation
    if no_augmentation:
        params = yaml.safe_load(open('configs/config_{}_no_aug.yaml'.format(dataset_name), 'r'))
    else:    
        params = yaml.safe_load(open('configs/config_{}.yaml'.format(dataset_name), 'r'))

    print('Detecting anomaly for {} dataset'.format(dataset_name))

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
    real_model = setup.outlier_dataset(dataset_name)
    print(real_model.mean, real_model.std)

    # reference model
    ref_cov = stats.Covariance.from_diagonal([params['var_ref']**2]*real_model.dim)
    ref_model = stats.multivariate_normal(mean = real_model.mean, cov = ref_cov)

    # load and sample training data
    original_data = real_model.load_all().reshape(real_model.sample_size, real_model.dim)
    if no_augmentation:
        training_data_real = original_data
        training_data_ref = ref_model.rvs(params['nu']*real_model.sample_size).reshape(params['nu']*real_model.sample_size, real_model.dim)
    else:
        aug_data = ref_model.rvs(int(params['rho']*real_model.sample_size))
        training_data_real = np.concatenate((original_data, aug_data), axis = 0)
        training_data_ref = ref_model.rvs(params['nu']*int((1 + params['rho'])*real_model.sample_size)).reshape(params['nu']*int((1 + params['rho'])*real_model.sample_size), real_model.dim)
    
    # load evaluation data
    test_data, test_label = real_model.test()
    test_data = test_data.reshape(real_model.test_size, real_model.dim)
    ref_pdf = ref_model.pdf(test_data)
    print('Data loaded/generated and training starts. Time: {}'.format(time.time()-start_time))
    
    '''
    Network training
    '''
    evaluation = []
    for repetition in range(params['repetition_num']):
        model_net= network_training(True)
        model_pdf = setup.to_flattened_numpy(model_net(torch.from_numpy(test_data).to(device).float()))
        if no_augmentation:
            est_pdf = model_pdf
        else:
            est_pdf = (1 + params['rho'])*model_pdf - params['rho']*ref_pdf

        # evaluation
        precision = setup.precision_at_K(est_pdf, test_label)
        evaluation.append(precision)

        print('Network {} trained, precision: {}, time: {}'.format(repetition+1, precision, time.time()-start_time))

    np.savetxt(params['output_dir'], evaluation, delimiter=",")
    print('Evaluatation saved. The sumary is as follows:')
    print(np.mean(evaluation), np.std(evaluation))
    print('Experiment finished with total time: {}'.format(time.time()-start_time))