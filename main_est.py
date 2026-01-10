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

        current_loss = real_loss + fake_loss

        if detail and (epoch+1)%params['log_step'] == 0:
            print('epoch: {}, loss: {}'.format(epoch + 1, current_loss))

        net_optimizer.step()
    return model_net

'''
Main estimation function
'''
if __name__ == '__main__':
    start_time = time.time()

    # Dimension
    d = 2 

    # Parameters
    # Type of real distribution, can be 'indep_GMM', 'octagon_GMM', or 'involute'
    model_type = 'indep_GMM'
    rho = 0.3
    params = yaml.safe_load(open('configs/config_' + model_type + '.yaml', 'r'))

    print('Estimating density of model:', model_type)
    print('Using rho =', rho)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    # Label Convention
    real_lb = 1.0
    fake_lb = 0.0

    # random seed
    setup.setup_seed(params['seed'])

    '''
    Model specification
    '''
    # real model
    if model_type == 'indep_GMM':
        real_model = setup.indep_GMM(d)
    elif model_type == 'octagon_GMM':
        n_components = 8
        radius = 3
        real_model_list = []
        for idx in range(n_components):
            theta = 2*np.pi*idx/float(n_components)
            real_model_list.append(stats.multivariate_normal([radius*np.cos(theta), radius*np.sin(theta)], setup.cal_cov(theta)))

        real_model = setup.mixture_model(real_model_list)
    elif model_type == 'involute':
        real_model = setup.involute_mdoel()
    else:
        raise ValueError('Wrong model name: the model_type should be \'indep_GMM\', \'octagon_GMM\', or \'involute\'.')
    
    # reference model
    ref_cov = stats.Covariance.from_diagonal([params['var_ref']**2]*d)
    ref_model = stats.multivariate_normal(mean = [0]*d, cov = ref_cov)

    '''
    Data Generation
    '''
    # sample training data
    original_data = real_model.rvs(params['sample_size']).reshape(params['sample_size'], d)
    if rho == 0:
        training_data_real = original_data
    else:
        aug_data = ref_model.rvs(int(rho*params['sample_size']))
        training_data_real = np.concatenate((original_data, aug_data), axis = 0)
    training_data_ref = ref_model.rvs(params['nu']*int((1 + rho)*params['sample_size'])).reshape(params['nu']*int((1 + rho)*params['sample_size']), d)
    # generate evaluation data
    eva_size, eva_x, eva_y, eva_points, real_pdf = setup.create_eva_data(params['x_interval'], params['y_interval'], params['precision'], real_model)
    ref_pdf = ref_model.pdf(eva_points)
    print('Data generated and training starts. Time: {}'.format(time.time()-start_time))
    
    '''
    Network training
    '''
    for repetition in range(params['repetition_num']):
        model_net = network_training(repetition==0)

        model_pdf = model_net(torch.from_numpy(eva_points).to(device).float()).detach().cpu().numpy().reshape(eva_size)
        est_pdf = (1+rho)*model_pdf - rho*ref_pdf
        est_pdf = np.array(setup.truncation(est_pdf))

        # evaluation
        if repetition == 0:
            # save model pdf for plot
            np.savetxt(params['output_dir'] + 'est_pdf_rho=' + str(rho) + '.csv', est_pdf, delimiter=",")
            print('Model plot points saved.')

            evaluation = np.array(setup.get_eva(eva_x, eva_y, real_pdf, est_pdf, params['precision'], cal_real_mom = True))
        else:
            eva_temp = np.array(setup.get_eva(eva_x, eva_y, real_pdf, est_pdf, params['precision']))
            evaluation = np.vstack((evaluation, eva_temp))

        print('Network {} trained, time: {}'.format(repetition+1, time.time()-start_time))

    np.savetxt(params['output_dir'] + 'evaluation_rho=' + str(rho) + '.csv', evaluation, delimiter=",")
    print('Evaluatation saved. The sumary is as follows:')
    print(np.mean(evaluation, axis = 0), np.std(evaluation, axis = 0))
    print('Simulation finished with total time: {}'.format(time.time()-start_time))