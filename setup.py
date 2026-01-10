import numpy as np
import random
from scipy import stats
import torch
import torch.nn as nn

'''
Models
'''
class network_FNN(nn.Module):
    '''
    Model network: A ReLU neural newtork for density estimation
    An additional activation function is applied to the output, such that the output is bounded (or positive)
    Depth = nd + 1, width = nw
    Params:
        dim: input dimension
        nd: depth - 1
        nw: width
        output_scale: constant prefactor to rescale output
        unbounded: Boolean value that represents applying Sigmoid or ReLU to output
    '''
    def __init__(self, dim, nd, nw, output_scale = 1, bounded = True):
        super(network_FNN, self).__init__()
        # Network
        modules = [nn.Sequential(nn.Linear(dim, nw), nn.LeakyReLU(0.1, inplace = True))]

        for _ in range(nd):
            modules.append(nn.Sequential(nn.Linear(nw, nw), nn.LeakyReLU(0.1, inplace = True)))

        if bounded:
            self.main = nn.Sequential(*modules, nn.Linear(nw, 1), nn.Sigmoid())
        else:
            self.main = nn.Sequential(*modules, nn.Linear(nw, 1), nn.ReLU(inplace = True))

        self.output_scale = output_scale

    def forward(self, input):
        output = self.main(input)*self.output_scale
        return output
    
class indep_GMM(object):
    '''Independent Gaussian mixture model'''
    def __init__(self, dim, centers = [-1, 0, 1], sigma = 0.1, weights = None):
        self.dim = dim
        self.centers = centers
        self.sigma = sigma
        if weights is None:
            weights = [1 for _ in centers]
        if len(weights) != len(centers):
            raise(ValueError(f'There are {len(centers)} centers and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        self.num_components = len(centers)
        
    def pdf(self, x):
        assert x.shape[1]==self.dim
        size = len(x)
        density = []
        for i in range(self.dim):
            density_mat = np.zeros((self.num_components, size))
            for j in range(size):
                for k in range(self.num_components):
                    density_mat[k,j] = stats.norm.pdf(x[j,i], loc=self.centers[k], scale=self.sigma)
            density.append(np.mean(density_mat, axis=0))
        density = np.stack(density)
        return np.prod(density, axis=0)

    def rvs(self, size):
        center_choices = np.random.choice(self.num_components, size=self.dim*size, p = self.weights)
        rvs = np.asarray([np.random.normal(self.centers[i], self.sigma) for i in center_choices])
        return rvs.reshape(size, self.dim)
    
class involute_mdoel(object):
    '''Distribution model of involute'''
    def __init__(self,  n_full = 100000, theta = 2*np.pi, scale = 2, sigma = 0.4):
        self.n_full = n_full
        r_list = np.linspace(0, theta, n_full)
        self.data_center = np.vstack((r_list*np.sin(scale*r_list), r_list*np.cos(scale*r_list))).T
        self.full_data = self.data_center + np.random.normal(0, sigma ,size=(n_full, 2))
        self.const_pr = 1./(np.sqrt(2*np.pi)*sigma)
        self.const_de = 2*sigma**2

    def pdf(self, x):
        pdf = [np.mean(self.const_pr*np.exp(-np.sum((np.tile(point, [self.n_full, 1]) - self.data_center)**2, axis=1)/self.const_de)) for point in x]
        return np.asarray(pdf)
    
    def rvs(self, size):
        rvs = data_loader(size, self.full_data)
        return rvs
    
class mixture_model(stats.rv_continuous):
    '''General mixture distribution model'''
    def __init__(self, submodels, weights = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        
    def pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x) * weight
        return pdf

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        rvs = np.asarray([self.submodels[i].rvs() for i in submodel_choices])
        return rvs
    
def cal_cov(theta, sx = 1, sy = 0.4**2):
    '''Calculate covariance in octagan_GMM'''
    Scale = np.array([[sx, 0], [0, sy]])
    c, s = np.cos(theta), np.sin(theta)
    Rotation = np.array([[c, -s], [s, c]])
    product = Rotation.dot(Scale)
    cov = product.dot(product.T)
    return cov

# Outliner dataset (https://shebuti.com/outlier-detection-datasets-odds/)
class outlier_dataset(object):
    def __init__(self, data_name):
        data_path = 'datasets/{}/data.npz'.format(data_name)
        data_dic = np.load(data_path)

        self.X_train, self.X_test, self.label_test = self.normalize(data_dic)
        self.sample_size = self.X_train.shape[0]

        self.dim = self.X_train.shape[1]
        self.test_size = self.X_test.shape[0]

        self.mean = np.mean(self.X_train, axis = 0)
        self.std = np.std(self.X_train, axis = 0)

    def normalize(self,data_dic):
        data = data_dic['arr_0']
        label = data_dic['arr_1']
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        label_test = label[-N_test:]
        data_train = data[0:-N_test]
        return data_train, data_test, label_test

    def load_all(self):
        return self.X_train
    
    def test(self):
        return self.X_test, self.label_test

'''
Utils
'''
def data_loader(load_size, data_set):
    '''Data loader'''
    index_loaded = random.sample(range(data_set.shape[0]), load_size)
    points = data_set[index_loaded]
    return points

def get_prob(ref_pdf, mod_pdf, nu):
    '''Generate the predicted label'''
    return torch.div(mod_pdf, torch.add(input = mod_pdf, other = ref_pdf, alpha = nu))

def setup_seed(seed):
    '''Set random seed'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def to_flattened_numpy(x):
    '''Flatten a torch tensor "x" and convert it to numpy'''
    return x.detach().cpu().numpy().reshape((-1,))

def truncation(values, thershold = 10**(-16)):
    '''Truncate small values'''
    return [max(thershold, val) for val in values]

'''
Evaluation Functions
'''
def create_grid_data(x_min, x_max, y_min, y_max, x_n, y_n):
    '''Create grid data for evaluation and plot'''
    line_x = np.linspace(x_min, x_max, x_n)
    line_y = np.linspace(y_min, y_max, y_n)
    mesh_x, mesh_y = np.meshgrid(line_x, line_y)
    ravel_x = mesh_x.ravel()
    ravel_y = mesh_y.ravel()
    data_grid = np.vstack((ravel_x, ravel_y)).T # shape: (x_n*y_n, 2)
    return ravel_x, ravel_y, data_grid

def create_eva_data(x_interval, y_interval, precision, real_model):
    '''Generate evaluation data'''
    x_min, x_max = x_interval[0], x_interval[1]
    y_min, y_max = y_interval[0], y_interval[1]
    x_n = int((x_max - x_min))*precision + 1 # divide unit(say, interval [0,1]) into 'precision'(say, 100) subintervals
    y_n = int((y_max - y_min))*precision + 1
    eva_x, eva_y, eva_points = create_grid_data(x_min, x_max, y_min, y_max, x_n, y_n)
    real_pdf = real_model.pdf(eva_points)
    return x_n*y_n, eva_x, eva_y, eva_points, real_pdf

def cal_mom(pdf, x, y, piece_area):
    '''Calculate 1st and 2nd moments'''
    mom_1 = [np.sum(np.multiply(pdf, x)), np.sum(np.multiply(pdf, y))]
    mom_2 = [np.sum(np.multiply(pdf, np.square(x))), np.sum(np.multiply(pdf, np.multiply(x, y))), np.sum(np.multiply(pdf, np.square(y)))]
    mom_1 = [term*piece_area for term in mom_1]
    mom_2 = [term*piece_area for term in mom_2]
    return mom_1, mom_2

def cal_div(pdf_1, pdf_2, piece_area):
    '''Calculate KL, inverse KL and JS divergences'''
    log_density_ratio = np.log(pdf_1/pdf_2)
    kl1 = piece_area*np.sum(np.multiply(pdf_1, log_density_ratio))
    kl2 = piece_area*np.sum(np.multiply(pdf_2, -log_density_ratio))
    js = (kl1+kl2)/2
    return [kl1, kl2, js]

def get_eva(eva_x, eva_y, real_pdf, model_pdf, precision, cal_real_mom = False):
    '''Estimate moments and divergences on a 2D rectangle to evaluate the density estimation'''
    piece_area = (1/precision)**2

    model_mom_1, model_mom_2 = cal_mom(model_pdf, eva_x, eva_y, piece_area)
    L2_distance = np.sqrt(piece_area*np.sum(np.square(model_pdf-real_pdf)))
    div = cal_div(real_pdf, model_pdf, piece_area) # KL(true||model), KL(model||true), and JS

    evaluation = [*model_mom_1, *model_mom_2, L2_distance, *div]
    
    if cal_real_mom:
        # calculate real moments
        real_mom_1, real_mom_2 = cal_mom(real_pdf, eva_x, eva_y, piece_area)
        print('real 1st momment:', real_mom_1, 'real 2nd momment:', real_mom_2)

    return evaluation

def precision_at_K(score, label):
    rank = stats.rankdata(score)
    num_outlier = np.sum(label)
    num_correct = sum((item[0] <= num_outlier) and (item[1] == 1) for item in zip(rank, label))
    precision = num_correct/num_outlier
    return precision