import numpy as np
import matplotlib.pyplot as plt

# Specificatiion
data_folder = 'data/'
# methods for density estimation
method_list = ['real_density/', 'DANNCE/', 'NNCE/', 'Roundtrip/', 'Two-stage/']
# Type of real distribution, can be 'indep_GMM', 'octagon_GMM', or 'involute'
model_list = ['indep_GMM', 'octagon_GMM', 'involute']
precision = 100 # precision of evaluation, as per main_est
n = 100 + 1 # Roundtrip

# specify evaluation parameters
class eva_para():
    def __init__(self, model_type, method_type = None):
        self.model_type = model_type
        self.method_type = method_type
        if model_type == 'indep_GMM':
            # evaluation setting
            x_interval = [-1.5, 1.5]
            y_interval = [-1.5, 1.5]
        elif model_type == 'octagon_GMM':
            # evaluation setting
            x_interval = [-5, 5]
            y_interval = [-5, 5]
        elif model_type == 'involute':
            # evaluation setting
            x_interval = [-6, 5]
            y_interval = [-5, 5]
        else:
            raise ValueError('Wrong model name: the model_type should be \'indep_GMM\', \'octagon_GMM\', or \'involute\'.')
        self.x_interval = x_interval
        self.y_interval = y_interval
        
    def x_min(self):
        return self.x_interval[0]
    def x_max(self):
        return self.x_interval[1]
    def y_min(self):
        return self.y_interval[0]
    def y_max(self):
        return self.y_interval[1]
    def x_n(self):
        if self.method_type == 'Roundtrip/':
            return n
        else:
            return int((self.x_max() - self.x_min()))*precision + 1
    def y_n(self):
        if self.method_type == 'Roundtrip/':
            return n
        else:
            return int((self.y_max() - self.y_min()))*precision + 1

# read the data
data_list = []
for model in model_list:
    for method in method_list:
        pdf = np.genfromtxt(data_folder + method + model + '.csv', delimiter=",").reshape(eva_para(model, method).y_n(), eva_para(model, method).x_n())
        data_list.append(pdf)

'''
plot
'''
title_list = ['Real Density', 'DANNCE', 'NNCE', 'Roundtrip', 'Two-stage']

fig, axes = plt.subplots(nrows = 3, ncols = 5, figsize=(15,8))
for i, ax in enumerate(axes.flat):
    para = eva_para(model_list[i//5])
    im = ax.imshow(data_list[i], extent=[para.x_min(), para.x_max(), para.y_min(), para.y_max()], cmap='Blues', alpha=0.9)
    if i <= 4:
        ax.set_title(title_list[i])
    if (i % 5) == 0:
        fig.colorbar(im, ax=axes.ravel().tolist()[i:i+5])

plt.savefig(data_folder+"2dim_visualization.png")