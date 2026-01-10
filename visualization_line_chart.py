import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Specify data location
data_folder = 'results/'
# values of rho and d for plot
rho_list = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 5]
dim_list = [3, 4, 5, 6, 7]

class result_provider():
    def __init__(self, data):
        self.mean = np.nanmean(data, axis = 1)
        std = np.nanstd(data, axis = 1, ddof = 1)
        count = np.sum(~np.isnan(data), axis = 1)

        # 95% confidence interval
        t_crit = stats.t.ppf(0.975, df = count - 1)
        ci95 = t_crit * std / np.sqrt(count)
        self.lower = self.mean - ci95
        self.upper = self.mean + ci95

# read the data
data = result_provider(np.genfromtxt(data_folder + 'varrho/evaluation.csv', delimiter=","))

data_list = []
for d in dim_list:
    data_item = result_provider(data = np.genfromtxt(data_folder + f'varrho_md/evaluation_{d}.csv', delimiter=","))
    data_list.append(data_item)

'''
Plot
'''
plot_list = [0, 1, 2, 3, 4, 5, 6, 7]
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 6))

ax1.plot(plot_list, data.mean)
ax1.fill_between(plot_list, data.lower, data.upper, alpha=0.3)
ax1.set_xticks(plot_list)
ax1.set_xticklabels(rho_list)
ax1.set_xlabel('$\\rho$')
ax1.set_ylabel('$L_{2}$ distance')
ax1.set_title('(a) 2-D case')

lines = []
labels = []

for i, d in enumerate(dim_list):
    line, = ax2.plot(plot_list, data_list[i].mean, label = f'd = {d}')
    ax2.fill_between(plot_list, data_list[i].lower, data_list[i].upper, alpha=0.3, label = "_nolegend_")
    lines.append(line)
    labels.append(f'd = {d}')
ax2.legend(lines, labels)
# ax2.legend(('d = 3', 'd = 4', 'd = 5', 'd = 6', 'd = 7'))
ax2.set_xticks(plot_list)
ax2.set_xticklabels(rho_list)
ax2.set_xlabel('$\\rho$')
ax2.set_ylabel('Spearman rank correlation')
ax2.set_title('(b) Higher dimensional cases')

plt.savefig(data_folder + "line_chart_with_band.png")