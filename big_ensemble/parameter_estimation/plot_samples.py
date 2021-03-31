import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gpytorch
import emcee
from scipy.stats import beta
from multiprocessing import Pool
from surrogate import model, likelihood
from numpy.random import uniform
from mcmcplot import mcmcplot as mcp
import mcmcplot
import matplotlib


plt.rcParams.update({'font.size': 12})

### Evaluate the model at the samples
#######################################################################

# Load mcmc samples
samples = np.load('data/smooth3.npy')
#prior_means = np.load('data/prior_means.npy')
#prior_stds = np.load('data/prior_stds.npy')

samples = samples[::1]
print(samples.shape)
fig, axes = plt.subplots(6,6, figsize = (16, 16))
samples[:,0] /= 365.
samples[:,5] /= 1e3

#plt.scatter(samples[:,2], samples[:,4], s = 2)
#plt.show()
#quit()

print(samples[:,-1].min(), samples[:,-1].max())
#quit()

ranges = [
    [0.,4.],
    [-10., 10.],
    [-1.33, 1.33],
    [0.66, 1.33],
    [0., 0.33],
    [50., 2000.]
]

for i in range(6):
    for j in range(i+1,6):
        r = [ranges[j], ranges[i]]
        #H, xedges, yedges = np.histogram2d(samples[:,j], samples[:,i], bins = 25, range = r)

        #extent = [ranges[i][0], ranges[i][1], ranges[j][0], ranges[j][1]]

        #axes[j,i].imshow(H, interpolation='nearest', origin='low', aspect = 'auto', extent = extent, cmap = 'inferno')
        axes[j,i].scatter(samples[::40,i],samples[::40,j], s = 2, c = 'k', alpha = 1)
        #print(i, j)
        axes[j,i].set_xlim(ranges[i])
        axes[j,i].set_ylim(ranges[j])
        axes[i,j].axis('off')

for i in range(6):
    axes[i,i].hist(samples[:,i], bins = 50, range = ranges[i])
    axes[i,i].set_xlim(ranges[i])
    if i < 5:
        axes[i,i].get_xaxis().set_visible(False)


axes[0,0].get_yaxis().set_visible(False)
#axes[0,0].get_xaxis().set_visible(False)

axes[1,0].get_xaxis().set_visible(False)
#axes[1,1].get_xaxis().set_visible(False)
axes[1,1].get_yaxis().set_visible(False)

axes[2,0].get_xaxis().set_visible(False)
axes[2,1].get_xaxis().set_visible(False)
axes[2,1].get_yaxis().set_visible(False)
#axes[2,2].get_xaxis().set_visible(False)
axes[2,2].get_yaxis().set_visible(False)

axes[3,0].get_xaxis().set_visible(False)
axes[3,1].get_xaxis().set_visible(False)
axes[3,1].get_yaxis().set_visible(False)
axes[3,2].get_xaxis().set_visible(False)
axes[3,2].get_yaxis().set_visible(False)
#axes[3,3].get_xaxis().set_visible(False)
axes[3,3].get_yaxis().set_visible(False)

axes[4,0].get_xaxis().set_visible(False)
axes[4,1].get_xaxis().set_visible(False)
axes[4,1].get_yaxis().set_visible(False)
axes[4,2].get_xaxis().set_visible(False)
axes[4,2].get_yaxis().set_visible(False)
axes[4,3].get_xaxis().set_visible(False)
axes[4,3].get_yaxis().set_visible(False)
#axes[4,4].get_xaxis().set_visible(False)
axes[4,4].get_yaxis().set_visible(False)



axes[5,1].get_yaxis().set_visible(False)
axes[5,2].get_yaxis().set_visible(False)
axes[5,3].get_yaxis().set_visible(False)
axes[5,4].get_yaxis().set_visible(False)
axes[5,5].get_yaxis().set_visible(False)




axes[5,0].set_xlabel(r'Melt (ma$^{-1}$)')
axes[5,1].set_xlabel(r'$\Delta T$ ($^{\circ}$C)')
axes[5,2].set_xlabel('SMB Bias')
axes[5,3].set_xlabel(r'$\beta^2$ Scale')
axes[5,4].set_xlabel(r'$\beta^2$ Seasonality Scale')
axes[5,5].set_xlabel(r'$\sigma_{max}$ (kPa)')


axes[1,0].set_ylabel(r'$\Delta T$ ($^{\circ}$C)')
axes[2,0].set_ylabel('SMB Bias')
axes[3,0].set_ylabel(r'$\beta^2$ Scale')
axes[4,0].set_ylabel(r'$\beta^2$ Seasonality Scale')
axes[5,0].set_ylabel(r'$\sigma_{max}$ (kPa)')

plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.tight_layout()
plt.savefig('samples.pdf', dpi = 500)
plt.show()
