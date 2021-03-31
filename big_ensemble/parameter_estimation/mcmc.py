import numpy as np
import torch
import gpytorch
import emcee
from numpy.random import uniform
import scipy

"""
Generate samples from the posterior distribution. 
"""

x = np.load('data/x.npy')
y = np.load('data/y.npy')

train_x = torch.from_numpy(x.T)
train_y = torch.from_numpy(y)

train_x = train_x.cuda()
train_y = train_y.cuda()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims = train_x.shape[1], nu = 1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

noise = torch.ones_like(train_y)*2.5e-4
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    noise = noise, learn_additional_noise=False)
model = ExactGPModel(train_x, train_y, likelihood)

model = model.cuda()
likelihood = likelihood.cuda()

model.eval()
likelihood.eval()

state_dict = torch.load('data/model_state.pth')
model.load_state_dict(state_dict)
errors_std = np.load('data/errors_std.npy')



with torch.no_grad(), gpytorch.settings.fast_pred_var():

    # Log of the posterior probability density
    def f(x):

        x = np.array(x[np.newaxis,:], np.float32)
        pred = likelihood(model(torch.from_numpy(x).cuda()))

        # Predicted mean and variance
        m1 = pred.mean.cpu().item() * errors_std
        m1 = max(0., m1)
        v1 = pred.variance.cpu().item() * errors_std

        # Penalize out of bounds parameters
        x[np.logical_and(x > -1.72, x < 1.72)] = 0.
        m1 += 100.*(x**4).sum(axis = 1)
        
        # Likelihood mean and variance
        m2 = 0.
        v2 = 100.**2

        # Compute the probabiltiy given surrogate uncertainty
        C = ((v1*v2) / (v1 + v2)**2) * (m1 - m2)**2
        C += (v1*v2 / (v1 + v2)) * np.log(2*np.pi*(v1 + v2))
        D = 2.*(v1*v2 / (v1 + v2))
        p = -C / D
        
        return p

   
    # Number of parallel chains
    nwalkers = 20
    # Initial guess for each chain
    x0 = np.array(uniform(-1.72, 1.72, size = (nwalkers,6)),
                  dtype = np.float32)
    # The ensemble sampler
    sampler = emcee.EnsembleSampler(nwalkers, 6, f)
    # Burn in
    state = sampler.run_mcmc(x0, 5000, progress = True)
    sampler.reset()
    # Then sample
    sampler.run_mcmc(state, 20000, progress = True)


samples = sampler.get_chain(flat=True)

# Convert back to non-normalized coordinates
param_means = np.load('data/param_means.npy')
param_stds = np.load('data/param_stds.npy')
samples *= param_stds
samples += param_means

# Save the samples
np.save('data/samples', samples)
