import numpy as np
import torch
import gpytorch

"""
Train the surrogate model on all data.
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

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.3)
# Loss function 
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 1500
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print(loss)
    #print(model.covar_module.base_kernel.lengthscale.tolist())
    optimizer.step()

model.eval()
likelihood.eval()

# Save the surrogate model params
torch.save(model.state_dict(), 'data/model_state.pth')
