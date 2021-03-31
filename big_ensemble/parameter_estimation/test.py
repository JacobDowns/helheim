import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
import gpytorch

"""
Divide the GP into training and test data to see how well it predicts.
"""

plt.rcParams.update({'font.size': 12})

x = np.load('data/x.npy')
y = np.load('data/y.npy')

indexes = np.random.choice(x.shape[1], 2400, replace=False)

train_x = torch.from_numpy(x.T[indexes])
train_y = torch.from_numpy(y[indexes])

test_indexes = np.arange(x.shape[1])
test_indexes = np.delete(test_indexes, indexes)

train_x = train_x.cuda()
train_y = train_y.cuda()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims = train_x.shape[1], nu=1.5))

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



plt.figure(figsize = (12,12))
errors_std = np.load('data/errors_std.npy')

# Plot observed v. modeled test points
with torch.no_grad(), gpytorch.settings.fast_pred_var():

    observed_pred = likelihood(model(torch.from_numpy(x.T).cuda()))
    lower, upper = observed_pred.confidence_region()
    lower = lower.cpu().numpy()*errors_std
    upper = upper.cpu().numpy()*errors_std
    mean =  observed_pred.mean.cpu().numpy()*errors_std
    y *= errors_std
    
    plt.plot([lower[test_indexes], upper[test_indexes]],
             [y[test_indexes], y[test_indexes]], 'k-', alpha = 0.25)
    #plt.scatter(mean, y, s = 20, color = 'b')
    plt.plot([-2.5*errors_std, 5*errors_std], [-2.5*errors_std, 5*errors_std], 'k-')
    plt.xlim([-.5*errors_std, 5*errors_std])
    plt.ylim([0., 5*errors_std])
    plt.scatter(mean[test_indexes], y[test_indexes], s = 20, color = 'r')
    plt.xlabel('Surrogate Predicted SSE')
    plt.ylabel('Modeled SSE')
    plt.grid(True, linestyle = ':')

    plt.savefig('surrogate_fit.pdf', dpi = 500)
    plt.show()
