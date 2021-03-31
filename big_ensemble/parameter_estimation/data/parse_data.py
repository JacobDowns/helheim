from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

"""
Normalize the training parameter values and SSE's for training the 
surrogate model.
"""

# Load modeled v. front offsets 
errors = np.loadtxt('errors.txt')
# Load prior samples
params = np.loadtxt('params.txt').T

# Normalize the errors
np.save('errors_std', errors.std())
errors = errors / errors.std()

# Normalize the parameter values
means = []
stds = []
for i in range(6):
    means.append(params[i].mean())
    stds.append(params[i].std())
    params[i] = (params[i] - params[i].mean()) / params[i].std()

means = np.array(means)
stds = np.array(stds)

np.save('param_means', means)
np.save('param_stds', stds)
    
# Appropriate Datatype for gpytorch
params = np.array(params, dtype = np.float32)
errors = np.array(errors, dtype = np.float32)

np.save('x', params)
np.save('y', errors)
