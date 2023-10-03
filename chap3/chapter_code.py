# %% Setup
import numpy as np
import torch
import pyro
from pyro import distributions as dist
from pyro.ops import stats

import matplotlib.pyplot as plt
import arviz as az

# %% Create test data

rng = np.random.default_rng()

grid = torch.linspace(0, 1, 1000)
prior = torch.ones((1000,))
likelihood = dist.Binomial(total_count=9, probs=grid).log_prob(torch.tensor(6.)).exp()
posterior = likelihood * prior
posterior /= posterior.sum()

# %% Sample from posterior

samples = dist.Empirical(grid, posterior.log()).sample(torch.Size([10000]))
plt.scatter(np.arange(len(samples)), samples, alpha=0.4, edgecolor="none")
plt.xlabel("Sample number")
plt.ylabel("Probability of water")
plt.show()

# %% Plot density

plt.figure()
az.plot_dist(samples)
plt.show()
