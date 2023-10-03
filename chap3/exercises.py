import torch
import pyro
import pyro.distributions as dist
from pyro.ops import stats

import matplotlib.pyplot as plt
import arviz as az

# %% 3M1 - Set up prior and likelihood
grid = torch.linspace(0, 1.0, 1000)
prior = torch.ones((1000,))
likelihood = dist.Binomial(15, probs=grid).log_prob(torch.tensor(8.)).exp()
posterior = likelihood * prior
posterior /= posterior.sum()

# %% 3M2 - Sample from posterior and calculate 90% HDPI
samples = dist.Empirical(grid, posterior.log()).sample(torch.Size([10_000]))
hpdi = stats.hpdi(samples, 0.9)

print(f"estimated p: {samples.mean()}")
print(f"HPDI: [{hpdi[0]:.2f}, {hpdi[1]:.2f}]")

# %% 3M3 - Posterior predictive checks
water = dist.Binomial(15, probs=samples).sample().long()

plt.figure()
az.plot_dist(water, kind="hist")
plt.show()

print(torch.count_nonzero(water == 8) / len(water))

# %% 3M4 - 6 out of 9 water, using previous posterior

water2 = dist.Binomial(total_count=9, probs=samples).sample().long()

plt.figure()
az.plot_dist(water2, kind="hist")
plt.show()

print(torch.count_nonzero(water2 == 6) / len(water2))

# %% 3M5 - Different priors
prior = torch.ones((1000,))
prior[grid < 0.5] = 0

likelihood = dist.Binomial(15, probs=grid).log_prob(torch.tensor(8.)).exp()
posterior = likelihood * prior
posterior /= posterior.sum()

samples = dist.Empirical(grid, posterior.log()).sample(torch.Size([10_000]))
hpdi = stats.hpdi(samples, 0.9)

print(f"estimated p: {samples.mean()}")
print(f"HPDI: [{hpdi[0]:.2f}, {hpdi[1]:.2f}]")

# %% 3M6 - PI range maximum of 0.05

true_p = 0.7
for num_tosses in [15, 25, 50, 100, 250, 500, 1_000, 10_000, 25_000, 50_000]:
    num_expected_water = round(num_tosses * true_p)

    likelihood = (dist.Binomial(total_count=num_tosses, probs=grid)
                  .log_prob(torch.tensor(num_expected_water))
                  .exp())
    posterior = likelihood * prior
    posterior /= posterior.sum()

    samples = dist.Empirical(grid, posterior.log()).sample(torch.Size([10_000]))
    pi = stats.pi(samples, 0.99)
    print(f"Num tosses: {num_expected_water}/{num_tosses}, pi: [{pi[0]:.2f}, {pi[1]:.2f}]")

    if (pi[1] - pi[0]) < 0.05:
        break
