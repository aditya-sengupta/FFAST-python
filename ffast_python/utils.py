import numpy as np

norm_squared = lambda x: np.inner(x.conjugate(), x)
positive_mod = lambda a, b: (a % b) + b * (a % b < 0)

def coprime(bins):
    return all([all([i == j or np.gcd(bins[i], bins[j]) == 1 
    for i in range(len(bins))]) for j in range(len(bins))])