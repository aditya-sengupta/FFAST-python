import numpy as np

norm_squared = lambda x: np.linalg.norm(x)**2
positive_mod = lambda a, b: (a % b) + b * (a % b < 0)

def coprime(bins):
    return all([all([i == j or np.gcd(bins[i], bins[j]) == 1 
    for i in range(len(bins))]) for j in range(len(bins))])


def get_multiplicative_inverse(a, m, p, n):
    F = p**n
    q = (p-1)*p**(n-1)-1
    ra1 = pow(a, q, F)
    ram = pow(ra1, m, F)
    return ram