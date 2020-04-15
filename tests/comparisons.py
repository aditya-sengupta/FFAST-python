import sys
sys.path.append("..")
from ffast import *

from matplotlib import pyplot as plt
import numpy as np
import time

params = make_args()
params.primes = [2, 3]
params.prime_powers = [7, 5]
params.bins = [i**j for i,j in zip(params.primes, params.prime_powers)]
params.length = np.prod(params.bins)
params.iterations = 50
params.snr = 6
params.sparsity = 1
methods = ['ml', 'kay', 'kay2', 'new']
times = []

config = Config(params)
input_signal = ExperimentInputSignal(config)
input_signal.process()

for m in methods:
    ti = time.time() * 1000
    params.bin_processing_method = m
    chains_and_delays = {'ml': (120, 1), 'kay': (14, 2), 'kay2': (14, 2), 'new': (1, 2)}
    params.chains, params.delays = chains_and_delays[m]
    config = Config(params)
    config.compute_params()
    output_signal = ExperimentOutputSignal(config, input_signal)
    ffast = FFAST(config, input_signal, output_signal)
    ffast.process()
    output_signal.process()
    tf = time.time() * 1000
    times.append(tf - ti)
    print('\n')
    ffast.display_results(tf - ti)
    output_signal.check_full_recovery()
    print(output_signal.statistics())
