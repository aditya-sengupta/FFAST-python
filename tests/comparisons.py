import sys
sys.path.append("..")
from ffast import *

from matplotlib import pyplot as plt
import numpy as np
import time

testname = "one_iteration"
params = make_args()
params.primes = [7, 19]
params.prime_powers = [3, 2]
params.bins = [i**j for i,j in zip(params.primes, params.prime_powers)]
params.length = np.prod(params.bins)
params.iterations = 50
params.sparsity = 1
methods = ['kay', 'kay2', 'new'] # ML is out for now
for s in [-14, -16, -18, -20]:
    params.snr = s

    config = Config(params)
    input_signal = ExperimentInputSignal(config)
    input_signal.process()

    delay_sweep = np.arange(2, 11, 1)
    touched_samples = np.zeros((4, delay_sweep.size))
    times = np.zeros((4, delay_sweep.size))
    failures = np.zeros((4, delay_sweep.size))

    for i, d in enumerate(delay_sweep):
        for j, m in enumerate(methods):
            ti = time.time() * 1000
            params.bin_processing_method = m
            chains_and_delays = {'ml': (120, 1), 'kay': (14, 2), 'kay2': (14, 2), 'new': (1, 2)}
            params.chains, params.delays = chains_and_delays[m][0], d
            config = Config(params)
            config.compute_params()
            output_signal = ExperimentOutputSignal(config, input_signal)
            ffast = FFAST(config, input_signal, output_signal)
            ffast.process()
            output_signal.process()
            tf = time.time() * 1000
            print('\n')
            output_signal.check_full_recovery()
            ffast.set_results(tf - ti)
            touched_samples[j][i] = ffast.proportion
            times[j][i] = ffast.time_per
            failures[j][i] = output_signal.binning_failures_nb

    f, (ax1, ax2) = plt.subplots(1, 2)
    for i, m in enumerate(methods):
        ax1.plot(delay_sweep, touched_samples[i], label=m)
    ax1.legend()
    ax1.set_title("Touched samples, SNR dB = {}".format(params.snr))
    ax1.set_xlabel("Delays")

    for i, m in enumerate(methods):
        ax2.plot(delay_sweep, times[i], label=m)
    ax2.legend()
    ax2.set_title("Times, SNR dB = {}".format(params.snr))
    ax2.set_xlabel("Delays")
    plt.savefig('./results/test_' + testname + "_dB_" + str(params.snr) + ".pdf")

    print("Binning failure locations")
    print(np.where(failures != 0))
