from .config import *
from .input_signal import *
import numpy as np
import pdb

class FrontEnd:
    def __init__(self, config, input_signal):
        self.signal_length = config.signal_length
        self.config = config
        self.input_signal = input_signal
        self.input_signal.process()
        self.sampling_period = self.signal_length / config.bins
        self.used_samples = set()
        self.compute_delays()
        self.observation_matrix = np.zeros((len(self.delays), sum(config.bins)), dtype=np.complex128)
        self.count_samples_done = False

    def process(self):
        signal = self.input_signal.time_signal
        # go over each stage
        for stage in range(len(self.config.bins)):
            # sampling period at the stage
            stage_sampling = int(self.sampling_period[stage])
            # go over the delays
            for i, d in enumerate(self.delays):
                # print('frontend delay: {}'.format(d))

                # subsample the signal
                self.used_samples = self.used_samples.union(set(range(d, len(signal), stage_sampling)))

                # delays should wrap around the signal length
                subsampling_points = np.arange(d, d+self.config.signal_length, stage_sampling) % self.config.signal_length
                subsampled_signal = np.sqrt(stage_sampling) * signal[subsampling_points] * self.window(stage_sampling)
                transformed = np.fft.fft(subsampled_signal)

                s, e = self.config.bin_offsets[stage], self.config.bin_offsets[stage] + self.config.bins[stage]

                # print(s)
                # print(e)
                # print(i)
                # print(self.observation_matrix.shape)

                self.observation_matrix[i][s:e] = transformed / np.sqrt(self.config.bins[stage])
        self.count_samples_done = True

    def compute_delays(self):
        if (self.config.noisy or self.config.apply_window_var):
            if self.config.bin_processing_method == 'ml':
               # self.delays = np.random.uniform(0, self.signal_length, self.config.delays_nb)
               self.delays = np.random.choice(self.signal_length, self.config.delays_nb, replace=False)

            elif self.config.bin_processing_method == 'kay':
                # uses FFAST_Search
                delay_roots = np.random.choice(self.signal_length, self.config.chains_nb, replace=False)
                self.delays = np.zeros(self.config.delays_nb)
                
                delay_index = 0
                for chain_index in range(self.config.chains_nb):
                    jump = 2**chain_index
                    for delay_within_chain_index in range(self.config.delays_per_bunch_nb):
                        self.delays[delay_index] = delay_roots[chain_index] + jump*delay_within_chain_index
                        delay_index += 1
                self.delays %= self.signal_length

            elif self.config.bin_processing_method == 'new':
                print('not implemented')
        else:
            self.delays = np.array(list(range(self.config.delays_nb)))
        self.delays = list(map(int, self.delays))

    def get_used_samples_nb(self):
        return len(self.used_samples)

    def window(self, i):
        if not self.config.apply_window_var:
            return 1
        else:
            # Blackmann-Nuttall window
            a = [0.3635819, -0.4891775, 0.1365995, -0.0106411]
            return 2 * (a[0] + sum([a[j] * np.cos(2*np.pi * j * i) / (self.config.signal_length - 1) for j in range(1,4)]))
