from .config import *
from .input_signal import *
import numpy as np

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
        for stage in range(len(self.config.bins)):
            stage_sampling = int(self.sampling_period[stage])
            for i, d in enumerate(self.delays):
                # subsample the signal
                self.used_samples = self.used_samples.union(set(range(d, len(signal), stage_sampling)))
                subsampled_signal = np.sqrt(stage_sampling) * signal[d::stage_sampling] * self.window(stage_sampling)
                transformed = np.fft.fft(subsampled_signal)
                s, e = self.config.bin_offsets[stage], self.config.bin_offsets[stage] + self.config.bins[stage]
                self.observation_matrix[i][s:e] = transformed / np.sqrt(self.config.bins[stage])
        self.count_samples_done = True

    def compute_delays(self):
        if (self.config.noisy or self.config.apply_window_var):
            if self.config.need_to_use_ml_detection():
               self.delays = np.random.uniform(0, self.signal_length, self.config.delays_nb)
            else:
                # uses FFAST_Search
                self.delays = np.random.uniform(0, self.signal_length, self.config.delays_per_bunch_nb)
                for i in range(self.config.delays_per_bunch_nb):
                    self.delays[i] += i * 2 ** i
                self.delays %= self.signal_length
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
