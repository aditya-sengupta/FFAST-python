from .config import *
from .input_signal import *
import numpy as np

class FrontEnd:
    def __init__(self, config, input_signal):
        signal_length = config.signal_length
        self.config = config
        self.input_signal = input_signal
        print("Processing input signal")
        self.input_signal.process()
        self.sampling_period = signal_length / config.bins

        print("Defining delays")
        self.compute_delays()
        self.observation_matrix = np.zeros((len(config.bins), len(self.delays)), dtype=np.complex128)
        self.count_samples_done = False

    def process(self):
        signal = self.input_signal.time_signal
        for stage in range(len(self.config.bins)):
            stage_sampling = int(self.sampling_period[stage])
            for d in self.delays:
                # subsample the signal
                subsampled_signal = np.sqrt(stage_sampling) * signal[::stage_sampling] * self.window(stage_sampling)
                transformed = np.fft.fft(subsampled_signal)
                bin_abs_index = self.config.bin_offsets[stage]
                for bin_rel_index in range(self.config.bins[stage]):
                    self.observation_matrix[bin_abs_index][delay_index] = transformed[bin_rel_index] / np.sqrt(self.config.bins[stage])
                    bin_abs_index += 1
        self.count_samples_done = True

    def compute_delays(self):
        if (self.config.noisy or self.config.apply_window_var):
            if self.config.need_to_use_ml_detection():
               self.delays = np.random.uniform(0, signal_length, self.config.delays_nb)
            else:
                # uses FFAST_Search
                self.delays = np.random.uniform(0, signal_length, self.config.delays_per_bunch_nb)
                for i in range(self.config.delays_per_bunch_nb):
                    self.delays[i] += i * 2 ** i
                self.delays %= signal_length
        else:
            print(self.config.delays_nb)
            self.delays = np.array(list(range(self.config.delays_nb)))


    def window(self, i):
        if not self.config.apply_window_var:
            return 1
        else:
            # Blackmann-Nuttall window
            a = [0.3635819, -0.4891775, 0.1365995, -0.0106411]
            return 2 * (a[0] + sum([a[j] * np.cos(2*np.pi * j * i) / (self.config.signal_length - 1) for j in range(1,4)]))
