from config import *
from input_signal import *
import numpy as np

class FrontEnd:
    def __init__(self, config, input_signal):
        signal_length = config.signal_length
        self.config = config
        self.input_signal = input_signal
        self.input_signal.process()
        self.sampling_period = signal_length / config.bins
        self.dft_results = np.zeros(max(config.bins))
        self.compute_delays()

    def process(self):
        signal = self.input_signal.time_signal
        for stage in range(config.bins_nb):
            stage_sampling = self.sampling_period[stage]
            for d in self.delays:
                # subsample the signal
                subsampled_signal = np.sqrt(stage_sampling) * signal[::stage_sampling] * self.window(stage_sampling)
                transformed = np.fft.fft(subsampled_signal)

    def compute_delays(self):
        if (config.noisy or config.apply_window_var):
            if config.need_to_use_ml_detection():
               self.delays = np.random.uniform(0, signal_length, config.get_delays_nb)
            else:
                # uses FFAST_Search
                self.delays = np.random.uniform(0, signal_length, config.get_delays_per_bunch_nb)
                for i in range(config.get_delays_per_bunch_nb):
                    self.delays[i] += i * 2 ** i
                self.delays %= signal_length
        else:
            self.delays = np.array(list(range(self.config.get_delays_nb)))


    def window(self, i):
        if not self.config.apply_window:
            return 1
        else:
            # Blackmann-Nuttall window
            a = [0.3635819, -0.4891775, 0.1365995, -0.0106411]
            return 2 * (a[0] + sum([a[j] * np.cos(2*np.pi * j * i) / (self.config.signal_length - 1) for j in range(1,4)]))
