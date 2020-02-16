from .config import *
from .backend import *
import numpy as np

class OutputSignal:
    pass

class ExperimentOutputSignal(OutputSignal):
    def __init__(self, config, input_signal):
        self.config = config
        self.binning_failures_nb = 0
        self.full_recoveries_nb = 0
        self.input_signal = input_signal

    def process(self):
        decoded_frequencies = {}
        if self.backend is None:
            return
        if config.signal_length_original != config.signal_length:
            for f in self.backend.decoded_frequencies:
                if f + 1 in self.input_signal.nonzero_freqs:
                    decoded_frequencies[f + 1] = self.backend.decoded_frequencies[f]
                elif f - 1 in self.input_signal.nonzero_freqs:
                    decoded_frequencies[f - 1] = self.backend.decoded_frequencies[f]
        elif f + 2 in self.input_signal.nonzero_freqs:
            decoded_frequencies[f + 2] = self.backend.decoded_frequencies[f]
        elif f - 2 in self.input_signal.nonzero_freqs:
            decoded_frequencies[f - 2] = self.backend.decoded_frequencies[f]
        else:
            decoded_frequencies[f] = self.backend.decoded_frequencies[f]

        backend.decoded_frequencies = decoded_frequencies

    def set_backend(self, new_backend):
        self.backend = new_backend
        