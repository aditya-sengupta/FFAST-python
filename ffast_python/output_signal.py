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
        if self.config.signal_length_original != self.config.signal_length:
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

        self.backend.decoded_frequencies = decoded_frequencies

    def set_backend(self, new_backend):
        self.backend = new_backend

    def check_full_recovery(self):
        missed_locations = set()
        for f in self.input_signal.nonzero_freqs:
            if f not in self.backend.decoded_frequencies:
                missed_locations.add(f)
        
        if missed_locations:
            binning_failure = False
            for stage in range(len(self.config.bins)):
                bin_status = np.zeros((self.config.bins[stage],), dtype=int)
                for i in range(len(missed_locations)):
                    bin_status[i % self.config.bins[stage]] += 1
                if all(bin_status != 1):
                    binning_failure = True
                    break
                
        if binning_failure:
            self.binning_failures_nb += 1
            missed_locations_iterations_nb = 0
            
        