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
        
        if self.backend is None:
            return


        # this is for off-grid
        # off-grid is not implemented in this version of the code
        # decoded_frequencies = {}
        # if self.config.signal_length_original != self.config.signal_length:
        #     for f in self.backend.decoded_frequencies:
        #         if f + 1 in self.input_signal.nonzero_freqs:
        #             decoded_frequencies[f + 1] = self.backend.decoded_frequencies[f]
        #         elif f - 1 in self.input_signal.nonzero_freqs:
        #             decoded_frequencies[f - 1] = self.backend.decoded_frequencies[f]
        #         elif f + 2 in self.input_signal.nonzero_freqs:
        #             decoded_frequencies[f + 2] = self.backend.decoded_frequencies[f]
        #         elif f - 2 in self.input_signal.nonzero_freqs:
        #             decoded_frequencies[f - 2] = self.backend.decoded_frequencies[f]
        #         else:
        #             decoded_frequencies[f] = self.backend.decoded_frequencies[f]
        #     self.backend.decoded_frequencies = decoded_frequencies

        self.check_full_recovery()

    def set_backend(self, new_backend):
        self.backend = new_backend

    def check_full_recovery(self):
        """
        Here we compare the decoded frequencies to input signal frequencies
        to build the statistics.
        """
        # print('decoded frequencies:')
        # print(self.backend.decoded_frequencies)

        missed_locations = set()
        for f in self.input_signal.freqs:
            if f not in self.backend.decoded_frequencies:
                missed_locations.add(f)
        
        # we check if the missed locations are due to binning failure, that is, 
        # due to the sparse graph code having a trapping set, or due to the errors
        # in the noisy recovery.  we do this by going over the missing locations 
        # and look at the residual graph remaining from them.  if there is no 
        # singleton bin, that means that there was a trapping set
        if missed_locations:
            binning_failure = True
            
            # go over each stage
            for stage in range(len(self.config.bins)):
                bin_status = np.zeros((self.config.bins[stage],), dtype=int)
                
                for i in range(len(missed_locations)):
                    bin_status[i % self.config.bins[stage]] += 1

                if any(bin_status == 1):
                    binning_failure = False
                    break
                
            if binning_failure:
                self.binning_failures_nb += 1
        else:
            self.full_recoveries_nb += 1

    def statistics(self):
        return self.full_recoveries_nb, self.binning_failures_nb

            
        