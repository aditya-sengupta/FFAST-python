from config import *
from frontend import *
from binprocessor import *
import numpy as np

class BackEnd:
    def __init__(self, config, frontend):
        self.observation_matrix = frontend.observation_matrix
        self.changed = np.ones(config.bins_sum, dtype=bool)
        self.config = config
        self.frontend = frontend
        self.decoded_frequencies = {}
        self.real_freq_inds

    def process(self):
        binprocessor = BinProcessor(self.config, self.frontend.delays, self.observation_matrix)
        singleton_found = True
        stop_peeling = False
        while singleton_found and not stop_peeling:
            singleton_found = False
            for stage in range(config.bins_nb):
                self.bin_absolute_index = config.bin_offset[stage]
                for bin_relative_index in range(config.bin_size[stage]):
                    binprocessor.adjust_to(self.bin_absolute_index, bin_relative_index, stage)
                    if self.changed[self.bin_absolute_index] and binprocessor.is_singleton() and len(self.decoded_frequencies[binprocessor.location]) == 0:
                        singleton_found = True
                        self.decoded_frequencies[binprocessor.location] = binprocessor.amplitude
                        self.peel_from(binprocessor.location)
                    self.changed[self.bin_absolute_index] = False
                    if len(self.decoded_frequencies) == config.signal_sparsity_peeling:
                        stop_peeling = True
                        break
                # after the break, you're out here
                if stop_peeling:
                    break
        
        if config.apply_window_var:
            self.get_clustered_freqs()
        else:
            for i in self.decoded_frequencies:


    def peel_from(self, location):

