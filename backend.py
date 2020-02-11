from config import *
from frontend import *
from binprocessor import *
import numpy as np

class BackEnd:
    def __init__(self, config, frontend):
        self.observation_matrix = frontend.observation_matrix
        changed = np.ones(config.bins_sum, dtype=bool)
        self.config = config
        self.frontend = frontend
        self.decoded_frequencies = 

    def process(self):
        binprocessor = BinProcessor(self.config, self.frontend.delays, self.observation_matrix)
        singleton_found = True
        while singleton_found:
            singleton_found = False
            for stage in range(config.bins_nb):
                self.bin_absolute_index = config.bin_offset[stage]
                for bin_relative_index in range(config.bin_size[stage]):
                    binprocessor.adjust_to(self.bin_absolute_index, bin_relative_index, stage)
                    if changed[self.bin_absolute_index] and binprocessor.is_singleton() and 
