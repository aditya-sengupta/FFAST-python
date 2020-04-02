from .config import *
from .frontend import *
from .binprocessor import *
import numpy as np
import pdb

class BackEnd:
    def __init__(self, config, frontend):
        self.observation_matrix = frontend.observation_matrix
        self.changed = np.ones(config.bins_sum, dtype=bool)
        self.config = config
        self.frontend = frontend
        self.decoded_frequencies = {}
        self.real_freq_inds = []

    def initialize(self):
        self.decoded_frequencies = {}
        self.changed = np.ones(self.config.bins_sum, dtype=bool)
        self.observation_matrix = self.frontend.observation_matrix

    def process(self):
        self.initialize()
        binprocessor = BinProcessor(self.config, self.frontend.delays, self.observation_matrix)

        singleton_found = True
        stop_peeling = False

        # iterative decoding
        while singleton_found and not stop_peeling:
            singleton_found = False

            # go over each stage
            for stage in range(len(self.config.bins)):
                stage_head_bin = self.config.bin_offsets[stage]

                # go over the bins in each stage
                for bin_relative_index in range(self.config.bins[stage]):
                    self.bin_absolute_index = stage_head_bin + bin_relative_index

                    binprocessor.adjust_to(self.bin_absolute_index, bin_relative_index, stage)

                    if binprocessor.is_singleton():
                        # pdb.set_trace()
                        print('found a singleton at {} -- stage {} -- bin {}'.format(binprocessor.location, stage, bin_relative_index))

                    if self.changed[self.bin_absolute_index] and binprocessor.is_singleton() and (binprocessor.location not in self.decoded_frequencies.keys()):
                        singleton_found = True
                        self.decoded_frequencies[binprocessor.location] = binprocessor.amplitude
                    
                        self.peel_from(binprocessor)
                    
                    # since we have removed the singleton from this bin, it is a zeroton now
                    # so we do not need to look at it again
                    self.changed[self.bin_absolute_index] = False
                    if len(self.decoded_frequencies) == self.config.signal_sparsity_peeling:
                        stop_peeling = True
                        break

                # after the break, you're out here
                if stop_peeling:
                    break
        
        if self.config.apply_window_var:
            self.get_clustered_freqs()
        else:
            self.real_freq_inds = list(self.decoded_frequencies.keys())

    def get_hashed_bin(self, location, stage):
        return (location % self.config.bins[stage]) + self.config.bin_offsets[stage]

    def peel_from(self, binprocessor):
        # go over stages
        for stage in range(len(self.config.bins)):
            # this is the bin it hashes to
            hash_int = self.get_hashed_bin(binprocessor.location, stage)

            # this can be made as a vector subtraction
            # for delay_index in range(self.config.delays_nb):
            #     self.observation_matrix[hash_int][delay_index] -= binprocessor.signal_vector[delay_index]
            # self.observation_matrix[:, hash_int] -= binprocessor.signal_vector

            self.observation_matrix[:, hash_int] -= self.calculate_signature_at_stage(binprocessor, stage)
            self.changed[hash_int] = True

    def calculate_signature_at_stage(self, binprocessor, stage):
        signal_vector = np.zeros(self.frontend.get_max_delays(), dtype=np.complex128)

        delays = self.frontend.get_delays_for_stage(stage)
        dirvector = np.exp(1j * binprocessor.signal_k * binprocessor.location * np.array(delays))
        signal_vector[0:len(delays)] = binprocessor.amplitude * dirvector
        return signal_vector

    # this is for off-grid
    # off-grid is not implemented in this version of the code
    # def get_clustered_freqs(self):
    #     total_energy = 0
    #     peak = 0
    #     amplitude = 0

    #     ratio = config.signal_length_original / config.get_signal_length
    #     nb = config.signal_sparsity * 100

    #     for f in sorted(self.decoded_frequencies):
    #         energy = self.decoded_frequencies[f] ** 2
    #         peak += energy * f
    #         total_energy += energy
    #         amplitude += self.decoded_frequencies[f] * energy
    #         if all([f + i not in self.decoded_frequencies for i in [1, 2]]):
    #             weighted_avg_freq = ratio * peak / total_energy
    #             int_wtd_avg_freq = int(np.round(weighted_avg_freq))
    #             self.real_freq_inds.append(weighted_avg_freq)
    #             self.decoded_frequencies[int_wtd_avg_freq] = 0
    #             for i in range(nb):
    #                 self.decoded_frequencies[int_wtd_avg_freq] += self.frontend.input_signal.time_signal[i] * np.exp(-2*np.pi * 1j * (peak / total_energy)) / config.signal_length

    #             self.decoded_frequencies[int_wtd_avg_freq] = np.sqrt(total_energy) * np.exp(1j * 2 * np.pi * self.decoded_frequencies[int_wtd_avg_freq])

