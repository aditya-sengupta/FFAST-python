from .config import *
from .frontend import *
from .binprocessor import *
import numpy as np

class BackEnd:
    def __init__(self, config, frontend):
        self.observation_matrix = frontend.observation_matrix
        self.changed = np.ones(config.bins_sum, dtype=bool)
        self.config = config
        self.frontend = frontend
        self.decoded_frequencies = {}
        self.real_freq_inds = []

    def process(self):
        binprocessor = BinProcessor(self.config, self.frontend.delays, self.observation_matrix)
        singleton_found = True
        stop_peeling = False
        while singleton_found and not stop_peeling:
            singleton_found = False
            for stage in range(len(config.bins)):
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
            self.real_freq_inds = list(self.decoded_frequencies.keys())

    def peel_from(self, location, binprocessor):
        for stage in range(self.config.bins_nb):
            hash_int = (location % self.config.bin_size[stage]) + self.config.bin_offsets[stage]
            for delay_index in range(self.config.delays_nb):
                self.observation_matrix[hash_int][delay_index] -= binprocessor.signal_vector[delay_index]

            self.changed[hash_int] = True

    def get_clustered_freqs(self):
        total_energy = 0
        peak = 0
        amplitude = 0

        ratio = config.signal_length_original / config.get_signal_length
        nb = config.signal_sparsity * 100

        for f in sorted(self.decoded_frequencies):
            energy = self.decoded_frequencies[f] ** 2
            peak += energy * f
            total_energy += energy
            amplitude += self.decoded_frequencies[f] * energy
            if all([f + i not in self.decoded_frequencies for i in [1, 2]]):
                weighted_avg_freq = ratio * peak / total_energy
                int_wtd_avg_freq = int(np.round(weighted_avg_freq))
                self.real_freq_inds.append(weighted_avg_freq)
                self.decoded_frequencies[int_wtd_avg_freq] = 0
                for i in range(nb):
                    self.decoded_frequencies[int_wtd_avg_freq] += self.frontend.input_signal.time_signal[i] * np.exp(-2*np.pi * 1j * (peak / total_energy)) / config.signal_length

                self.decoded_frequencies[int_wtd_avg_freq] = np.sqrt(total_energy) * np.exp(1j * 2 * np.pi * self.decoded_frequencies[int_wtd_avg_freq])

