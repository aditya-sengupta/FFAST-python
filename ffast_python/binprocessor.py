from .config import *
import numpy as np
import pdb

class BinProcessor:
    def __init__(self, config, delays, observation_matrix):
        # delays object is set in frontend
        self.config = config
        self.delays = delays
        self.observation_matrix = observation_matrix

        self.signal_length = config.signal_length
        self.delays_nb = config.delays_nb
        self.chains_nb = config.chains_nb
        self.delays_per_bunch_nb = config.delays_per_bunch_nb

        # bin processing method
        self.bin_processing_method = config.bin_processing_method

        # this is w0
        self.signal_k = 2 * np.pi / self.signal_length
        self.signal_invk = 1/self.signal_k

        self.signal_vector = np.zeros(self.delays_nb, dtype=np.complex128)
        self.thresholds = np.ones(len(config.bins), dtype=np.float64)
        self.dirvector = np.zeros(self.delays_nb, dtype=np.complex128)

        self.compute_thresholds()

        if self.bin_processing_method == 'kay':
            self.angles = np.zeros(self.delays_per_bunch_nb - 1)
            # self.weights = np.zeros(self.delays_per_bunch_nb - 1)
            self.compute_weights()

    def adjust_to(self, new_bin_abs_index, new_bin_rel_index, new_stage):
        self.bin_absolute_index = new_bin_abs_index
        self.bin_relative_index = new_bin_rel_index
        self.stage = new_stage
        self.bin_size = self.config.bins[self.stage]

    def process(self):
        if self.bin_processing_method == 'ml':
            self.ml_process()
        elif self.bin_processing_method == 'kay':
            self.compute_location()
            self.estimate_bin_signal()
        elif self.bin_processing_method == 'new':
            print('not implemented yet')

    def ml_process(self):
        ml_noise = float('inf')
        ml_location = 0
        
        # start with the first frequency location that aliases to the bin
        self.location = self.bin_relative_index

        while self.location < self.signal_length:
            self.estimate_bin_signal()
            
            if self.noise < ml_noise:
                ml_noise = self.noise
                ml_location = self.location
            
            # binSize is equal to the number of bins at the stage
            # this is the aliasing jumps a location i and i+self.bin_size
            # aliases to the same bin
            self.location += self.bin_size
        
        self.location = ml_location
        self.estimate_bin_signal()

    def compute_location(self):
        location_bis = 0

        for i in range(self.chains_nb):
            temp_location = self.signal_invk * self.get_omega(i) % self.signal_length
            temp_location += (temp_location > 0) * self.signal_length
            loc_update = temp_location / 2**i - location_bis
            r = loc_update - np.round((loc_update * 2**i) / self.signal_length) * (self.signal_length / 2**i)
            location_bis += r

        self.location = positive_mod(int(np.round(location_bis - self.bin_relative_index)/self.bin_size)*self.bin_size + self.bin_relative_index, self.signal_length)

    def get_omega(self, i):
        omega = 0
        need_to_shift = False
        for j in range(self.delays_per_bunch_nb - 1):
            y0 = self.observation_matrix[i * self.delays_per_bunch_nb + j, self.bin_absolute_index]
            y1 = self.observation_matrix[i * self.delays_per_bunch_nb + j + 1, self.bin_absolute_index]


            self.angles[j] = np.angle(y1*np.conj(y0))

            if not need_to_shift and self.angles[j] < -np.pi/2:
                need_to_shift = True
            omega += self.weights[j] * self.angles[j]
        
        if need_to_shift:
            omega += 2 * np.pi * np.sum(self.weights * (self.angles < 0))

        return omega + 2 * np.pi * (omega < 0)

    def estimate_bin_signal(self):
        """
        Given the location estimates the signal
        """
        delay_index = 0
        amplitude = 0
        
        # TODO: check if this works
        self.dirvector = np.exp(1j * self.signal_k * self.location * np.array(self.delays))
        amplitude = np.conj(self.dirvector).dot(self.observation_matrix[:, self.bin_absolute_index])

        self.amplitude = amplitude / self.delays_nb
        self.noise = 0

        self.signal_vector = self.amplitude * self.dirvector
        self.noise = norm_squared(self.observation_matrix[:, self.bin_absolute_index] - self.signal_vector)
        self.noise /= self.config.delays_nb

    def is_singleton(self):
        if self.is_zeroton():
            return False
        
        self.process()

        return self.noise <= self.thresholds[self.stage] and norm_squared(self.amplitude) > self.minimum_energy

    def is_zeroton(self):
        energy = norm_squared(self.observation_matrix[:,self.bin_absolute_index])
        return energy <= self.thresholds[self.stage]

    def compute_thresholds(self):
        # total number of bins
        self.energy_bins = np.zeros(self.config.bins_sum)

        energy_bin_counter = 0

        # this is to estimate the noise level
        # let's assume we know the noise level for now
        # if self.config.noisy or self.config.quantize or self.config.apply_window_var:
        #     # go over each stage
        #     for stage in range(self.config.get_num_stages()):
        #         # go over each bin in the stage
        #         for i in range(self.config.bins[stage]):
        #             print(self.config.bin_offsets[stage] + i)
        #             # pdb.set_trace()
        #             self.energy_bins[energy_bin_counter] = norm_squared(self.observation_matrix[:, self.config.bin_offsets[stage] + i])
        #             energy_bin_counter += 1

        #     self.energy_bins /= self.config.delays_nb

        #     self.energy_bins.sort()
        #     print(self.energy_bins)

        #     inv_eta = self.config.signal_sparsity_peeling * len(self.config.bins) / self.config.bins_sum
        #     max_value_wanted = np.round(self.config.bins_sum / np.e)
        #     max_value_wanted += self.config.bins_sum * sum([np.exp(-inv_eta) * pow(inv_eta, i) for i in (1, 2)]) 
        #     # in general, divide by gamma(i), but for i = 1 or 2 this is 1

        #     noise_level_crossed = False
        #     noise_estimation = 0
        #     energy_histogram_bins_counted = 0
        #     while not noise_level_crossed:
        #         noise_estimation += self.energy_bins[energy_histogram_bins_counted]
        #         energy_histogram_bins_counted += 1
        #         assert self.energy_bins[energy_histogram_bins_counted - 1] != 0, "div by 0"
        #         if (self.energy_bins[energy_histogram_bins_counted] / self.energy_bins[energy_histogram_bins_counted - 1]) >= 5:
        #             noise_level_crossed = True

        #     noise_estimation /= energy_histogram_bins_counted

        if self.config.noisy:
            noise_estimation = self.config.get_noise_sd()**2

        if not (self.config.noisy or self.config.quantize or self.config.apply_window_var):
            self.minimum_energy = 1e-8
        elif (self.config.noisy or self.config.quantize and not self.config.apply_window_var):
            self.minimum_energy = min(0.1 * noise_estimation * pow(10, self.config.SNR_dB / 10), 1000 * noise_estimation)
        
        if self.config.apply_window_var:
            self.minimum_energy = 0.1 * self.config.min_fourier_magnitude

        base_threshold = 1e-13
        if self.delays_nb < 10:
            factor = 4
        elif self.delays_nb < 20:
            factor = 3
        elif self.delays_nb < 50:
            factor = 2
        else:
            factor = 1.5

        if self.config.apply_window_var or self.config.noisy or self.config.quantize:
            self.thresholds *= 1e-10 + factor * noise_estimation
        else:
            self.thresholds *= base_threshold
        
    def compute_weights(self):
        """
        These weights are given in the original paper by Kay
        """
        base_weight = 6 / (self.delays_per_bunch_nb * (self.delays_per_bunch_nb ** 2 - 1))
        self.weights = base_weight * np.fromfunction(lambda i: (i + 1) * (self.delays_per_bunch_nb  - (i + 1)), (self.delays_per_bunch_nb-1,))
