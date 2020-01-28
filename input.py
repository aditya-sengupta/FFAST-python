from config import *
import numpy as np
from collections import Counter

# ffast_real is a set(int)

class ExperimentInput:
    def __init__(self, config):
        self.config = config
        self.signal_magnitude = 1
        self.noise_sd = pow(10, -config.SNR_dB / 20) # noise_power is square of this (for use outside add_noise)
        self.nonzero_freqs = {}

    def process(self):
        self.needed_samples = set(range(self.config.signal_length)) 
        # I think this can be replaced everywhere by range(config.signal_length)
        self.generate_nonzero_freqs()
        self.frequency_to_time() # to check
        if self.config.noisy:
            self.add_noise()

        # scaling the Fourier transform
        self.time_signal /= np.sqrt(self.config.signal_length_original)
        if self.config.quantize:
            self.apply_quantization(self.config.quantization_bits_nb)

    def generate_nonzero_freqs(self):
        '''
        Generates a dictionary of frequencies: keys are frequencies, values are complex numbers encoding magnitude and phase.
        '''
        temp_locations = Counter()

        while len(temp_locations) < config.signal_sparsity:
            dist_call = distribution(np.random.uniform(), config.distribution) # config.distribution not set yet!
            temp_location = int(np.floor(config.signal_length_original * dist_call) % config.signal_length_original)

            # for off-grid we need guard bands (?)
            if (config.signal_length_original != config.signal_length):
                if sum([temp_locations[temp_location - i] for i in range(-5, 6)]) == 0:
                    temp_locations[temp_location] += 1
            else:
                temp_locations[temp_location] += 1

        for l in temp_locations:
            self.nonzero_freqs[l] = (self.signal_magnitude * np.exp(1j * self.get_random_phase()))

    def frequency_to_time(self):
        w0 = 2 * np.pi / self.config.signal_length_original
        self.time_signal = np.zeros(self.config.signal_length_original)
        for f in self.nonzero_freqs:
            self.time_signal += 0 
            # ref experimentinput.cpp line 297: is this just a complicated way of adding sines?

    def add_noise(self):
        signal_power = self.signal_magnitude ** 2
        noise_power = 0
        for i in range(self.config.signal_length):
            noise_norm_factor = -2 * np.log(np.random.uniform())
            noise_phase = np.random.uniform(0, 2 * np.pi)
            noise_power += noise_norm_factor # redundant?
            self.time_signal[i] += self.noise_sd * sqrt(noise_norm_factor / 2) * np.exp(1j * noise_phase)

        self.real_snr = signal_power * self.signal_length / (self.noise_sd ** 2)

    def apply_quantization(self, bits_nb):
        self.not_initialized = True
        levels_nb = 1 << bits_nb
        min_val = min(np.min(np.real(self.time_signal), np.imag(self.time_signal)))
        max_val = max(np.max(np.real(self.time_signal), np.imag(self.time_signal)))

        scaling = (levels_nb - 1) / (max_val - min_val)

        self.time_signal = np.round(scaling * (self.time_signal - min_val)) / scaling + min_val

    # utilities

    def get_random_phase(self):
        if config.phases_nb < 1:
            return np.random.uniform(0, 2 * np.pi)
        elif config.phases_nb == 1:
            return 0
        elif config.phases_nb == 2:
            return np.random.choice([0, np.pi])
        else:
            return (np.floor(np.random.uniform(0, config.phases_nb)) * 2 + 1) * np.pi / config.phases_nb

    def distribution(urand, F):
    '''
    urand is of type ffast_real
    F is a numpy array with doubles?
    '''
    for i in range(1, F.size): #0, or 1?
        if urand < F[i]:
            return (urand - F[i])/(F[i] - F[i-1]) + i/(F.size - 1)
    return 1
        
