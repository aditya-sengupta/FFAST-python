from .config import *
import numpy as np
from collections import Counter

# ffast_real is a double

class InputSignal:
    pass

class ExperimentInputSignal(InputSignal):
    def __init__(self, config):
        self.config = config
        self.signal_magnitude = 1
        self.noise_sd = pow(10, -config.SNR_dB / 20) # noise_power is square of this (for use outside add_noise)
        self.nonzero_freqs = {}

    def process(self):
        '''
        Goes through the process to create an input signal.
        Note that this creates the whole input signal, not just the needed samples.
        '''
        print("Starting generate_nonzero_freqs")
        self.generate_nonzero_freqs()
        print("Starting f_to_t")
        self.frequency_to_time() # to check
        if self.config.noisy:
            self.add_noise()

        print("Scaling the FT")
        # scaling the Fourier transform
        self.time_signal /= np.sqrt(self.config.signal_length_original)
        if self.config.quantize:
            self.apply_quantization(self.config.quantization_bits_nb)

    def generate_nonzero_freqs(self):
        '''
        Generates numpy arrays containing frequencies, magnitudes, phases of the signal.
        '''
        temp_locations = Counter()

        print(self.config.signal_sparsity)
        while sum([temp_locations[i] for i in temp_locations.keys()]) < self.config.signal_sparsity:
            dist_call = self.distribution(np.random.uniform(), self.config.distribution)
            temp_location = int(np.floor(self.config.signal_length_original * dist_call) % self.config.signal_length_original)
            # for off-grid we need guard bands (?)
            if (self.config.signal_length_original != self.config.signal_length):
                if sum([temp_locations[temp_location - i] for i in range(-5, 6)]) == 0:
                    temp_locations[temp_location] += 1
            else:
                temp_locations[temp_location] += 1

        print(temp_locations)
        self.freqs = np.array(temp_locations) # I think
        self.magnitudes = np.ones(self.freqs.size) * self.signal_magnitude # confirm they should all be the same magnitude
        self.phases = [self.get_random_phase() for _ in range(self.config.signal_sparsity)]

    def frequency_to_time(self):
        w0 = 2 * np.pi / self.config.signal_length_original
        self.time_signal = np.zeros(self.config.signal_length_original)
        for f in self.nonzero_freqs:
            self.time_signal += self.nonzero_freqs[f]

    def add_noise(self):
        signal_power = self.signal_magnitude ** 2
        noise_power = 0
        for i in range(self.config.signal_length):
            noise_norm_factor = -2 * np.log(np.random.uniform())
            noise_phase = np.random.uniform(0, 2 * np.pi)
            noise_power += noise_norm_factor # redundant?
            self.time_signal[i] += self.noise_sd * sqrt(noise_norm_factor / 2) * np.exp(1j * noise_phase)

        self.noise_power = noise_power * self.noise_sd ** 2
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
        phases_nb = self.config.phases_nb
        if phases_nb < 1:
            return np.random.uniform(0, 2 * np.pi)
        elif phases_nb == 1:
            return 0
        elif phases_nb == 2:
            return np.random.choice([0, np.pi])
        else:
            return (np.floor(np.random.uniform(0, phases_nb)) * 2 + 1) * np.pi / phases_nb

    def distribution(self, urand, F):
        # urand is of type ffast_real
        # F is a numpy array with doubles?
        for i in range(1, F.size): #0, or 1?
            if urand < F[i]:
                return (urand - F[i])/(F[i] - F[i-1]) + i/(F.size - 1)
        return 1
        
