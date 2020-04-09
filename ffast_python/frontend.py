from .config import *
from .input_signal import *
import numpy as np
import pdb

class FrontEnd:
    def __init__(self, config, input_signal):
        self.signal_length = config.signal_length
        self.config = config
        self.input_signal = input_signal
        self.input_signal.process()
        self.sampling_period = self.signal_length / config.bins
        self.used_samples = set()
        self.compute_delays()
        self.observation_matrix = np.zeros((len(self.delays), sum(config.bins)), dtype=np.complex128)
        self.count_samples_done = False


    def stage_begin_end(self, stage):
        return self.config.bin_offsets[stage], self.config.bin_offsets[stage] + self.config.bins[stage]

    def get_delays_for_stage(self, stage):
        # if delays is a list of lists
        if isinstance(self.delays[0], list):
            return self.delays[stage]
        else:
            return self.delays

    def process(self):
        # re-compute delays (this can be done once or each time)
        self.compute_delays()
        # re initialize the uses samples if we are recomputing delays
        self.used_samples = set()

        signal = self.input_signal.time_signal

        # make the observation matrix
        # TODO: make this a list of bins
        self.observation_matrix = np.zeros((self.get_max_delays(), sum(self.config.bins)), dtype=np.complex128)

        # go over each stage
        for stage in range(self.config.get_num_stages()):
            # sampling period at the stage
            stage_sampling = int(self.sampling_period[stage])
            s, e = self.stage_begin_end(stage)

            for i, d in enumerate(self.get_delays_for_stage(stage)):
                # print('frontend delay: {}'.format(d))
                # delays should wrap around the signal length
                subsampling_points = np.arange(d, d+self.config.signal_length, stage_sampling) % self.config.signal_length
                # subsample the signal
                self.used_samples = self.used_samples.union(set(subsampling_points))
                subsampled_signal = np.sqrt(stage_sampling) * signal[subsampling_points] * self.window(stage_sampling)
                transformed = np.fft.fft(subsampled_signal)    

                self.observation_matrix[i][s:e] = transformed / np.sqrt(self.config.bins[stage])
        self.count_samples_done = True

    def get_max_delays(self):
        # if it is a list of lists
        if isinstance(self.delays[0], list):
            max_val = 0
            for d in self.delays:
                if len(d) > max_val:
                    max_val = len(d)
        else:
            max_val = len(self.delays)
        return max_val


    def compute_delays(self):
        if (self.config.noisy or self.config.apply_window_var):
            
            # ml approach takes random samples------------------------------------------------------
            if self.config.bin_processing_method == 'ml':                
               self.delays = np.random.choice(self.signal_length, self.config.delays_nb, replace=False)
               self.delays = list(map(int, self.delays))

            # kay's method takes samples with increasing gaps---------------------------------------
            elif self.config.is_kay_based_binprocessing():
                # uses FFAST_Search
                delay_roots = np.random.choice(self.signal_length, self.config.chains_nb, replace=False)
                self.delays = np.zeros(self.config.delays_nb)
                delay_index = 0
                for chain_index in range(self.config.chains_nb):
                    jump = 2**chain_index
                    for delay_within_chain_index in range(self.config.delays_per_bunch_nb):
                        self.delays[delay_index] = delay_roots[chain_index] + jump*delay_within_chain_index
                        delay_index += 1
                self.delays %= self.signal_length
                self.delays = list(map(int, self.delays))

            # this takes samples with respect to different bits of the prime factor-----------------
            elif self.config.bin_processing_method == 'new':
                self.delays = self.delays_for_stages()
        else:
            self.delays = np.array(list(range(self.config.delays_nb)))
            self.delays = list(map(int, self.delays))
        
    def delays_for_factor(self, stage_index, factor_index):
        p = self.config.primes[factor_index]
        q = self.config.prime_powers[factor_index]
        
        # these are the other factors we need to annihilate
        annihilating_jump = self.config.signal_length/(self.config.bins[factor_index]*self.config.bins[stage_index])
        
        delays_for_factor = []
        for qi in range(q-1,-1,-1):
            # random offsets
            # root = np.random.choice(self.config.signal_length, 1)
            # default 0 offset
            root = 0
            
            t = np.arange(0, p*self.config.delays_per_bunch_nb) * (annihilating_jump * p**qi)
            t = (root + t) % self.config.signal_length
            t = t.astype(int)
            delays_for_factor += t.tolist()
        return delays_for_factor
    
    def delays_for_stage(self, stage_index):
        delays = []
        for fi in range(self.config.get_num_stages()):
            if fi != stage_index:
                delays += self.delays_for_factor(stage_index, fi)
        return delays
    
    def delays_for_stages(self):
        delays = []
        for si in range(self.config.get_num_stages()):
            delays.append(self.delays_for_stage(si))
        return delays
    
    def indices_for_stage_factor_bit(self, stage_index, factor_index, bit_index):
        assert stage_index != factor_index, 'factor and stage should be different'
        assert bit_index < self.config.prime_powers[factor_index], 'factor does not have that bit'
        
        p = self.config.primes[factor_index]
        b = 0
        for fi in range(0, factor_index):
            if fi != stage_index:
                b += self.config.prime_powers[fi] * self.config.primes[fi] * self.config.delays_per_bunch_nb
        
        b += p * bit_index * self.config.delays_per_bunch_nb
        e = b + p * self.config.delays_per_bunch_nb
        return b,e

    def get_used_samples_nb(self):
        return len(self.used_samples)

    def window(self, i):
        if not self.config.apply_window_var:
            return 1
        else:
            # Blackmann-Nuttall window
            a = [0.3635819, -0.4891775, 0.1365995, -0.0106411]
            return 2 * (a[0] + sum([a[j] * np.cos(2*np.pi * j * i) / (self.config.signal_length - 1) for j in range(1,4)]))
