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

        if self.is_kay_based_binprocessing():
            self.angles = np.zeros(self.delays_per_bunch_nb - 1)
            # self.weights = np.zeros(self.delays_per_bunch_nb - 1)
            self.compute_weights()

    def is_kay_based_binprocessing(self):
        if self.bin_processing_method == 'kay' or self.bin_processing_method == 'kay2':
            return True
        else:
            return False

    def get_delays_for_stage(self, stage):
        # if delays is a list of lists
        if isinstance(self.delays[0], list):
            return self.delays[stage]
        else:
            return self.delays

    def adjust_to(self, new_bin_abs_index, new_bin_rel_index, new_stage):
        self.bin_absolute_index = new_bin_abs_index
        self.bin_relative_index = new_bin_rel_index
        self.stage = new_stage
        self.bin_size = self.config.bins[self.stage]

    def process(self):
        if self.bin_processing_method == 'ml':
            self.ml_process()
        elif self.is_kay_based_binprocessing():
            self.compute_location()
            self.estimate_bin_signal()
        elif self.bin_processing_method == 'new':
            self.find_location_new()
            self.estimate_bin_signal()
            

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
            if self.bin_processing_method == 'kay':
                temp_location = self.signal_invk * self.get_omega(i) % self.signal_length
            elif self.bin_processing_method == 'kay2':
                temp_location = self.signal_invk * self.get_omega2(i) % self.signal_length
            else:
                raise NotImplementedError
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

    def get_omega2(self, i):
        """
        This is my best implementation of doing phase unwarping in kay's method
        """
        a = i * self.delays_per_bunch_nb
        b = a + self.delays_per_bunch_nb

        my_signal = self.observation_matrix[a:b, self.bin_absolute_index]
        
        # we will check 4 settings
        t = np.arange(self.delays_per_bunch_nb)
        
        v_best = -np.inf
        omega_best = 0
        for k in range(4):
            rotater = np.exp(-1j*np.pi*k*t/2)
            y = my_signal*rotater
            angle_diff = np.angle(np.conj(y[0:-1])*y[1:])
            omega = self.weights.dot(angle_diff)
            
            y_hat = np.exp(1j*omega*t)
            
            v = np.abs(np.dot(y, np.conj(y_hat)))
            
            if v > v_best:
                omega_best = (omega + k*np.pi/2) % (2*np.pi)
                v_best = v
        
        return omega_best


    def decode_amplitude(self, v):
        """
        The theory requires things to be on a constellation
        So here we decode which constellation point we are sending
        """
        if v > 0:
            return 1
        else:
            return -1

    def estimate_bin_signal(self):
        """
        Given the location estimates the signal

        The theory requires things to be on constellation, so we are going to assume the
        signal either is +1 or -1 here
        """
        # TODO: check if this works

        delays = self.get_delays_for_stage(self.stage)
        self.dirvector = np.exp(1j * self.signal_k * self.location * np.array(delays))
        amplitude = np.conj(self.dirvector).dot(self.observation_matrix[:len(delays), self.bin_absolute_index])
        amplitude = amplitude / len(delays)

        # here we edit the amplitude maybe using decode_amplitude
        self.amplitude = amplitude
        
        self.noise = 0
        self.signal_vector = self.amplitude * self.dirvector
        self.noise = norm_squared(self.observation_matrix[:len(delays), self.bin_absolute_index] - self.signal_vector)
        self.noise /= len(delays)

    def is_singleton(self):
        """
        Singleton detection 

        TODO: make this part of the code more robust
        - in the case of low SNR, the zeroton test (which is energy based) is not robust anymore,
        - look into zeroton test in low SNR,
        - one option to get around this is to look at the amplitude of the signal at the bin and declare it is
          a singleton if it is large.  However, when we have small number of delays this might not be robust 
          anymore.
        """
        
        # if self.is_zeroton():
            # return False
        
        self.process()

        # this technique is by looking at the residual energies
        # if self.noise <= self.thresholds[self.stage] and norm_squared(self.amplitude) > self.minimum_energy:
            # is_singleton = True

        # Alternate "signal detection" methos
        # this is not working robustly so far
        if np.abs(self.amplitude) > 0.9:
            is_singleton = True
        else:
            is_singleton = False

        return is_singleton

    def is_zeroton(self):
        energy = norm_squared(self.observation_matrix[:,self.bin_absolute_index])

        print('energy: {}'.format(energy))
        print('threshold: {}'.format(self.thresholds[self.stage]))
        return energy <= self.thresholds[self.stage]

    def compute_thresholds(self):
        # total number of bins
        self.energy_bins = np.zeros(self.config.bins_sum)

        # this is to estimate the noise level
        # let's assume we know the noise level for now
        # energy_bin_counter = 0
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

    def get_inverter_for_stage(self):
        r_total = 1
        p = self.config.primes[self.stage]
        n = self.config.prime_powers[self.stage]
        F = p**n
        
        for si in range(self.config.get_num_stages()):
            if si != self.stage:
                a = self.config.primes[si]
                m = self.config.prime_powers[si]
                r = get_multiplicative_inverse(a, m, p, n)
                r_total = (r_total*r)%F
        
        return r_total

    def get_residual_location_for_stage(self):
        r = self.get_inverter_for_stage()
        return r * self.bin_relative_index

    def indices_for_stage_factor_bit(self, factor_index, bit_index):
        assert self.stage != factor_index, 'factor and stage should be different'
        assert bit_index < self.config.prime_powers[factor_index], 'factor does not have that bit'
        
        p = self.config.primes[factor_index]
        b = 0
        for fi in range(0, factor_index):
            if fi != self.stage:
                b += self.config.prime_powers[fi] * self.config.primes[fi] * self.config.delays_per_bunch_nb
        b += p * bit_index * self.config.delays_per_bunch_nb
        e = b + p * self.config.delays_per_bunch_nb
        return b, e

    def estimate_bit_ml(self,
                        chain,
                        sampling_points,
                        ref_w0,
                        prime_base,
                        statistics_function=np.abs):
        """
        chain:               pairs of samples obtained from the chain
        sampling_points:     the points at which the samples are obtained
        ref_w0:              the reference frequency obtained so far
        p:                   prime that is equal to the base
        statistics_function: when we know the amplitude we can use np.real 
                             as the statistics function.  However, when the
                             coefficient is not known, we need to use np.abs
        """
        # number of pairs in a chain
        max_inprod = -np.inf
        best_loc = 0
        rotater = np.exp(-1j*ref_w0*sampling_points)
        chain = chain*rotater

        # go over possible bits
        t = np.arange(len(sampling_points))
        for l in range(prime_base):
            steering_vector = np.exp(-1j*2*np.pi*l*t/prime_base)
            current_inprod = chain.dot(steering_vector)

            if statistics_function(current_inprod) > max_inprod:
                max_inprod = statistics_function(current_inprod)
                best_loc = l
        return int(best_loc)

    def recover_modulo(self, factor_index):
        """
        Make the one with the signal pre-made.
        """
        r = self.get_residual_location_for_stage()
        F = self.config.bins[self.stage] 
        p = self.config.primes[factor_index]
        q = self.config.prime_powers[factor_index]
        N = self.config.signal_length / self.config.bins[self.stage]
        
        c_so_far = 0
        # go over each chain
        for chain_index in range(q):
            delay_b, delay_e = self.indices_for_stage_factor_bit(factor_index, chain_index)
            # relevant chunk of the observation matrix
            x = self.observation_matrix[delay_b:delay_e, self.bin_absolute_index]
            # the delays (sampling locations for the tone)
            t = np.array(self.get_delays_for_stage(self.stage)[delay_b:delay_e])
            # rotate the signal with respect to the residual location in the stage
            y = x * np.exp(-1j*2*np.pi*t*r/F)
            ref_w0 = (2*np.pi)*c_so_far/N
            bit = self.estimate_bit_ml(y, t, ref_w0, prime_base=p)
            c_so_far += bit * (p**chain_index)

        return c_so_far

    # TODO: fix this part
    def find_location_new(self):
        N = self.config.signal_length // self.config.bins[self.stage]
        
        u = 0
        # go over stages (si: stage index)
        for si in range(self.config.get_num_stages()):
            if si != self.stage:
                p = self.config.primes[si]
                q = self.config.prime_powers[si]
                
                N_bar = int(N // self.config.bins[si])
                
                minv = get_multiplicative_inverse(N_bar, 1, p, q)
                
                ri = self.recover_modulo(si)
                
                u += minv*ri*N_bar
                
        u = u % N
        
        loc = u * self.config.bins[self.stage] + self.get_residual_location_for_stage() * N

        self.location = int(loc % self.config.signal_length)
