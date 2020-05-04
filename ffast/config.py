from .utils import *
import numpy as np
import copy

class Config:
    def __init__(self, options=None):

        # first set options to default
        self.setDefaultOptions()
        # get the options from command line
        if options is not None:
            self.setOptionsFromCommandLine(options)
        # if bins are not given, propose bins
        # we will mostly rely on bins to be given
        if self.bins is None:
            self.proposed_bins()

        self.bins = np.array(self.bins)

        # if anything is not given, compute them
        self.compute_params()

    def is_kay_based_binprocessing(self):
        return self.bin_processing_method in ['kay', 'kay2']

    def compute_params(self):
        if self.apply_window_var:
            self.signal_sparsity_peeling = 3 * self.signal_sparsity
        self.max_SNR_dB = max(self.max_SNR_dB, self.SNR_dB)

        self.set_bin_offsets_and_sum()

        if self.distribution is None:
            self.preprocess_distribution("0 1")

        self.eff_snr = 10 ** (self.SNR_dB/10)
        if self.default_delays and self.apply_window_var:
            main_lobe_power = 2 * sum([pow(np.sin(i * np.pi / 2)/np.sin(i * np.pi/(2 * self.signal_length)), 2) for i in [1, 3, 5]])
            snr_from_offgrid = main_lobe_power / (self.signal_length ** 2 - main_lobe_power)
            self.offgrid_snr_dB = 10 * np.log10(snr_from_offgrid)
            self.eff_snr = 1/(1 / self.eff_snr + 1/snr_from_offgrid)

        if self.default_delays:
            self.delay_scaling = 3 * self.eff_snr/ (1 + 4 * np.sqrt(self.eff_snr))
            self.chains_nb = np.ceil(np.log(self.signal_length)/np.sqrt(self.delay_scaling))
            if self.is_kay_based_binprocessing():
                self.delays_per_bunch_nb = 2 * int(np.ceil(pow(np.log(self.signal_length), 1/3)) / np.sqrt(self.delay_scaling))
            self.chains_nb = max(self.chains_nb, 1)
            self.delays_per_bunch_nb = max(self.delays_per_bunch_nb, 2)
        

        if self.apply_window_var:
            self.chains_nb *= (1 + np.log(self.signal_sparsity)/np.log(10))
        self.delays_nb = int(self.chains_nb * self.delays_per_bunch_nb)
        if (not self.noisy and not self.apply_window_var):
            self.chains_nb = 1
            self.delays_per_bunch_nb = self.delays_nb
        assert self.signal_length >= self.signal_sparsity
        
        # TODO: uncomment these lines
        # assert self.delays_nb >= 2
        # assert self.delays_nb <= self.signal_length / max(self.bins)
        
        if self.is_kay_based_binprocessing():
            assert self.delays_per_bunch_nb >= 2

    def setDefaultOptions(self):
        self.output_file = "ffastOutput.txt"
        self.signal_length = 124950
        self.signal_length_original = self.signal_length
        self.signal_sparsity_peeling = 40
        self.signal_sparsity = 40
        self.length_factor = 1 # n = LCM(Bins)*lengthfactor

        # there will be multiple methods Kay, ML, new-one
        self.maximum_likelihood = False
        
        self.count_samples = True
        self.delays_per_bunch_nb = 2
        self.chains_nb = 1
        self.off_grid_SNR_dB = 100
        self.min_fourier_magnitude = 1
        self.help_displayed = False

        # information about the bins
        self.primes = None
        self.prime_powers = None
        self.bins = None

        self.distribution = None

        # for experiment mode
        self.iterations = 10
        self.experiment_mode = True
        # Phase = 0 implies phase of non-zero 
        # coefficients is uniformly random in [0,2*pi]. 
        self.phases_nb = 0
        # FFTWstrategy = FFTW_ESTIMATE # replace this with np.fft.fft
        self.display_iteration_time = False
        self.noisy = False
        self.quantize = False
        self.SNR_dB = 50
        self.quantization_bits_nb = 0
        self.max_SNR_dB = -float('inf')
        self.verbose = False
        self.reconstruct_signal_in_backend = False
        self.off_grid = False
        self.default_delays = True
        self.apply_window_var = False

    def setOptionsFromCommandLine(self, options):
        '''
        Takes in the flags passed into 'main' and sets options accordingly.
        '''
        
        self.experiment_mode = self.experiment_mode or options.experiment
        

        # now we get primes, prime powers, and bins separately
        # we do not need to do that
        if options.primes is not None:
            self.primes = np.array(options.primes)
        if options.prime_powers is not None:
            self.prime_powers = np.array(options.prime_powers)
        if options.bins is not None:
            self.bins = np.array(options.bins) # c++ converts to ints but shouldn't need to do that here
        
        if options.length is not None:
            self.signal_length = options.length
            self.signal_length_original = options.length
        self.count_samples = not options.samples
        if options.delays is not None:
            self.delays_per_bunch_nb = options.delays
            self.default_delays = False
        if options.chains is not None:
            self.chains_nb = options.chains
            self.default_delays = False
        if options.file is not None:
            self.input_file = options.file
            self.experiment_mode = False
        if options.minmagnitude is not None:
            self.min_fourier_magnitude = options.minmagnitude
        if options.iterations is not None:
            self.iterations = options.iterations
        if options.sparsity is not None:
            self.signal_sparsity = options.sparsity
            self.signal_sparsity_peeling = options.sparsity

        # method for bin processing ['kay', 'kay2', ml', 'new']
        self.bin_processing_method = options.bin_processing_method
        
        if options.factor is not None:
            self.length_factor = options.factor
        if options.quantization is not None:
            self.quantize = True
            self.quantization_bits_nb = options.quantization
        self.reconstruct_signal_in_backend = options.reconstruct
        
        self.noisy = options.snr is not None
        self.SNR_dB = options.snr or self.SNR_dB
        
        if options.maxsnr is not None:
            self.max_SNR_dB = options.maxsnr
        if options.distribution is not None:
            self.preprocess_distribution(self.distribution)
        self.output_file = options.outfile
        self.verbose = options.verbose

    def preprocess_distribution(self, new_distribution):
        temp_distribution = new_distribution.split(" ")
        l = len(temp_distribution)
        self.distribution = np.zeros(l)
        for i in range(l):
            self.distribution[i] = self.distribution[i - 1] + float(temp_distribution[i])
        self.distribution /= self.distribution[l - 1]

    def proposed_bins(self, new_range_semilength=None):
        F = int(self.signal_length ** (1/3))
        if new_range_semilength is None or new_range_semilength > F:
            range_semilength = F - 2
        else:
            range_semilength = new_range_semilength
        best_length = (F - range_semilength) ** 3
        tested_bins = [] # this isn't the fastest but eh

        for f1 in range(F - range_semilength, F + range_semilength + 1):
            tested_bins.append(f1)
            p = f1 * (F - range_semilength)
            for f2 in range(F - range_semilength, F + range_semilength + 1):
                tested_bins.append(f2)
                r = int(np.floor(self.signal_length / p))
                for n in range(p * r, best_length, -p):
                    f3 = n // p
                    if f3 <= F + range_semilength and f3 >= F - range_semilength:
                        tested_bins.append(f3)
                        if coprime(tested_bins):
                            best_length = n
                            best_bins = copy.deepcopy(tested_bins)
                        
                        tested_bins.pop()
                
                tested_bins.pop()
                p += f1
            tested_bins.pop()
        
        self.bins = best_bins
        self.apply_window_var = (self.signal_length != best_length) or self.apply_window_var 
        self.signal_length = best_length

    def set_bin_offsets_and_sum(self):
        # self.bins.sort() # lets not sort this
        self.bins_sum = sum(self.bins)
        self.bin_offsets = np.roll(np.cumsum(self.bins),1)
        self.bin_offsets[0] = 0

    def get_num_stages(self):
        return len(self.bins)

    def need_to_use_ml_detection(self):
        return self.maximum_likelihood

    def get_noise_sd(self):
        return pow(10, -self.SNR_dB / 20)

    def __str__(self):
        s_list = []

        s_list.append("Running experiment mode (for now)")
        s_list.append("Signal length: %d" % self.signal_length)
        s_list.append("Signal sparsity: %d" % self.signal_sparsity)

        if self.noisy:
            s_list.append("Signal-to-noise ratio (dB): %d" % self.SNR_dB)
        else:
            s_list.append("Noiseless signal")

        if self.phases_nb == 0:
            s_list.append("Random phase")

        s_list.append("Delays: %d" % self.delays_nb)
        s_list.append("Bins: " + ' '.join([str(b) + '' for b in self.bins]))

        s_list.append("Bin processor: {}".format(self.bin_processing_method))

        return '\n'.join(s_list)

    def display(self):
        print(self.__str__())