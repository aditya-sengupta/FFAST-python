from utils import *
import numpy as np

class Config:
    def __init__(self, options):
        setDefaultOptions()
        setOptionsFromCommandLine(options)
        if len(self.bins) == 0:
            self.proposed_bins()
        if self.apply_window_var:
            self.signal_sparsity_peeling = 3 * self.signal_sparsity
        self.max_SNR_dB = max(self.max_SNR_dB, self.SNR_dB)
        self.set_bin_offsets_and_sum()
        if len(self.distribution) == 0:
            self.preprocess_distribution("1")
        if self.default_delays and (self.noisy or self.apply_window_var):
            self.eff_snr = 10 ** (self.SNR_dB/10)
            if self.apply_window_var:
                main_lobe_power = 2 * sum([pow(np.sin(i * np.pi / 2)/np.sin(i * np.pi/(2 * self.signal_length)), 2) for i in [1, 3, 5]])
                snr_from_offgrid = main_lobe_power / (self.signal_length ** 2 - main_lobe_power)
                self.offgrid_snr_dB = 10 * np.log10(snr_from_offgrid)
                self.eff_snr = 1/(1 / self.eff_snr + 1/snr_from_offgrid)

        self.delay_scaling = 3 * self.eff_snr/ (1 + 4 * np.sqrt(self.eff_snr))
        self.chains_nb = np.ceil(np.log(self.signal_length)/np.sqrt(self.delay_scaling))
        if not self.maximum_likelihood:
            self.delays_per_bunch_nb = 2 * np.ceil(pow(np.log(self.signal_length), 1/3)) / np.sqrt(delay_scaling)
        self.chains_nb = max(self.chains_nb, 1)
        self.delays_per_bunch_nb = max(self.delays_per_bunch_nb, 2)
        if self.apply_window_var:
            self.chains_nb *= (1 + np.log(self.signal_sparsity)/np.log(10))
        self.delays_nb = self.chains_nb * self.delays_per_bunch_nb
        if (not self.noisy and not self.apply_window_var):
            self.chains_nb = 1
            self.delays_per_bunch_nb = self.delays_nb
        assert self.signal_length >= self.signal_sparsity
        assert self.delays_nb >= 2
        assert self.delays_nb <= self.signal_length / max(self.bins)
        if not self.maximum_likelihood:
            assert self.delays_per_bunch_nb >= 2

    def setDefaultOptions(self):
        self.output_file = "ffastOutput.txt"
        self.signal_length = 124950
        self.signal_length_original = self.signal_length
        self.signal_sparsity_peeling = 40
        self.signal_sparsity = 40
        self.length_factor = 1 # n = LCM(Bins)*lengthfactor
        self.maximum_likelihood = False
        self.count_samples = True
        self.delays_per_bunch_nb = 2
        self.chains_nb = 1
        self.off_grid_SNR_dB = 100
        self.min_fourier_magnitude = 1

        # for experiment mode
        self.iterations = 1
        self.experiment_mode = True
        # Phase = 0 implies phase of non-zero 
        # coefficients is uniformly random in [0,2*pi]. 
        self.phases_nb = 0
        # FFTWstrategy = FFTW_ESTIMATE # replace this with np.fft.fft
        self.compare_with_FFTW = False
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
        if options.bins is not None:
            self.bins = np.array(options.bins.split(" ")) # c++ converts to ints but shouldn't need to do that here
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
        self.maximum_likelihood = options.ml
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
        # fftw_strategy to be replaced by np.fft.fft
        self.verbose = options.verbose
        self.compare_with_fftw = self.compare_with_fftw or options.fftw

    def preprocess_distribution(self, new_distribution):
        temp_distribution = new_distribution.split(" ")
        l = len(temp_distribution)
        self.distribution = np.zeros(l)
        for i in range(1, l):
            self.distribution[i] = self.distribution[i - 1] + float(temp_distribution[i])
        self.distribution /= self.distribution[l - 1]

    def proposed_bins(self, new_range_semilength=None):
        F = int(signal_length ** (1/3))
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
                r = np.floor(self.signal_length / p)
                for n in range(p * r, best_length, -p):
                    f3 = n / p
                    if f3 <= F + range_semilength and f3 >= F - range_semilength:
                        tested_bins.append(f3)
                        if coprime(tested_bins):
                            best_length = n
                            best_bins = tested_bins
                        
                        tested_bins.pop()
                
                tested_bins.pop()
                p += f1
            tested_bins.pop()
        
        self.bins = best_bins
        self.apply_window_var = (signal_length != best_length) or self.apply_window_var 
        self.signal_length = best_length

    def set_bin_offsets_and_sum(self):
        self.bins.sort()
        self.bins_sum = sum(self.bins)
        self.bin_offsets = np.cumsum(self.bins)

    def need_to_use_ml_detection(self):
        raise NotImplementedError()
        