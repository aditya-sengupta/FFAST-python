class Config:
    def __init__(self, options):
        setDefaultOptions()
        setOptionsFromCommandLine(options)

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

    def setOptionsFromCommandLine(self, options):
        '''
        Takes in the flags passed into 'main' and sets options accordingly.
        '''
        pass

