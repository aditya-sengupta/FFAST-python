from config import *
from input_signal import *

class FrontEnd:
    def __init__(self, config, input_signal):
        signal_length = config.signal_length
        self.sampling_period = [signal_length / config.bin_size[i] for i in range(len(config.))]