from config import *
from backend import *
import numpy as np

class OutputSignal:
    pass

class ExperimentOutputSignal(OutputSignal):
    def __init__(self, config):
        self.config = config
        self.binning_failures_nb = 0
        self.full_recoveries_nb = 0

    def process(self):
        if config.signal_length_original != config.signal_length:
            pass
        