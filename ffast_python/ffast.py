from .config import *
from .input_signal import *
from .output_signal import *

import numpy as np
import argparse

class FFAST:
    def __init__(self, config, input_signal, output_signal):
        print("Defining frontend")
        self.frontend = FrontEnd(config, input_signal)
        print("Defining backend")
        self.backend = BackEnd(config, self.frontend)
        print("Setting output signal on backend")
        output_signal.set_backend(backend)
        self.iteration = 0

    def get_delays(self):
        return self.frontend.delays

    def process(self):
        self.iteration += 1
        self.frontend.process()
        self.backend.process()

parser = argparse.ArgumentParser()
flags = [("-a", "--experiment", "store_true"),
         ("-b", "--bins", str),
         ("-c", "--samples", "store_true"),
         ("-d", "--delays", int),
         ("-e", "--chains", int),
         ("-f", "--file", str),
         ("-g", "--minmagnitude", float),
         ("-w", "--fftw", "store_true"), 
         ("-i", "--iterations", int),
         ("-k", "--sparsity", int),
         ("-l", "--ml", "store_true"),
         ("-m", "--factor", int),
         ("-n", "--length", int), 
         ("-o", "--optimize", "store_true"),
         ("-q", "--quantization", int),
         ("-r", "--reconstruct", "store_true"),
         ("-s", "--snr", float),
         ("-u", "--distribution", str),
         ("-v", "--verbose", "store_true"),
         ("-x", "--maxsnr", float),
         ("-z", "--outfile", str)
        ]

for flag_tuple in flags:
    short_flag, long_flag, type_input = flag_tuple
    if type_input == "store_true":
        parser.add_argument(short_flag, long_flag, action="store_true")
    else:
        parser.add_argument(short_flag, long_flag, type=type_input)

# '-w' is overloaded
# does '-k' need two ints passed in? does '-n'?
# '-u' needs additional action

def main():
    np.random.seed(0)
    args = parser.parse_args()
    config = Config(args)

    # if config.experiment_mode:
    input_signal = ExperimentInputSignal(config)
    output_signal = ExperimentOutputSignal(config, input_signal)
    '''else:
        input_signal = CustomizedInput(config)
        output_signal = CustomizedOutput(config, input_signal)'''
    
    print("Defining FFAST")
    ffast = FFAST(config, input_signal, output_signal)
    print("Running process")
    if not config.help_displayed:
        iterations = config.iterations
        config.display()
        print("Running iterations")
        for i in range(iterations):
            print("Running input")
            input_signal.process(FFAST.get_delays())
            print("Running FFAST process")
            ffast.process()
            print("Running output")
            output_signal.process()

        ffast.display_results()
        if config.experiment_mode:
            print("time taken")

if __name__ == "__main__":
    main()