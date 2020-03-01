from .config import *
from .input_signal import *
from .output_signal import *
import time

import numpy as np
import argparse

class FFAST:
    def __init__(self, config, input_signal, output_signal):
        self.config = config
        self.frontend = FrontEnd(config, input_signal)
        self.backend = BackEnd(config, self.frontend)
        output_signal.set_backend(self.backend)
        self.iteration = 0

    def get_delays(self):
        return self.frontend.delays

    def process(self):
        self.iteration += 1
        self.frontend.process()
        self.backend.process()

    def display_results(self, time):
        # print this by default
        self.config.display()
        print("<===== NUMBER OF SAMPLES =====>")
        samples_used = self.frontend.get_used_samples_nb()
        print("%d -> used samples" % samples_used)
        proportion = 100 * samples_used / self.config.signal_length_original
        print("%.2f%% samples touched" % proportion)
        print("<===== AVERAGE TIME (in milliseconds) =====>")
        print("Total time: %d" % time)
        time_per = time / self.iteration
        print("Time per iteration: % d" % time_per)

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
    time_initial = int(np.round(time.time() * 1000))
    # np.random.seed(0)
    args = parser.parse_args()
    config = Config(args)

    # if config.experiment_mode:
    input_signal = ExperimentInputSignal(config)
    output_signal = ExperimentOutputSignal(config, input_signal)
    '''else:
        input_signal = CustomizedInput(config)
        output_signal = CustomizedOutput(config, input_signal)'''
    
    ffast = FFAST(config, input_signal, output_signal)
    if not config.help_displayed:
        iterations = config.iterations
        for i in range(iterations):
            input_signal.process() # ffast.get_delays() in the CPP code as an argument
            ffast.process()
            output_signal.process()

        time_final = int(np.round(time.time() * 1000))

        ffast.display_results(time_final - time_initial)

if __name__ == "__main__":
    main()