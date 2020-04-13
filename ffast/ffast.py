from .config import *
from .input_signal import *
from .output_signal import *
import time
import argparse

def make_args():
    parser = argparse.ArgumentParser()
    # set up args to pass a list of integers
    flags = [("-a", "--experiment", "store_true"),
            ("-b", "--bins", "list"), #
            ("-c", "--samples", "store_true"),
            ("-d", "--delays", int), #
            ("-e", "--chains", int),
            ("-f", "--file", str),
            ("-g", "--minmagnitude", float), 
            ("-i", "--iterations", int), #
            ("-k", "--sparsity", int), #
            # ("-l", "--ml", "store_true"), put this below as it requires some processing
            ("-m", "--factor", int),
            ("-n", "--length", int), #
            ("-o", "--optimize", "store_true"),
            ("-q", "--quantization", int),
            ("-r", "--reconstruct", "store_true"),
            ("-s", "--snr", float), #
            ("-u", "--distribution", "list"),
            ("-v", "--verbose", "store_true"),
            ("-x", "--maxsnr", float),
            ("-z", "--outfile", str) 
            ]

    for flag_tuple in flags:
        short_flag, long_flag, type_input = flag_tuple
        if type_input == "store_true":
            parser.add_argument(short_flag, long_flag, action="store_true")
        elif type_input == "list":
            parser.add_argument(short_flag, long_flag, type=int, nargs='+')
        else:
            parser.add_argument(short_flag, long_flag, type=type_input)

    parser.add_argument('-l', '--bin-processing-method', type=str, choices={'kay', 'ml', 'new'}, default='kay')

    return parser.parse_args()

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
