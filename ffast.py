from config import *
from input_signal import *
import numpy as np
import argparse

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
    if config.experiment_mode:
        input_signal = ExperimentInputSignal(config)
        output_signal = ExperimentOutputSignal(config)
    else:
        input_signal = CustomizedInput(config)
        output_signal = CustomizedOutput(config)

if __name__ == "__main__":
    main()