from .ffast import FFAST, make_args
from .config import Config
from .input_signal import ExperimentInputSignal
from .output_signal import ExperimentOutputSignal

import time
import numpy as np
import argparse

if __name__ == "__main__":
    args = make_args()
    time_initial = int(np.round(time.time() * 1000))
    # np.random.seed(0)
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