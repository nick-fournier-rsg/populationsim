
import os
import logging
import sys

import argparse
from activitysim.core import config
from populationsim import steps

from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

from activitysim.core.config import handle_standard_args
from activitysim.core.tracing import print_elapsed_time
from activitysim.cli.run import add_run_args, run
from activitysim.core.config import setting
from populationsim import lp
from populationsim import multi_integerizer


# # Add (and handle) 'standard' activitysim arguments:
# #     --config : specify path to config_dir
# #     --output : specify path to output_dir
# #     --data   : specify path to data_dir
# #     --models : specify run_list name
# #     --resume : resume_after
# handle_standard_args()

# parser = argparse.ArgumentParser()
# add_run_args(parser)
# args = parser.parse_args()
# args.working_dir = os.path.dirname(__file__)

# tracing.config_logger()

# t0 = print_elapsed_time()

# logger = logging.getLogger('populationsim')

# logger.info("GROUP_BY_INCIDENCE_SIGNATURE: %s"
#             % setting('GROUP_BY_INCIDENCE_SIGNATURE'))
# logger.info("INTEGERIZE_WITH_BACKSTOPPED_CONTROLS: %s"
#             % setting('INTEGERIZE_WITH_BACKSTOPPED_CONTROLS'))
# logger.info("SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS: %s"
#             % setting('SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS'))
# logger.info("meta_control_data: %s"
#             % setting('meta_control_data'))
# logger.info("control_file_name: %s"
#             % setting('control_file_name'))

# logger.info("USE_CVXPY: %s" % lp.use_cvxpy())
# logger.info("USE_SIMUL_INTEGERIZER: %s" % multi_integerizer.use_simul_integerizer())


# # get the run list (name was possibly specified on the command line with the -m option)
# run_list_name = inject.get_injectable('run_list_name', 'run_list')

# # run list from settings file is dict with list of 'steps' and optional 'resume_after'
# run_list = setting(run_list_name)
# assert 'steps' in run_list, "Did not find steps in run_list"

# # list of steps and possible resume_after in run_list
# steps = run_list.get('steps')
# resume_after = run_list.get('resume_after', None)

# if resume_after:
#     print("resume_after", resume_after)

# pipeline.run(models=steps, resume_after=resume_after)


# # tables will no longer be available after pipeline is closed
# pipeline.close_pipeline()

# t0 = ("all models", t0)

@inject.injectable()
def log_settings():

    return [
        'multiprocess',
        'num_processes',
        'resume_after',
        'GROUP_BY_INCIDENCE_SIGNATURE',
        'INTEGERIZE_WITH_BACKSTOPPED_CONTROLS',
        'SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS',
        'meta_control_data',
        'control_file_name',
        'USE_CVXPY',
        'USE_SIMUL_INTEGERIZER'
    ]


if __name__ == '__main__':

    assert inject.get_injectable('preload_injectables', None)

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    args.working_dir = os.path.dirname(__file__)
    
    sys.exit(run(args))