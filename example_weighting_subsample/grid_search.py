
import os
import shutil
import itertools
import sys
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

import subprocess
from concurrent.futures import ProcessPoolExecutor

# Add (and handle) 'standard' activitysim arguments:
#     --config : specify path to config_dir
#     --output : specify path to output_dir
#     --data   : specify path to data_dir
#     --models : specify run_list name
#     --resume : resume_after

class GridSearch:    
    # Param grid
    sample_sizes: list = [1] #[1, 0.99, 0.9, 0.7]
    initial_perturbs: list = [0] # [0, 0.1, 0.5]
    max_exp_fact: list =  [4, 16] # [2, 3, 4, 5, 6, 7, 8, 16, 32, np.inf]
        
    # Base data and settings
    base_settings: dict = None
    base_seed: pd.DataFrame = None
    np.random.seed(0)

    
    def __init__(self):

        # Read in current settings and seed data
        this_dir = os.path.dirname(__file__)
                
        with open(os.path.join(this_dir, "configs", "settings.yaml"), "r") as f:
            self.settings = yaml.load(f, Loader=yaml.FullLoader)
    
        # set abslute upper limit to inf
        self.settings["absolute_upper_bound"] = np.inf
            
        # Read in the seed data
        self.seed_data = pd.read_csv(os.path.join(this_dir, "data", "seed_households.csv"))
           
    def update_args(self, current_params):
        # Multiply the params by 100
        sample_size = int(current_params[0] * 100)
        initial_perturb = int(current_params[1] * 100)
        max_exp = current_params[2]

        # Prepare the run id label
        sample_size = int(current_params[0] * 100)
        initial_perturb = int(current_params[1] * 100)
        max_exp = current_params[2]
                
        run_label = (
            f"sample_size={sample_size}%-initial_perturb={initial_perturb}%-max_exp={max_exp}"
        )
        
        # Update args
        working_dir = os.path.join(os.path.dirname(__file__), "runs", run_label)
        current_args = {
            "working_dir": working_dir,
            "config": os.path.join(working_dir, "configs"),
            "output": os.path.join(working_dir, "output"),
            "data": os.path.join(working_dir, "data")
        }
        
        for v in current_args.values():
            if not os.path.exists(v):
                os.makedirs(v)
                
        return current_args
        
    
    def prepare_data(self, current_params, current_args):
        
        # Unchanged inputs
        unchanged = [
            "configs/controls.csv",
            "data/control_totals_pumas.csv",
            "data/geo_cross_walk.csv"
        ]
        
        # Copy the unchanged files
        for file in unchanged:
            base_path = os.path.join(os.path.dirname(__file__), file)
            new_path = os.path.join(current_args['working_dir'], file)

            shutil.copy(base_path, new_path)

    
        # Update the settings
        new_settings = self.settings.copy()
        new_settings["max_expansion_factor"] = current_params[2]
        new_settings["min_expansion_factor"] = 1 / current_params[2]
        
        # Write the settings        
        with open(os.path.join(current_args['config'], "settings.yaml"), "w") as f:
            yaml.dump(new_settings, f)

        # Sample the seed data
        seed_data = self.seed_data.sample(frac=current_params[0]).copy()
        
        # Rescale the initial weights to match original total
        og_total = self.seed_data['initial_weight'].sum()
        sub_total = seed_data['initial_weight'].sum()
        seed_data['initial_weight'] *= og_total / sub_total
        
        # Add the perturbation of initial weights
        perturbs = 1 + np.random.uniform(-current_params[1], current_params[1], seed_data.shape[0])
        
        seed_data['initial_weight_orig'] = seed_data['initial_weight']
        seed_data['initial_weight'] *= perturbs
        
        # Write the seed data
        seed_data.to_csv(os.path.join(current_args['data'], "seed_households.csv"), index=False)


    def runner(self, sample_size, initial_perturb, max_exp):          

        msg = (
            f"sample_size: {sample_size}, "
            f"initial_perturb: {initial_perturb}, "
            f"max_exp_fact: {max_exp}, "
            f"min_exp_fact: {1 / max_exp}"
        )

        # Update the args and current params
        current_params = (sample_size, initial_perturb, max_exp)        
        current_args = self.update_args(current_params)
        
        # Skip if output already exists
        if os.path.exists(os.path.join(current_args['output'], "final_summary_hh_weights.csv")):
            print("Output already exists for:", msg)
            return

        print("Running:", msg)

        # Prepare the run data files
        self.prepare_data(current_params, current_args)
        
        # Get the path to the runner
        runner_path = os.path.join(os.path.dirname(__file__), "run_populationsim.py")
        
        # Convert to args list for string concatenation
        current_args['working_dir']
        
        args_list = []
        for k, v in current_args.items():
            args_list.extend([f"--{k}", v])
        
        # Build the command
        command = [sys.executable, runner_path] + ['--working_dir', current_args['working_dir']]
        
        # Call subprocess to run run_populationsim.py
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print("Completed:", msg)
        
        if result.returncode != 0:
            print("Error running command")
            return

    def run(self):
        param_grid = [self.sample_sizes, self.initial_perturbs, self.max_exp_fact]        

        n_workers = int(os.cpu_count() // (4 / 3))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit tasks to the executor
            futures = [
                executor.submit(self.runner, sample_size, initial_perturb, max_exp)
                for sample_size, initial_perturb, max_exp in itertools.product(*param_grid)
            ]

            # Optionally, process results or exceptions
            for future in futures:
                try:
                    future.result()  # This will raise exceptions if any occurred
                except Exception as e:
                    print(f"Error during execution: {e}")


if __name__ == '__main__':
    
    gs = GridSearch()
    gs.run()
    