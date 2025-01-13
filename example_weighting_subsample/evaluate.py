# Provides functions to load results from grid search and plot them
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
matplotlib.use('TkAgg')


class Evaluate:
    root_dir: Path = None
    run_dir: Path = None
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.run_dir = root_dir / "runs"
    
    def collate_results(self):
        """Collate the results from the runs"""
        results = []
        weight_fname = "final_summary_hh_weights.csv"

        # Read the results from each run
        for runname in os.listdir(self.run_dir):
            weights_path = self.run_dir / runname / "output" / weight_fname
            params = self.parse_dirname(runname)
            shortname = "-".join([f"{v.replace('%', '')}" for v in params.values()])
            
            if not os.path.exists(weights_path):
                continue
            
            weights = pd.read_csv(weights_path)
            weights["run"] = shortname
            weights["sample_size"] = params["sample_size"]
            weights["initial_perturb"] = params["initial_perturb"]
            weights["max_exp"] = params["max_exp"]
            results.append(weights)
            
        results_df = pd.concat(results)
        
        # Read the base data and append        
        base_df = pd.read_csv(self.root_dir / "output" / weight_fname)
        base_df["run"] = "base"
        base_df["sample_size"] = "100%"
        base_df["initial_perturb"] = "0%"
        base_df["max_exp"] = "8"
        
        # Append the base data
        self.results = pd.concat([base_df, results_df])   

    def parse_dirname(self, dirname):
        """ Extract the parameters from the directory name """
        params = dirname.split("-")
        params = {p.split("=")[0]: p.split("=")[1] for p in params}
        
        return params

    def plot_correllation(self):
        """Plot a set of correlation XY scatter plots against the base run"""
        
        params = ["sample_size", "initial_perturb", "max_exp"]
        pairs = [
            ("sample_size", "initial_perturb"),
            ("sample_size", "max_exp"),
            ("initial_perturb", "max_exp")
        ]
        
        # Rename the weight column
        results = self.results.rename(columns={"puma_group_balanced_weight": "hh_weight"})
        base_results = results[results["run"] == "base"]
        test_results = results[results["run"] != "base"]
        
        # For each pair, compare the scenario to the base
        for param1, param2 in pairs:
            
            # Hold the missing parameter as constant
            missing_param = list(set(params) - set([param1, param2]))[0]
            const_param = str(base_results[missing_param].iloc[0])
        
            test_df = test_results[test_results[missing_param] == const_param]
            
            groups = test_df.groupby([param1, param2])
            
            # Prepare a grid layout
            n_groups = len(groups)
            cols = 3  # Adjust the number of columns as needed
            rows = (n_groups + cols - 1) // cols  # Calculate the number of rows
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
            axes = axes.flatten()  # Flatten to easily iterate over axes

            for idx, ((param1_val, param2_val), df) in enumerate(groups):

                # Combine the base and test data weight columns into a single dataframe
                df_compare = df.merge(base_results, on="hh_id", suffixes=("", "_base"), how="inner")

                # Create scatterplot
                sns.scatterplot(x="hh_weight_base", y="hh_weight", data=df_compare, ax=axes[idx], s=10, alpha=0.5)
                axes[idx].set_title(f"{param1}={param1_val}, {param2}={param2_val}", fontsize=10)

                # Set equal aspect ratio and make the plot square
                axes[idx].set_aspect('equal', adjustable='box')
                
                # Add a diagonal line
                lims = [
                    df_compare["hh_weight"].min(),  # min of both axes
                    df_compare["hh_weight"].max(),  # max of both axes
                ]
                axes[idx].plot(lims, lims, 'k-', alpha=0.75, zorder=0)

                # Remove individual axis labels
                axes[idx].set_xlabel("")
                axes[idx].set_ylabel("")
                    
            # Remove unused subplots if groups < rows * cols
            for idx in range(len(groups), len(axes)):
                fig.delaxes(axes[idx])
                
            # Add shared labels for the whole grid
            fig.supxlabel("Test Weight (hh_weight)", fontsize=12)
            fig.supylabel("Base Weight (hh_weight_base)", fontsize=12)

            # Adjust layout and show plot
            fig.suptitle(f"Correlation for {param1} and {param2} (holding {missing_param}={const_param})", fontsize=16)
            plt.subplots_adjust(hspace=0.5)  # Add vertical spacing
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to leave space for suptitle
            
            # Plot dir
            plot_dir = self.root_dir / "plots"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plot_path = plot_dir / f"correlation_plot_{param1}-vs-{param2}.png"
            plt.savefig(plot_path, dpi=300)
            
        print("Done plotting")

    def bias_variance_decomposition(self):
        """Extract and plot the bias-variance decomposition"""
        # Read the control targets and incidence table
        targets = pd.read_csv(self.root_dir / "data" / "control_totals_pumas.csv")
        incidence = pd.read_csv(self.root_dir / "data" / "seed_households.csv")
        
        # Set the index
        targets.set_index("puma_group", inplace=True)
        incidence.set_index("hh_id", inplace=True)
        
        # Form into a matrix
        a_mat = incidence[targets.columns]
        
        bias_var = {}
        # For each result, calculate variance and the fit against the control targets (NRMSE)
        for run, df in self.results.groupby('run'):
            
            # Multiply the weights vector by the incidence table
            wts = df.set_index("hh_id")['puma_group_balanced_weight']
            
            # Multiply the weights by the incidence table and add puma_group column
            results = (a_mat.transpose() * wts).transpose().join(
                incidence["puma_group"]
            ).groupby("puma_group").sum()
            
            # Calculate fit and variance
            # Fit NRMSE
            nrmse = np.sqrt(np.mean((results - targets) ** 2)) / np.mean(targets)
            
            # Variance of weights (normalized by mean)
            variance = np.var(wts) / np.mean(wts)

            # Parse the run name, convert to float
            params = run.split("-")
            
            # Create a labeled name Sample size, Initial perturb, Max exp
            if len(params) == 1:
                run_label = params[0]
                params = ["100", "0", "8"]
            else:
                run_label = (
                    "Sample size: " + params[0] + 
                    "%, Initial perturb: " + params[1] +
                    "%, Max exp: " + params[2]
                )
            
            # Result dictionary
            result_dict = {k: float(v) for k, v in zip(["sample_size", "initial_perturb", "max_exp"], params)}
            result_dict = {**result_dict, "nrmse": nrmse, "variance": variance}
            
            bias_var[run_label] = result_dict
            
        self.bias_var = pd.DataFrame.from_dict(bias_var, orient="index")
        
        plot_df = self.bias_var.copy()
        plot_df["sample_size"] = plot_df["sample_size"].astype(str) + "%"
        
        # Plotly scatter plot
        fig = px.scatter(
            plot_df,
            x="nrmse",
            y="variance",
            color="sample_size",
            symbol="initial_perturb",
            labels={"nrmse": "Fit (NRMSE)", "variance": "Variance (normalized)"},
            title="Bias-Variance Decomposition",
            text=self.bias_var.index,
            color_discrete_sequence=px.colors.qualitative.Set1  # Use a categorical color scale
        )

        # Make text only visible on hover
        fig.update_traces(
            mode="markers",
            marker=dict(size=10, opacity=0.5),
            hovertemplate='<b>%{text}</b><br><br>NRMSE: %{x}<br>Variance: %{y}<extra></extra>'
        )
        # Save to html
        plot_dir = self.root_dir / "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig.write_html(plot_dir / "bias_variance_plot.html")
        
        
    
matplotlib.use('TkAgg')

    


if __name__ == "__main__":
    # Collate the results
    root_dir = Path(__file__).parent
    eval = Evaluate(root_dir)
    eval.collate_results()
    eval.bias_variance_decomposition()
    eval.plot_correllation()
    
