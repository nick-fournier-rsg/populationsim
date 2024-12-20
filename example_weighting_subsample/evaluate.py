# Provides functions to load results from grid search and plot them
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def collate_results(run_dir):
    """Collate the results from the runs"""
    results = []
    weight_fname = "final_summary_hh_weights.csv"

    for runname in os.listdir(run_dir):
        weights_path = os.path.join(run_dir, runname, "output", weight_fname)
        params = parse_dirname(runname)
        shortname = "-".join([f"{v.replace('%', '')}" for v in params.values()])
        
        weights = pd.read_csv(weights_path)
        weights["run"] = shortname
        weights["sample_size"] = params["sample_size"]
        weights["initial_perturb"] = params["initial_perturb"]
        weights["max_exp"] = params["max_exp"]
        results.append(weights)
        
    results_df = pd.concat(results)
    
    # Read the base data and append
    base_df = pd.read_csv(os.path.join(os.path.dirname(run_dir), 'output', weight_fname))
    base_df["run"] = "base"
    base_df["sample_size"] = "100%"
    base_df["initial_perturb"] = "0%"
    base_df["max_exp"] = "8"
    
    # Append the base data
    results_df = pd.concat([base_df, results_df])   
        
    return results_df

def parse_dirname(dirname):
    """ Extract the parameters from the directory name """
    params = dirname.split("-")
    params = {p.split("=")[0]: p.split("=")[1] for p in params}
    
    return params


def plot_correllation(results):
    """Plot a set of correlation XY scatter plots against the base run"""
    
    params = ["sample_size", "initial_perturb", "max_exp"]
    pairs = [
        ("sample_size", "initial_perturb"),
        ("sample_size", "max_exp"),
        ("initial_perturb", "max_exp")
    ]
    
    # Rename the weight column
    results = results.rename(columns={"puma_group_balanced_weight": "hh_weight"})
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
        plot_dir = os.path.join(os.path.dirname(__file__), "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_path = os.path.join(plot_dir, f"correlation_plot_{param1}-vs-{param2}.png")
        plt.savefig(plot_path, dpi=300)
        
    print("Done plotting")
        
    
matplotlib.use('TkAgg')

    


if __name__ == "__main__":
    # Collate the results
    run_dir = os.path.join(os.path.dirname(__file__), "runs")
    results = collate_results(run_dir)
    
    plot_correllation(results)
    
