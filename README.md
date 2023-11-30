PopulationSim
=============

[![Build Status](https://travis-ci.org/activitysim/populationsim.svg?branch=master)](https://travis-ci.org/ActivitySim/populationsim) [![Coverage Status](https://coveralls.io/repos/ActivitySim/populationsim/badge.png?branch=master)](https://coveralls.io/r/ActivitySim/populationsim?branch=master)<a href="https://medium.com/zephyrfoundation/populationsim-the-synthetic-commons-670e17383048"><img src="https://github.com/ZephyrTransport/zephyr-website/blob/gh-pages/img/badging/project_pages/populationsim/PopulationSim.png" width="72.6" height="19.8"></a>


PopulationSim is an open platform for population synthesis.  It emerged
from Oregon DOT's desire to build a shared, open, platform that could be
easily adapted for statewide, regional, and urban transportation planning
needs.  PopulationSim is implemented in the
[ActivitySim](https://github.com/activitysim/activitysim) framework.

## Documentation

https://activitysim.github.io/populationsim/


## Installing from this repository
The official PopulationSim releases have fallen behind ActvitySim, which is the primary dependency. As a result it has caused some incompatibilities unless package versions are locked. To avoid this, I have created this fork of PopulationSim that is compatible with the an older version of ActivitySim. In addition, I have included several bug fixes and enhancements that have not been merged into the official PopulationSim repository.

Some changes in this fork:
- Lock ActivitySim version to 1.1.3
- Added "hard_constraints" option to expansion factors forcing the max/min values to be respected. Official PopulationSim allows these limits to be exceeded slightly.
- Added various assertations throughout the code to catch errors earlier.
- Added "tqdm" progress bars to some processes.
- Converted some slow loops and pandas.apply() calls to vectorized pandas operations or slightly faster list comprehensions.

I have provided several different dependency management files to make it easier to install this fork using Conda/Mamba, pip, or Poetry.


## Installing with Conda/Mamba
The easiest way to install this fork is to use Conda or Mamba. This will install all dependencies and the forked version of PopulationSim.

```bash
# Create a new conda environment
conda create -n populationsim python=3.9
conda env create -f environment.yml
```

This will install the forked version of PopulationSim from this repository and all dependencies.


## Development install with pip
It is sometimes useful to install in an editable development mode. You may clone the repository and install with pip using the editable flag `-e`. This will install an editable version of PopulationSim from your local repository. This is useful if you want to make changes to the code and test them without having to reinstall the package.

```bash
# Create a new conda environment
conda create -n populationsim python=3.9

# Clone the repository
git clone -b v0.6.0 git@github.com:nick-fournier-rsg/populationsim.git
cd populationsim
pip install -e .
```

