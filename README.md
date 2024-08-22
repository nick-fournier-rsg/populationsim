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

**Important** PopulationSim requires Python version 3.9. You will need to install python 3.9 even if you have a newer version of Python installed.

The official PopulationSim releases have fallen behind ActvitySim, which is the primary dependency. As a result it has caused some incompatibilities unless package versions are locked. To avoid this, I have created this fork of PopulationSim that is compatible with the an older version of ActivitySim. In addition, I have included several bug fixes and enhancements that have not been merged into the official PopulationSim repository.

Some changes in this fork:
- Lock ActivitySim version to 1.1.3
- Added "hard_constraints" option to expansion factors forcing the max/min values to be respected. Official PopulationSim allows these limits to be exceeded slightly.
- Added various assertations throughout the code to catch errors earlier.
- Added "tqdm" progress bars to some processes.
- Converted some slow loops and pandas.apply() calls to vectorized pandas operations or slightly faster list comprehensions.

I have provided several different dependency management files to make it easier to install this fork using Conda/Mamba, pip, or Poetry.

### _Python Environment Update_!
In an effort to enhance stability I have used Poetry to manage the dependencies for this fork. I have migrated away from the older setup.py, setup.cfg, and requirements.txt files in lieu of the more modern and simpler pyproject.toml file. I also provided an environment.yml for Conda if that is preferred.


## Setup a PopulationSim environment using Poetry
Poetry is a modern dependency management tool used for both environment management and package management. It is similar to Conda but is more lightweight and is specifically designed for Python projects. However, since this project has a lot of outdated and deprecated dependencies, we have to be very careful about python versions.

**Important** Reminder that PopulationSim requires Python version 3.9. You will need to install python 3.9 even if you have a newer version of Python installed.

First, [install Poetry](https://python-poetry.org/docs/#installation). After installing Poetry, you will create a new environment using python 3.9 and install PopulationSim and its dependencies into that environment.

Decide where your new environment will be located. By default Poetry will create a new environment in a shared location your user folder e.g., `C:\Users\<username>\AppData\Local\pypoetry\Cache\virtualenvs\<populationsim-env>`. But if you want to create the environment in your project location, so there is a new environment for each project, you can set the following configuration:

_Optional:_
```bash
poetry config virtualenvs.in-project true
```

After installing python 3.9 and poetry, we must tell poetry to use python 3.9 when creating the environment:
```bash
py --list-paths
```

This will show you the paths to your python installations. You can then use the path to the python 3.9 installation to create a new environment.

```bash
poetry env use <path to python 3.9>
```

This will create a new environment using python 3.9. You can then add PopulationSim and its dependencies into this environment.

```bash
poetry add git+https://github.com/nick-fournier-rsg/populationsim.git@v0.6.2
```

You can then enter the environment using the following command:

```bash
poetry shell
```


## Installing from GitHub
You can install this fork directly from GitHub using pip. This will install all dependencies and the forked version of PopulationSim to your *current* Python environment.

```bash
pip install git+https://github.com/nick-fournier-rsg/populationsim.git@v0.6.2#egg=populationsim
```

This will install the forked version of PopulationSim from this repository and all dependencies.

