# SmartStartContinuous

SmartStart is a novel exploration method for reinforcement learning algorithms.
This work was done in collaboration with the Institute for Human and Machine
Cognition ([IHMC](https://www.ihmc.us)).

Previously this project was built for discrete state and action spaces by **Bart Keulen** [here](https://github.com/BartKeulen/smartstart). The projected here has been refactored forcing the agent classes to follow different interfaces.

Smart Start Continuous (as the name implies) allows for continuous state and action spaces.

For a detailed notes of the stuff in the project, please see `SmartStartContinuous.pdf`. It contains notes on the algorithms used, some insights in to what can be changed, and covers some of the design decision of Smart Start Continuous

## Getting Started

Documentation for the discrete part of this project can be found on the github pages of the base project, click
[here](https://bartkeulen.github.io/smartstart/).

See **Windows Setup** for dependencies (outside of Python) and pipenv installation instructions.

For the Continuous part of the project, please look at the continuous examples in the `examples` folder. 
* `DDPG_Baselines_example.py` and `SmartStart_DDPG_Baselines_example.py` show the syntax for creating agents (under `smartstart/RLAgents`), training agents with the `rlTrain` method, and saving the summary of the training results

* `DDPG_Baselines_experimenter.py` and `SmartStart_DDPG_Baselines_experimenter.py` show how to use the experimenter. For all possible combinations of parameters in `paramsGrid` in the main method the task will be run.

* `SmartStartContinuous` under `smartstart/smartexploration/smartexplorationcontinuous.py` has all the hyperparameter descriptions for SmartStart.

## Using SmartStartContinuous

The Smart Start Navigation uses a Neural Network Dynamics Estimator that relies on training data of the given environment. To generate new training data be sure to set the parameter `nnd_mb_load_existing_training_data` to `False`. To save this data set `nnd_mb_save_training_data` to `True` and be sure to set `nnd_mb_save_dir_name` to the name of the subdirectory (under `projectPath/models/NND_MB_agent/`) to save the training data.

## Windows Setup
Before using pipenv to install the pipfile.lock requirements, we will need
    
1. Place the correct pip-files into the project directory
    1. Determine which `Pipfile`/`Pipfile.lock` to use, 3 different options:
    
        1. `TensorflowCpu-NoAvx`: Tensorflow WITHOUT GPU support (CPU only) with a CPU that doesn't support AVX instructions (specifically has Tensorflow-CPU 1.5)
    
            * To check for AVX support, run [Coreinfo.exe](https://docs.microsoft.com/en-us/sysinternals/downloads/coreinfo) in terminal and `AVX -` indicates no support, `AVX             *` means it is supported
            
        2. `TensorflowCpu`: Tensorflow WITHOUT GPU support (CPU only) with a CPU that supports AVX instructions (specifically has Tensorflow-CPU latest version)
        
        3. `TensorflowGpu`: Tensorflow WITH GPU support, also needs a CPU with AVX instructions (specifically has Tensorflow-GPU latest version)
        
            * To install the prerequisites for GPU support see [TensorFlow with GPU support](https://www.tensorflow.org/install/install_windows)
            
    2. Copy `Pipfile`/`Pipfile.lock` from `/projectDirectory/alternativePipfiles/tensorflowSelection/` to the `/projectDirectory/` replacing the current files
        
1. Install MPI for Windows, currently I am able to get it to work with [Microsoft MPI v9.0.1](https://www.microsoft.com/en-us/download/details.aspx?id=56727)
    
    1. NOTE: for whatever reason I had int install both the `msmpisetup.exe` and the `msmpisetup.msi`

3. Install Python 3.6

4. Open terminal in project directory and run `pipenv install`

    * NOTE: pipenv currently has a bug, so it might say the dependencies conflict. To fix this:
        * comment out the `baselines = "*"` and the `tensorflow = "*"`/`tensorflow-gpu = "*"` in the `Pipfile` you dragged into the project directory 
        * run `pipenv install` again, it should resolve properly
        * run `pipenv shell` to load into the virtual environment (this doesn't work in mingW but works with Cmder)
        * run `pip install tensorflow`/`pip install tensorflow-gpu` and `pip install baselines`
    

**Latest code can be found in `continuous_ss_framework`. The docs and gh-pages have not been updated yet.**
