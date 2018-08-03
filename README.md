# SmartStart

SmartStart is a novel exploration method for reinforcement learning algorithms.
This work was done in collaboration with the Institute for Human and Machine
Cognition ([IHMC](https://www.ihmc.us)).

## Getting Started

Full documentation can be found on the github pages for this project, click
[here](https://bartkeulen.github.io/smartstart/).

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
        
2. Install MPI for Windows, currently I am able to get it to work with [Microsoft MPI v9.0.1](https://www.microsoft.com/en-us/download/details.aspx?id=56727)
    
    1. NOTE: for whatever reason I had int install both the `msmpisetup.exe` and the `msmpisetup.msi`

    

**Latest code can be found in Develop. The docs and gh-pages have not been updated yet.**
