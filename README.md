# SmartStart

SmartStart is a novel exploration method for reinforcement learning algorithms.
This work was done in collaboration with the Institute for Human and Machine
Cognition ([IHMC](https://www.ihmc.us)).

## Getting Started

Full documentation can be found on the github pages for this project, click
[here](https://bartkeulen.github.io/smartstart/).

## Windows Setup
Before using pipenv to install the pipfile.lock requirements, we will need
1. `make` command for windows (sepcifically needed for `baselines`, open-ai's python reinforcement learning algorithms package) I got the command through [MSYS2](https://github.com/msys2/msys2/wiki/MSYS2-installation)

    1. Install MSYS2 (easiest way is through [Chocolatey](https://chocolatey.org/packages/msys2), once Chocolatey is installed
    you can use `choco install msys2`)
    
    2. Add MSYS2 to Path environment variable (for me I added specifically the path `C:\tools\msys64\usr\bin` to the end of my path variable)
        1. Go to `Control Panel > System and Security > System > Advanced system settings > Environment Variables`
        2. Under System Variables, select the variable `path`, click edit
        3. In the window, click `New` and then `Browse...`, select the `C:\tools\msys64\usr\bin` and click `ok`
        
    3. Now in a terminal session, run the command `pacman -Syuu` to update the package databases
    
    4. Now run `pacman -Sy make` to install the make command
	
	   

**Latest code can be found in Develop. The docs and gh-pages have not been updated yet.**
