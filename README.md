# Panodisplay
Panoramic display for visual neuroscience

# What it is

![picture](https://github.com/aphilip442/Panodisplay/blob/master/panodisplay.png)

A spherical display for visual stimulation and 3D navigation. We use it with mice.

For more informations, see 'User guide'

# Create a conda environment with the necessary packages:

Download Git: https://git-scm.com/downloads


Download Miniconda (if you don't have Anaconda already installed): https://docs.conda.io/en/latest/miniconda.html


Clone the repository on your computer


Open Git Bash in your repository (right-click, 'Git Bash here')


In Git Bash:

    conda create -n psychopanda

    source activate psychopanda

    conda install python=3.7 matplotlib jupyter numpy opencv psychopy -c menpo -c cogsci

    pip install panda3d==1.10.4.1
    
    
Then you can use 'jupyter notebook' to access the scripts and run them

# TODOS

* Add recent picture of the working display

* Add Unity code

* Complete User Guide