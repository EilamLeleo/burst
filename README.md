# burst
Code for creation of main results of Burst Control, Leleo &amp; Segev 2021

Python scripts were used to run simulations of reconstructed L5PC activity with relevant synaptic activation, for burst initiation.
Some scripts with "cluster" suffix were ran using lab processing cluster. Others were primarily used for data visualisation, 
either by shorter simulation blocks (narrow parameter range, single run for voltage traces, etc.) or by importing data saved 
from multiple cluster runs. Advise inline comments for finding specific code for the different graphs and figures.

For running the scripts install NEURON from https://neuron.yale.edu/
Then extract files in zip and compile using command line nrnivmodl
Finally run any python script for neuron simulation and results visualization.
