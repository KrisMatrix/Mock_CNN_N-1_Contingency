# AGENTS.md

This is intended to me a machine learning project that is using mock data. In this 
project, we are employing CNN (Convolutional Neural Network), with 3 channels.

Channel 1 is an admittance-adjacency matrix.
Channel 2 is real power matrix
Channel 3 is reactive power matrix.

I want to develop a CNN model that can use the 3 channels and produce a recommendation
system for which N-1 contingency need to be studied. By "studied", I mean which buses or
branches should be considered for stress-testing using N-1 contingency analysis.

Do not use real case data, as I don't have the data for it. Create your own mock data 
generating process. A system with 35 buses should be sufficient.

Create a system of 35 buses. In this system, generate 5000 samples. Use 60 percent for 
training, 20 percent for validation, and 20 percent for testing. 

Each sample will have 3 channels. 
Channel 1 is an admittance-adjacency matrix.
Channel 2 is real power matrix
Channel 3 is reactive power matrix.

Based on these 3 channels, predict which buses or branches should be considered for stress-testing 
using N-1 contingency analysis. The output should be a matrix of the same size as the input matrix, 
with 1s indicating which buses or branches should be considered for stress-testing and 0s indicating 
which buses or branches should not be considered for stress-testing.

* Create a function to generate an electric grid model. A powerworld aux file format would be a great
choice as it is used industry wide. I would imagine, this would have bus, impedance, line, transformer,
generator, and other such data. It probably won't have real and reactive power data, but you can judge that.

* Create a function to generate data for this project based on the powerworld aux file

* Create a function to create a CNN model for this project.

* Create a function to train the CNN model for this project.

* Create a function to evaluate the CNN model for this project.

* Create a function to visualize the CNN model for this project.