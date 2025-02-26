# EuMaster4HPC Student Challenge 2025
## Chasing the Perfect Hue: A High-Performance Dive into Graph Coloring

A parallel branch-and-bound solver for determining the chromatic number of a graph. It utilizes MPI for distributed computation and heuristic-based bounding strategies.

This project was developed for the EuroHPC Summit Student Challenge 2025, leveraging the computational power of the Slovenian supercomputer [VEGA](https://izum.si/en/vega-en/).


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Parallel Version](#parallel-version)
  - [Sequential Version](#sequential-version)
  - [Usage on Vega](#usage-on-vega)
- [Documentation](#documentation)
- [Results & Benchmarks](#results--benchmarks)
- [References](#references)
- [Acknowledgments](#acknowledgments)


## Features
- **Branch-and-Bound Framework**: Efficiently explores the solution space with pruning and branching strategies.
- **Heuristic Bounds**: 
  - **Lower Bound**: Maximum clique estimation using techniques ranging from basic greedy methods to advanced adaptive strategies like DLS.
  - **Upper Bound**: Graph coloring heuristics, including DSATUR, TabuSearch, and BacktrackingDSatur.
- **MPI Parallelization**: Distributes computation across nodes/cores for scalability, using a manager/worker pattern.
- **Logging**: Tracks bounds, branching decisions, and runtime metrics for each node.


## Installation

### Prerequisites
- **MPI** (mpi4py)
- **Python 3**

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Rudolfovoorg/EuroHPC_Student_Challenge_2025_Team_3.git
   cd [name]
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 

### Parallel Version

To run the parallel solver, use:
   ```bash
python mpirun -n NUM_NODES python MPI/mpi.py [OPTIONS] instance outFolderPath
   ```
   #### Arguments:
   - ```NUM_NODES```: The number of MPI nodes to utilize
   - ```instance```: The instance file of the graph (utilizing .col format)

   - ```outFolderPath```: The folder to store the results

#### Optional Parameters:

   - ```-h, -help```: Show a help message
   - ```--cpusPerTask CPUSPERTASK```: Number of threads to use for each worker (ignored for non-parallel heuristics)
   - ```--branch BRANCH```: select a specific branching strategy, default: SaturationBranchingStrategy
   - ```--color COLOR```: select a specific coloring heuristic to compute the upper bound, default: ParallelBacktrackingDSatur
   - ```--clique CLIQUE```: select a specific max clique heuristic to compute the lower bound, default: ParallelDLS

#### Example of usage:
```
python mpirun -n 4 MPI/mpi.py ../instances/anna.col ../results/2h_test_output/ --cpusPerTask 16 --branch SaturationBranchingStrategy --color ParallelBacktrackingDSatur --clique ParallelDLS
 ```

#### Output:
The results will be displayed inside the specified ```outFolderPath``` folder, where there are:

- ```log```:
- ```file.output```: 

### Sequential Version

To utilize the parallel solver, type the following commnad:
   ```bash
python python SEQUENTIAL/seq.py [OPTIONS] instance outFolderPath
   ```

The optional arguments are the same of the parallel version

### Usage on Vega

To run on Vega, you have to 


## Documentation

For a detailed view of the documentation, look at the ```docs/``` folder

## Results & Benchmarks

TODO

## References



## Acknowledgments

- Supervisor: Janez Povh: <janez.povh@rudolfovo.eu>

- Mentors: Mirko Rahn: <mirko.rahn@itwm.fraunhofer.de>

- Institutions: Politecnico di Milano, Sofia University, Universitat Politècnica de Catalunya

