# EuMaster4HPC Student Challenge 2025
## Chasing the Perfect Hue: A High-Performance Dive into Graph Coloring

A parallel branch-and-bound solver for determining the chromatic number of a graph. Utilizes MPI for distributed computation and heuristic-based bounding strategies.

## Features
- **Branch-and-Bound Framework**: Efficiently explores the solution space with smart pruning and branching strategies.
- **Heuristic Bounds**: 
  - **Lower Bound**: Maximum clique estimation (e.g. Greedy, DLS).
  - **Upper Bound**: Graph coloring heuristics (e.g. DSATUR, TabuSearch).
- **MPI Parallelization**: Distributes computation across nodes/cores for scalability.
- **Logging**: Detailed logs of bounds, branching decisions, and runtime metrics.
- **Time Limit Handling**: Gracefully exits after a user-specified time (default: 10,000 seconds).

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

To utilize the parallel solver, type the following commnad:
   ```bash
python mpirun -n NUM_NODES python MPI/mpi.py [OPTIONS] instance outFolderPath
   ```
   Where the parameters are:
   - ```NUM_NODES```: The number of MPI nodes to utilize
   - ```instance```: The instance file of the graph (utilizing .col format)

   - ```outFolderPath```: The folder in which to insert the results

It is possible to add some optional arguments, including:

   - ```-h, -help```: Show a detailed help message about the program's usage
   - ```--cpusPerTask CPUSPERTASK```: Number of threads to use for each worker, ignored for non-parallel heuristics
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

