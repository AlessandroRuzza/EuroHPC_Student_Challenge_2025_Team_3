# EuMaster4HPC Student Challenge 2025
## Chasing the Perfect Hue: A High-Performance Dive into Graph Coloring

A parallel branch-and-bound solver for determining the chromatic number of a graph. It utilizes MPI for distributed computation and heuristic-based bounding strategies.

This project was developed for the EuroHPC Summit Student Challenge 2025, leveraging the computational power of the Slovenian supercomputer [VEGA](https://izum.si/en/vega-en/).


## Table of Contents
- [EuMaster4HPC Student Challenge 2025](#eumaster4hpc-student-challenge-2025)
  - [Chasing the Perfect Hue: A High-Performance Dive into Graph Coloring](#chasing-the-perfect-hue-a-high-performance-dive-into-graph-coloring)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage on local machine](#usage-on-local-machine)
    - [Parallel Version](#parallel-version)
      - [Optional Parameters:](#optional-parameters)
      - [Example of usage:](#example-of-usage)
      - [Output:](#output)
    - [Sequential Version](#sequential-version)
  - [Usage on Vega](#usage-on-vega)
    - [Output](#output-1)
  - [Documentation](#documentation)
  - [Results \& Benchmarks](#results--benchmarks)
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

See ```INSTALL.md```

## Usage on local machine

### Parallel Version

To run the parallel solver, use:
   ```bash
   cd code
   mpirun -n NUM_NODES python MPI/mpi.py [OPTIONS] instance outFolderPath
   ```
   Where the parameters are:
   - ```NUM_NODES```: The number of MPI nodes to utilize (**minimum of 2**)
   - ```instance```: The instance file of the graph (utilizing .col format)

   - ```outFolderPath```: The folder to store the results

#### Optional Parameters:

   - ```-h, -help```: Show a help message containing all possible options and heuristic choices
   - ```--cpusPerTask CPUSPERTASK```: Number of threads to use for each worker (ignored for non-parallel heuristics), default: 8
   - ```--branch BRANCH```: select a specific branching strategy, default: SaturationBranchingStrategy
   - ```--color COLOR```: select a specific coloring heuristic to compute the upper bound, default: ParallelBacktrackingDSatur
   - ```--clique CLIQUE```: select a specific max clique heuristic to compute the lower bound, default: ParallelDLS

#### Example of usage:
```bash
mpirun -n 4 MPI/mpi.py ../instances/anna.col ../results/my_test_run/ --cpusPerTask 16 --branch SaturationBranchingStrategy --color ParallelBacktrackingDSatur --clique ParallelDLS
 ```

#### Output:
The results will be displayed inside the specified ```outFolderPath``` folder, where there are:

- ```logs```: folder containing ```instance.log``` file 
- ```output```: folder containing ```instance.output``` file

This folder structure allows a neat file organization when running multiple instances targeting the same ```outFolderPath``` 

### Sequential Version

To utilize the parallel solver, type the following commnad:
   ```bash
python SEQUENTIAL/seq.py [OPTIONS] instance outFolderPath
   ```

The optional arguments are the same of the parallel version

## Usage on Vega

First, navigate to the ```code``` folder, where the ```.sh``` files are located.

Then, to run on Vega you have three options:
- Manual srun command, as detailed <a link="https://en-vegadocs.vega.izum.si/first-job/">here</a>:
  - First, load the required modules:
    - ```module load mpi4py```
    - ```module load SciPy-bundle```
  - Usage: ```srun [srun args] python MPI/mpi.py [OPTIONS] instance outFolder```
  - Note that the mpirun command is not required on VEGA (MPI ranks are determined by the number of tasks of the job)
  - Make sure to run at least 2 tasks (e.g. with the srun argument ```--ntasks 4```)
- Our ```run_job.sh```, to be run with sbatch command (**suggested**) (<a link="https://en-vegadocs.vega.izum.si/first-job/">sbatch command details</a>)
  - Usage: ```sbatch [optional sbatch args] run_job.sh <Instance name> <outFolderPath>```
  - The instance name should only be ```name.col```, the shell will internally complete the path to ```../instances/name.col```.
  - Edit ```run_job.sh``` if you want to add optional parameters to the execution or edit the resources requested by the job.
- Our ```launch_all_instances.sh``` shell
  - Usage: ```sh launch_all_instances.sh <outFolderPath>```
  - The shell will launch a batch job (using ```run_job.sh```) for each instance in the ```../instances/``` folder.
  - The output of each job will be collected in the output folder passed as argument (```outFolderPath```)

### Output

When using our shells (either ```run_job.sh``` or ```launch_all_instances.sh```), the ```outFolderPath``` will also contain files ```instance.stdout``` containing the console output of the job.

```launch_all_instances.sh``` also generates ```instance.job_out``` files containing the batch job output for each instance 

## Documentation

For a detailed view of the documentation, open the ```docs/index.html``` file in the local browser. <br>
The documentation was generated using Doxygen.

## Results & Benchmarks

TODO

## References

- R.M.R Lewis. A Guide to Graph Coloring - Algorithms and Applications
- Qinghua Wu. The maximum clique problems with applications to graph coloring.

## Acknowledgments

- Supervisor: Janez Povh: <janez.povh@rudolfovo.eu>

- Mentors: Mirko Rahn: <mirko.rahn@itwm.fraunhofer.de>

- Institutions: Politecnico di Milano, Sofia University, Università della Svizzera Italiana, Universitat Politècnica de Catalunya

