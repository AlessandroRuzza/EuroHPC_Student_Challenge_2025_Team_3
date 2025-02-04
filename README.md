# EuMaster4HPC Student Challenge 2025
## Chasing the Perfect Hue: A High-Performance Dive into Graph Coloring

A parallel branch-and-bound solver for determining the chromatic number of a graph. Utilizes MPI for distributed computation and heuristic-based bounding strategies.


## Features
- **Branch-and-Bound Framework**: Efficiently explores the solution space with smart pruning.
- **Heuristic Bounds**: 
  - **Lower Bound**: Maximum clique estimation using greedy/approximation algorithms.
  - **Upper Bound**: Graph coloring heuristics (e.g., DSATUR, RLF).
- **MPI Parallelization**: Distributes computation across nodes/cores for scalability.
- **Logging**: Detailed logs of bounds, branching decisions, and runtime metrics.
- **Time Limit Handling**: Gracefully exits after a user-specified time (default: 10,000 seconds).

## Installation
- not yet

### Prerequisites
- **MPI** (OpenMPI or MPICH)
- **C++17 Compiler** (GCC, Clang, or MSVC)
- **CMake** (≥3.15)
- **Python 3** (for benchmark scripts, optional)

- I am guessing xd

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/
   cd [name]

## Usage 

## Documentation

## Results & Benchmarks

## References

## Acknowledgments

- Supervisor: Prof. Janez Povh

- Mentors: [Mentor Names]

- Institutions: [Universities]
