# EuMaster4HPC Student Challenge 2025
## Chasing the Perfect Hue: A High-Performance Dive into Graph Coloring

A parallel branch-and-bound solver for determining the chromatic number of a graph. Utilizes MPI for distributed computation and heuristic-based bounding strategies.

## Features
- **Branch-and-Bound Framework**: Efficiently explores the solution space with smart pruning.
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


## Documentation

## Results & Benchmarks

## References


## Acknowledgments

- Supervisor: Prof. Janez Povh janez.povh@rudolfovo.eu or roman.kuzel@rudolfovo.eu.

- Mentors: Rahn, Mirko <mirko.rahn@itwm.fraunhofer.de>

- Institutions: [Universities]
