#!/bin/sh
#SBATCH -J dp_test
#SBATCH -p edu_2080ti 
#SBATCH -N 2         # total number of Nodes
#SBATCH -n 2         # total number of process (# of process have to same with # of nodes in dp)
#SBATCH -o %x.o%j 
#SBATCH -e %x.e%j 
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1 # number of gpu for each node. total numder of gpu = gres x node

module purge 
module load gcc/4.8.5 cuda/10.2 conda/tensorflow_2.4.1_cuda_10 

srun python3 ./dp_test.py
