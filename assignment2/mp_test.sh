#!/bin/sh
#SBATCH -J mp_test
#SBATCH -p edu_2080ti 
#SBATCH -N 1         # total number of Nodes
#SBATCH -n 1         # total number of process
#SBATCH -o %x.o%j 
#SBATCH -e %x.e%j 
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:2 # number of gpu for each node. total numder of gpu = gres x nodes

module purge 
module load gcc/4.8.5 cuda/10.2 conda/tensorflow_2.4.1_cuda_10 

srun python3 ./mp_test.py
