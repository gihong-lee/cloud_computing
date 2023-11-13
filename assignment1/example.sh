#!/bin/sh

#SBATCH -J test_train
#SBATCH -p edu_2080ti 
#SBATCH -N 1   
#SBATCH -n 1
#SBATCH -o %x.o%j 
#SBATCH -e %x.e%j 
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1

module purge 
module load gcc/4.8.5 cuda/10.2 conda/tensorflow_2.4.1_cuda_10 

srun python3 <train code>.py

