#!/bin/bash

#SBATCH --job-name=runtime_fast
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=scavenge
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=ALL

module load miniconda
conda activate gspa

for count in 100 1000 5000 25000 50000 100000;
    do for reduced in False True;
	    do echo ${count} ${reduced}; python 0.0_fast_GSPA.py ${count} ${reduced};
    done
done
