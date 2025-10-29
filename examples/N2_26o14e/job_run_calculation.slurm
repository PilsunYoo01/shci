#!/bin/bash
#SBATCH -p develop
#SBATCH -t 336:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --job-name=check
#SBATCH --output=Result-%x.%j.out
#SBATCH --error=Result-%x.%j.err

#cd $SLURM_SUBMIT_DIR

#source /home/yoo_ps/apps/QunovaPulsar-qiskit-functions-hivqe-release-2025Oct/activate

#python hivqe_runner.py 2>&1 | tee output.hivqe

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export LD_LIBRARY_PATH=/home/_qamd/apps/openmpi-3.0.6_build/lib:$LD_LIBRARY_PATH

rm wf_eps1_*

/home/_qamd/apps/openmpi-3.0.6_build/bin/mpirun -np $(($SLURM_NTASKS_PER_NODE*$SLURM_NNODES)) /home/_qunova/apps/shci_singleshot/shci > output.dat
