#!/bin/bash

#SBATCH --job-name=herzigMA1

#SBATCH --cpus-per-task=4
#SBATCH --partition=test

#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:2

#SBATCH --time=15:00
#SBATCH --error=job.%j.err

#SBATCH --output=job.%J.out

#SBATCH --mail-user=robert.herzig@student.uni-tuebingen.de

singularity exec --nv /common/singularityImages/TCML-Cuda9_0Tensorflow1_6.simg python3 main_controller.py

echo DONE
