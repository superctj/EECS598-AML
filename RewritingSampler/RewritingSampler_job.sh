#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=mnli
#SBATCH --account=eecs598s007w21_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000m
#SBATCH --mail-type=BEGIN,END

# The application(s) to execute along with its input arguments and options:

/bin/hostname
python RewritingSampler_attack.py