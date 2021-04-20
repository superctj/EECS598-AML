#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=mnli
#SBATCH --account=jiadeng
#SBATCH --partition=jiadeng
#SBATCH --gpus=2
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20g
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=BEGIN,END

# The application(s) to execute along with its input arguments and options:

python RewritingSampler_attack.py