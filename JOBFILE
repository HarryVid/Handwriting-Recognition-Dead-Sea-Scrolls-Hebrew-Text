#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=8GB
#SBATCH --job-name=HWR-Group5
#SBATCH --output=HWR_ConsoleLog

module load Python/3.7.4-GCCcore-8.3.0
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
python -m pip install -r requirements.txt --user


python ./main.py --image ./test_images