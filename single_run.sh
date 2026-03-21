#!/bin/bash
#SBATCH --job-name=TestS
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

python PXRD_solve.py --infer-spg --spg-backend smart-cell --input Examples --workers 8 --output Results-s2
python PXRD_solve.py --infer-spg --spg-backend smart-cell --input Examples --workers 8 --output Results-s3
