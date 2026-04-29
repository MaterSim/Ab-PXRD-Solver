#!/bin/bash
#SBATCH --job-name=0428
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=192G
#SBATCH --time=48:00:00

#python PXRD_solve.py --infer-spg --input GSAS_PXRD --workers 64 --output Total
#python PXRD_resume.py --input-dir GSAS_PXRD --workers 48 --output Bugs-Mono #--symmetry cubic
#python PXRD_solve.py --max-volume 1200.0 --infer-spg --use-list --input data/failed.txt --output Bugs-0415-FM3 --workers 12
python PXRD_solve.py --max-volume 1200.0 --infer-spg --use-list --input data/test.txt --output 0428 --workers 48 --use-qrs
#python PXRD_solve.py --max-volume 1200.0 --infer-spg --use-list --input data/mono.txt --output Bugs-0419 --workers 48
