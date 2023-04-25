#! /bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
python -m visdom.server
