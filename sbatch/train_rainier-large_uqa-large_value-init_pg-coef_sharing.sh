#!/bin/bash
#SBATCH --job-name=train_rainier-large_uqa-large_value-init_pg-coef_sharing
#SBATCH --partition=devlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=8
#SBATCH --constraint="volta32gb"
#SBATCH --time=72:00:00
#SBATCH --output="/private/home/ljc/rainier/logs/%J.%x.out"

cat $0
echo "--------------------"

module load anaconda3
source "/public/apps/anaconda3/2022.05/etc/profile.d/conda.sh"
conda activate rainier
cd /private/home/ljc/rainier/rainier
python main.py --mode train \
    --gain 3.575475037847048 --bias 0.032954977862281395 \
    --use_model_ckpt_for_value \
    --pg_coef 20.0 \
    --policy_value_sharing

