#!/bin/bash
#SBATCH --job-name=2gtdist
#SBATCH --output=zlogs/slurm.log
#SBATCH --error=zlogs/slurm.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -p rtx8000,v100
#SBATCH --mem=200000M
#SBATCH --time=0:40:00
#SBATCH --mail-type=ALL

source $HOME/.bashrc

srcid=ID_TO_REPLACE
mark=eu001_182_t2k_prefn4_r661_h1h_on_gpt_gene_teacher820_top1h_b20_r9005_dist${srcid}
#mark=eu001_182_t2k_prefn4_r661_concat_on_gpt_gene_teacher820_top1h_b10_r9001_dist${srcid}

config_file=configs.student_tf_1h_for_gene_teacher_dist_two_level_euro6_clip
#trained_teacher_model=outputs/dv031_990k_teacher200_bs280_r230/all1/model_best.ckpt
trained_teacher_model=outputs/euro6v001_182_t2k_bs32_r660_r820/all${srcid}/model_best.ckpt

output_path=outputs/$mark
log=logs/teacher/logteacherg$mark
err=logs/teacher/errteacherg$mark

mkdir -p $output_path logs/teacher

pythont=python

#CUDA_VISIBLE_DEVICES=${gpuid} \
$pythont gpt2_train.py \
  --do-train \
  --do-eval \
  --pretrained-model-name=gpt2-small \
  --trained_teacher_model=${trained_teacher_model} \
  --config-train=$config_file \
  --output-dir=$output_path \
  --teacher_generate_mode=1 \
  > $log 2> $err

rm -f ${output_path}/model.ckpt
rm -f ${output_path}/model_best.ckpt
