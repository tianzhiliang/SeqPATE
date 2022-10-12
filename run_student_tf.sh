#!/bin/bash
#SBATCH --job-name=811stu_tf
#SBATCH --output=zlogs/slurm.log
#SBATCH --error=zlogs/slurm.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -p rtx8000,v100
#SBATCH --mem=90000M
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL

source $HOME/.bashrc

mark=air1h_on_v037_1h_loadr255_1m_lr0.000001_sac1_b10_epinf_eval2_r811
lr=0.000001

config_file=configs.student_tf_1h_ongpt_air_gt
trained_teacher_model=outputs/d58_990k_t2k_bs32_r422/0/model_best.ckpt
#loadmodel_for_student=outputs/debugt1_prefix8_clip_gpt_generated_data_5w_nll_lr3en4_sac1_d143/model_best.ckpt
#loadmodel_for_student=outputs/airv034_100w_prefix4_gt_5w_lr0p000003_r765/model_best.ckpt
loadmodel_for_student=outputs/dv034_gpt_generated_r200_1M_wogptwarm_student_bs250_eval4_r255/model_best.ckpt

output_path=outputs/$mark
log=logs/log$mark
err=logs/err$mark

mkdir -p $output_path

pythont=python

#CUDA_VISIBLE_DEVICES=${gpuid} \
$pythont gpt2_train.py \
  --do-train \
  --do-eval \
  --pretrained-model-name=gpt2-small \
  --trained_teacher_model=${trained_teacher_model} \
  --checkpoint ${loadmodel_for_student} \
  --lr ${lr} \
  --config-train=$config_file \
  --output-dir=$output_path  \
  > $log 2> $err
#  --checkpoint ${loadmodel_for_student} \
