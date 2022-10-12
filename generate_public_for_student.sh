#!/bin/bash
#SBATCH --job-name=gpub4stu
#SBATCH --output=zlogs/slurm.log
#SBATCH --error=zlogs/slurm.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -p rtx8000
#SBATCH --mem=200000M
#SBATCH --time=5:00:00
#SBATCH --mail-type=ALL

source $HOME/.bashrc

mark=eu
pred_mark=gpt4pub
config_file=configs.gpt_generate_for_student_public_euro6

prompt_token_num=-1
results_path=results/${mark}_${pred_mark}/
log=loggs/logg${mark}_${pred_mark}
err=loggs/errg${mark}_${pred_mark}

mkdir -p $results_path

pythont=python

function generate() {
#CUDA_VISIBLE_DEVICES=${gpuid} \
$pythont test.py \
  --do-test \
  --config-train=$config_file \
  --temperature=0.7 \
  --top-k=40 \
  --output-dir=$results_path \
  --pretrained-model-name=gpt2-small  \
  --prompt_token_num=$prompt_token_num \
  > $log 2> $err
}

function evaluate() {
gt=${results_path}/test_samples.txt.gt
hyp=${results_path}/test_samples.txt.hyp
base=${results_path}/test_samples.txt
bleu=${base}.bleu
dist=${base}.dist
$pythont scripts/eval/bleu.py $gt $hyp 0 > $bleu
$pythont scripts/eval/distinct.py < $hyp > $dist
}

function evaluate_with_posteval2() {
gt=${results_path}/test_samples.txt.gt
hyp=${results_path}/test_samples.txt.hyp
new_gt=${results_path}/test_samples.txt.new.gt
new_hyp=${results_path}/test_samples.txt.new.hyp
base=${results_path}/test_samples.txt
bleu=${base}.posteval2.bleu
$pythont scripts/eval/posteval2_split_word_punctuation.py < $gt > $new_gt &
$pythont scripts/eval/posteval2_split_word_punctuation.py < $hyp > $new_hyp &
wait
$pythont scripts/eval/bleu.py $new_gt $new_hyp 0 > $bleu
}

generate
evaluate
#evaluate_with_posteval2
