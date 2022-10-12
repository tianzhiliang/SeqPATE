data_path=$1
mark=$2
model_size=gpt2-medium
config_train=configs.teacher_dv039_997k
is_train_only=0

max_seq_length=40

mkdir -p logs/prepare_data

pythont=python
$pythont prepare_data.py --data-dir $data_path \
    --max-seq-length=${max_seq_length} \
    --output-dir=$data_path \
    --config-train=$config_train \
    --is_train_only=$is_train_only \
    --pretrained-model-name=$model_size \
    > logs/prepare_data/logd$mark 2> logs/prepare_data/errd$mark
