data_path=$1
mark=$2
model_size=gpt2-medium
#config_train=configs.config_train_news_h45M
#config_train=configs.config_teacher_train_dv035_39_997k
config_train=configs.config_teacher_train_dv040_999k

max_seq_length=40

mkdir -p logs/prepare_data

$pythont prepare_data.py --data-dir $data_path \
    --max-seq-length=${max_seq_length} \
    --output-dir=$data_path \
    --config-train=$config_train \
    --pretrained-model-name=$model_size \
    > logs/prepare_data/logd$mark 2> logs/prepare_data/errd$mark
