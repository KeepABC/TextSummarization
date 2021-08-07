log_dir="./logs"

data_dir="../../data/concat_longtitle_content"
model_name_or_path="../../pretrained_models/uer-pegasus-base-chinese-cluecorpussmall/"
max_target_length=32
max_length=1024
max_train_steps=300
save_steps=10
learning_rate=5e-5
num_beams=3
batch_size=4
gradient_accumulation_steps=4

log_file="${log_dir}/v3_${ts}_max_target_length_${max_target_length}_max_length_${max_length}_max_train_steps_${max_train_steps}_learning_rate_${learning_rate}_num_beams_${num_beams}.log"

echo "log_file: $log_file"

CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 ../../code/summary.py \
        --train_file ${data_dir}/dev.json \
        --validation_file ${data_dir}/dev.json \
        --text_column title_content --summary_column push_title \
        --model_name_or_path $model_name_or_path \
        --max_target_length $max_target_length --max_length $max_length \
        --max_train_steps $max_train_steps --save_steps $save_steps \
        --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --ignore_pad_token_for_loss true \
        --learning_rate $learning_rate \
        --num_beams $num_beams \
        --local_rank 0 \
        --output_dir ../../savaed_models/v3 \
        --model_type pegasus > $log_file 2>&1 &

tail -f $log_file