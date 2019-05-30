work_path=$(dirname $0)
    python train_lstm.py  \
        --lr 1e-4 \
        --style_weight 5e5 \
        --content_weight 1 \
        --long_weight 100 \
        --short_weight 100 \
        --dataset VideoNet \
        --epoch 100 \
        --save_dir $work_path/checkpoint/ \
        2>&1 | tee $work_path/train.log
# CUDA_VISIBLE_DEVICES=$1 \
        # --parallel \
