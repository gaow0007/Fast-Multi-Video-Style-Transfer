export CUDA_VISIBLE_DEVICES=1
python train.py  \
        --lr 1e-3 \
        --batch_size 4 \
        --style_weight 5e4 \
        --content_weight 1 \
        --save_dir experiments/ \
        2>&1 | tee ./train_stagemimic.log
