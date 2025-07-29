#train with test data
CUDA_VISIBLE_DEVICES=0 python microast.py --save_model_interval 1000 --save_dir ./experiments_micro_photo --source-domains photo --target-domains cartoon art_painting sketch