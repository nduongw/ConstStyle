#train with test data
CUDA_VISIBLE_DEVICES=0 python nst.py --save_model_interval 1000 --save_dir ./experiments3 --source-domains photo --target-domains cartoon art_painting sketch