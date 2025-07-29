#!/bin/bash
DATA=./DATA
DATASET=pacs
D1=art_painting
D2=sketch
D3=cartoon
D4=photo
SEED=1
method=conststyle

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} ${D3} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify22/${D1}_${D3} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} ${D4} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify22/${D1}_${D4} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D3} ${D4} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify22/${D3}_${D4} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify22/${D1} \
# --cluster ot \
# --num_clusters 1 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D3} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify22/${D3} \
# --cluster ot \
# --num_clusters 1 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D4} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify22/${D4} \
# --cluster ot \
# --num_clusters 1 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} ${D2} \
# --target-domains ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify23/${D1}_${D2} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} ${D4} \
# --target-domains ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify23/${D1}_${D4} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D4} \
# --target-domains ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify23/${D2}_${D4} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} \
# --target-domains ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify23/${D2} \
# --cluster ot \
# --num_clusters 1 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} \
# --target-domains ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify23/${D1} \
# --cluster ot \
# --num_clusters 1 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)


# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D4} \
# --target-domains ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify23/${D4} \
# --cluster ot \
# --num_clusters 1 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} \
# --target-domains ${D4} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify24/${D2}_${D3} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --wandb 0 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} ${D3} \
# --target-domains ${D4} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify24/${D1}_${D3} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --wandb 0 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} ${D2} \
# --target-domains ${D4} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify24/${D1}_${D2} \
# --cluster ot \
# --num_clusters 2 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --wandb 0 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D3} \
# --target-domains ${D4} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify24/${D3} \
# --cluster ot \
# --num_clusters 1 \
# --update_interval 25 \
# --prob 0.5 \
# --alpha 0.5 \
# --conststyle_type ver5 \
# --wandb 0 \
# --resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify24/${D1} \
--cluster ot \
--num_clusters 1 \
--update_interval 25 \
--prob 0.5 \
--alpha 0.5 \
--conststyle_type ver5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D4} \
--target-domains ${D1} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify24/${D4} \
--cluster ot \
--num_clusters 1 \
--update_interval 25 \
--prob 0.5 \
--alpha 0.5 \
--conststyle_type ver5 \
--resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
