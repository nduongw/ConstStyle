#!/bin/bash
DATA=./DATA
DATASET=pacs
D1=art_painting
D2=cartoon
D3=photo
D4=sketch
SEED=42
method=baseline

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} ${D4} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify2/${D2}_${D3}_${D4} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify2/${D2}_${D3} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D4} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify2/${D2}_${D4} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D3} ${D4} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify2/${D3}_${D4} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify2/${D2} \
# --resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D3} \
--target-domains ${D1} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify2/${D3} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D4} \
--target-domains ${D1} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify2/${D4} \
--resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
