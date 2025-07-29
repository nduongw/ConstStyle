#!/bin/bash
DATA=./DATA
DATASET=pacs
D1=cartoon
D2=sketch
D3=photo
D4=art_painting
SEED=42
method=baseline

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D3} ${D4} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D3}-${D4}_${D2} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D4} ${D1} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D4}-${D1}_${D2} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D4} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D4}_${D2} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D3} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D3}_${D2} \
--resume false)


(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} ${D3} ${D4} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1}-${D3}-${D4}_${D2} \
--resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
