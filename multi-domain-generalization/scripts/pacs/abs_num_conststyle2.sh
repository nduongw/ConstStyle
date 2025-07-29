#!/bin/bash
DATA=./DATA
DATASET=pacs
D1=cartoon
D2=art_painting
D3=photo
D4=sketch
SEED=42
method=conststyle

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--uncertainty 0.5 \
--trainer ConstStyleTrainer \
--source-domains ${D1} ${D3} ${D4} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--cluster ot \
--num_clusters 3 \
--num_conststyle 3 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--uncertainty 0.5 \
--trainer ConstStyleTrainer \
--source-domains ${D1} ${D3} ${D4} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--cluster ot \
--num_clusters 3 \
--num_conststyle 4 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--resume false)
