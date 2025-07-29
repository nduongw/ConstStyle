#!/bin/bash
DATA=./DATA
DATASET=digit5
D1=mnist
D2=mnist_m
D3=svhn
D4=syn
D5=usps
SEED=13
method=conststyle

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D2} ${D3} ${D4} ${D5} \
--target-domains ${D1} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1} \
--cluster ot \
--test_split val \
--num_clusters 4 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--wandb 0 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D1} ${D3} ${D4} ${D5} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--cluster ot \
--test_split val \
--num_clusters 4 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D1} ${D2} ${D4} ${D5} \
--target-domains ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D3} \
--cluster ot \
--test_split val \
--num_clusters 4 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D1} ${D2} ${D3} ${D5} \
--target-domains ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D4} \
--cluster ot \
--test_split val \
--num_clusters 4 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--resume false) 

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D1} ${D2} ${D3} ${D4} \
--target-domains ${D5} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D5} \
--cluster ot \
--test_split val \
--num_clusters 4 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--resume false) 

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
