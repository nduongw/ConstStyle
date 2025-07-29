#!/bin/bash
DATA=./DATA
DATASET=digit5
D1=mnist
D2=mnist_m
D3=svhn
D4=syn
D5=usps
SEED=13
method=baseline

(CUDA_VISIBLE_DEVICES=1 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D2} ${D3} \
--target-domains ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verifyparadox/${D2}_${D3}-${D1} \
--resume false)

(CUDA_VISIBLE_DEVICES=1 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D3} \
--target-domains ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verifyparadox/${D2}_${D3}-${D1} \
--resume false)


echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
