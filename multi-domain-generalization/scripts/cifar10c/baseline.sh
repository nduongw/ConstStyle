#!/bin/bash
DATA=./DATA
DATASET=cifar10_c
D1=cifar10
D2=cifar10_c
SEED=40
method=baseline

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_1.yaml \
--config-file configs/trainers/dg/vanilla/cifar10.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_2.yaml \
--config-file configs/trainers/dg/vanilla/cifar10.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_3.yaml \
--config-file configs/trainers/dg/vanilla/cifar10.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_4.yaml \
--config-file configs/trainers/dg/vanilla/cifar10.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_5.yaml \
--config-file configs/trainers/dg/vanilla/cifar10.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
