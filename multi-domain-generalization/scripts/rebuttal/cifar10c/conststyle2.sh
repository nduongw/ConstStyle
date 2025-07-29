#!/bin/bash
DATA=./DATA
DATASET=cifar10_c
D1=cifar10
D2=cifar10_c
SEED=40
method=conststyle

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_1_cs.yaml \
# --config-file configs/trainers/dg/vanilla/cifar10.yaml \
# --output-dir output/dg/${DATASET}/${method}/${D2} \
# --cluster ot \
# --num_clusters 1 \
# --num_conststyle 3 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_2_cs.yaml \
# --config-file configs/trainers/dg/vanilla/cifar10.yaml \
# --output-dir output/dg/${DATASET}/${method}/${D2} \
# --cluster ot \
# --num_clusters 1 \
# --num_conststyle 3 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D1} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_3_cs.yaml \
# --config-file configs/trainers/dg/vanilla/cifar10.yaml \
# --output-dir output/dg/${DATASET}/${method}/${D2} \
# --cluster ot \
# --num_clusters 1 \
# --num_conststyle 3 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_4_cs.yaml \
--config-file configs/trainers/dg/vanilla/cifar10.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--cluster ot \
--num_clusters 1 \
--num_conststyle 3 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_5_cs.yaml \
--config-file configs/trainers/dg/vanilla/cifar10.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--cluster ot \
--num_clusters 1 \
--num_conststyle 3 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
