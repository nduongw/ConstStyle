#!/bin/bash
DATA=./DATA
DATASET=digit5
D1=usps
D2=mnist_m
D3=svhn
D4=syn
D5=mnist
SEED=13
method=conststyle

(CUDA_VISIBLE_DEVICES=1 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D4} \
--target-domains ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify3/${D3}_${D4}-${D1} \
--cluster ot \
--num_clusters 4 \
--update_interval 25 \
--conststyle_type ver5 \
--prob 0.5 \
--alpha 0.5 \
--wandb 0 \
--resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} ${D5} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}_${D3}_${D5}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D4} ${D5} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}_${D4}_${D5}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D3} ${D4} ${D5} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D3}_${D4}_${D5}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}_${D3}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D4} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}_${D4}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D5} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}_${D5}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D3} ${D4} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D3}_${D4}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D3} ${D5} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D3}_${D5}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D4} ${D5} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D4}_${D5}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D2} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D3} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D3}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D4} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D4}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer ConstStyleTrainer \
# --uncertainty 0.5 \
# --source-domains ${D5} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D5}-${D1} \
# --cluster ot \
# --num_clusters 4 \
# --update_interval 25 \
# --conststyle_type ver5 \
# --prob 0.5 \
# --alpha 0.5 \
# --resume false)


echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
