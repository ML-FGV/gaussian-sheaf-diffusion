#!/bin/sh

python -m exp.run \
    --dataset=erdos_renyi \
    --d=3 \
    --layers=1 \
    --hidden_channels=5 \
    --rest_maps_mlp_layers=1 \
    --rest_maps_mlp_hc=10 \
    --final_mlp_layers=1 \
    --final_mlp_hc=15 \
    --early_stopping=100 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.2 \
    --lr_decay_patience=20 \
    --weight_decay=5e-3 \
    --input_dropout=0.0 \
    --dropout=0.7 \
    --use_act=False \
    --model=$1 \
    --normalised=True \
    --sparse_learner=True \
    --entity="andrerg00" \
    --rest_maps_type='diag'