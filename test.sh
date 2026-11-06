#!/usr/bin/env bash
# Main non-IID experiments for the DPA paper.
# Based on https://github.com/aparna-aketi/global_update_tracking
#
# Run all experiments for one seed with SEED=0 bash test.sh.
# Repeat with SEED=1 and SEED=2. Run from the repository root.
# For a smaller selection, keep the setup and your chosen experiment blocks.

set -euo pipefail

SEED="${SEED:-0}"
DEVICES="${DEVICES:-4}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Run a method and save its console output.
run() {
    local log_name="$1"
    shift
    python -u trainer.py "${experiment_args[@]}" \
        --devices "$DEVICES" --seed "$SEED" \
        --data-dir "$DATA_DIR" --save-dir "$OUTPUT_DIR" \
        "$@" 2>&1 | tee "$LOG_DIR/${log_name}_${experiment}_seed${SEED}.log"
}

# Shared coefficient schedules for DPA and DPA-1G.
dpa_schedule=(
    --a-schedule exp --a-start 0.005 --a-end 0.0001 --qa 1
    --b-schedule exp --b-start 0.3 --b-end 0.1 --qb 1
    --a-epoch-frac 1.0
)

# 1. CIFAR-10 / ResNet-20 / 16-agent ring / alpha=0.1
experiment=cifar10_resnet20_ring_n16_alpha0p1
experiment_args=(
    --dataset cifar10 --classes 10 --arch resnet --depth 20 --normtype evonorm
    --world_size 16 --batch-size 512 --graph ring --neighbors 2
    --epochs 200 --skew 0.1
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.90 --beta-2 0.80
    --optimal-epoch 200 --b-epoch-frac 0.15 --warmup-epoch-frac 0.25
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.1 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.1 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.06

# 2. CIFAR-10 / ResNet-20 / 16-agent ring / alpha=0.01
experiment=cifar10_resnet20_ring_n16_alpha0p01
experiment_args=(
    --dataset cifar10 --classes 10 --arch resnet --depth 20 --normtype evonorm
    --world_size 16 --batch-size 512 --graph ring --neighbors 2
    --epochs 200 --skew 0.01
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.90 --beta-2 0.80
    --optimal-epoch 200 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.1 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.1 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.06

# 3. CIFAR-10 / VGG-11 / 16-agent ring / alpha=0.1
experiment=cifar10_vgg11_ring_n16_alpha0p1
experiment_args=(
    --dataset cifar10 --classes 10 --arch vgg11 --normtype evonorm
    --world_size 16 --batch-size 512 --graph ring --neighbors 2
    --epochs 200 --skew 0.1
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.95 --beta-2 0.85
    --optimal-epoch 200 --b-epoch-frac 0.15 --warmup-epoch-frac 0.25
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.1 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.1 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.08

# 4. CIFAR-10 / VGG-11 / 16-agent ring / alpha=0.01
experiment=cifar10_vgg11_ring_n16_alpha0p01
experiment_args=(
    --dataset cifar10 --classes 10 --arch vgg11 --normtype evonorm
    --world_size 16 --batch-size 512 --graph ring --neighbors 2
    --epochs 200 --skew 0.01
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.95 --beta-2 0.85
    --optimal-epoch 200 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.1 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.05 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.09

# 5. CIFAR-100 / ResNet-20 / 16-agent ring / alpha=0.1
experiment=cifar100_resnet20_ring_n16_alpha0p1
experiment_args=(
    --dataset cifar100 --classes 100 --arch resnet --depth 20 --normtype evonorm
    --world_size 16 --batch-size 512 --graph ring --neighbors 2
    --epochs 200 --skew 0.1
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.90 --beta-2 0.80
    --optimal-epoch 200 --b-epoch-frac 0.15 --warmup-epoch-frac 0.25
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.1 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.1 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.005

# 6. CIFAR-100 / ResNet-20 / 16-agent ring / alpha=0.01
experiment=cifar100_resnet20_ring_n16_alpha0p01
experiment_args=(
    --dataset cifar100 --classes 100 --arch resnet --depth 20 --normtype evonorm
    --world_size 16 --batch-size 512 --graph ring --neighbors 2
    --epochs 200 --skew 0.01
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.90 --beta-2 0.80
    --optimal-epoch 200 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.1 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.05 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.005

# 7. Fashion-MNIST / LeNet-5 / 16-agent ring / alpha=0.1
experiment=fmnist_lenet5_ring_n16_alpha0p1
experiment_args=(
    --dataset fmnist --classes 10 --arch lenet5 --normtype evonorm
    --world_size 16 --batch-size 512 --graph ring --neighbors 2
    --epochs 100 --skew 0.1
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 0.1 --beta-1 0.95 --beta-2 0.85
    --optimal-epoch 100 --b-epoch-frac 0.15 --warmup-epoch-frac 0.25
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.01 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.05 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.01

# 8. Fashion-MNIST / LeNet-5 / 16-agent ring / alpha=0.01
experiment=fmnist_lenet5_ring_n16_alpha0p01
experiment_args=(
    --dataset fmnist --classes 10 --arch lenet5 --normtype evonorm
    --world_size 16 --batch-size 512 --graph ring --neighbors 2
    --epochs 100 --skew 0.01
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 0.1 --beta-1 0.95 --beta-2 0.85
    --optimal-epoch 100 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.01 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.05 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.005

# 9. CIFAR-10 / ResNet-20 / 32-agent ring / alpha=0.1
experiment=cifar10_resnet20_ring_n32_alpha0p1
experiment_args=(
    --dataset cifar10 --classes 10 --arch resnet --depth 20 --normtype evonorm
    --world_size 32 --batch-size 1024 --graph ring --neighbors 2
    --epochs 200 --skew 0.1
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.90 --beta-2 0.80
    --optimal-epoch 200 --b-epoch-frac 0.15 --warmup-epoch-frac 0.25
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.05 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.05 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.04

# 10. CIFAR-10 / ResNet-20 / 32-agent ring / alpha=0.01
experiment=cifar10_resnet20_ring_n32_alpha0p01
experiment_args=(
    --dataset cifar10 --classes 10 --arch resnet --depth 20 --normtype evonorm
    --world_size 32 --batch-size 1024 --graph ring --neighbors 2
    --epochs 200 --skew 0.01
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.90 --beta-2 0.80
    --optimal-epoch 200 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run gt --method GT --lr 0.05 --gt-beta 0.0 --gt-wd 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run mt --method MT --lr 0.05 --mt-beta 0.9 --mt-wd 0.0 --mt-paper-init
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.04

# 11. CIFAR-10 / VGG-11 / 32-agent ring / alpha=0.1
experiment=cifar10_vgg11_ring_n32_alpha0p1
experiment_args=(
    --dataset cifar10 --classes 10 --arch vgg11 --normtype evonorm
    --world_size 32 --batch-size 1024 --graph ring --neighbors 2
    --epochs 200 --skew 0.1
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.95 --beta-2 0.85
    --optimal-epoch 200 --b-epoch-frac 0.15 --warmup-epoch-frac 0.25
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.08

# 12. CIFAR-10 / VGG-11 / 32-agent ring / alpha=0.01
experiment=cifar10_vgg11_ring_n32_alpha0p01
experiment_args=(
    --dataset cifar10 --classes 10 --arch vgg11 --normtype evonorm
    --world_size 32 --batch-size 1024 --graph ring --neighbors 2
    --epochs 200 --skew 0.01
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.95 --beta-2 0.85
    --optimal-epoch 200 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
# Additional DSGD example
run dsgd --method gut --lr 0.1 --momentum 0.0 --scaling 0.0
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.08

# 13. CIFAR-10 / ResNet-20 / 32-agent torus / alpha=0.01
experiment=cifar10_resnet20_torus_n32_alpha0p01
experiment_args=(
    --dataset cifar10 --classes 10 --arch resnet --depth 20 --normtype evonorm
    --world_size 32 --batch-size 1024 --graph torus --neighbors 4
    --epochs 200 --skew 0.01
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.90 --beta-2 0.70
    --optimal-epoch 200 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.05

# 14. CIFAR-10 / ResNet-20 / 32-agent dyck / alpha=0.01
experiment=cifar10_resnet20_dyck_n32_alpha0p01
experiment_args=(
    --dataset cifar10 --classes 10 --arch resnet --depth 20 --normtype evonorm
    --world_size 32 --batch-size 1024 --graph dyck --neighbors 3
    --epochs 200 --skew 0.01
)
dpa_args=(
    "${dpa_schedule[@]}" --lr 5.0 --beta-1 0.95 --beta-2 0.65
    --optimal-epoch 200 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50
)

run dpa --method DPA --t-gossip-x 0 "${dpa_args[@]}"
run dpa-1g --method DPA-1G --t-gossip-x 5 "${dpa_args[@]}"
run qg-dsgdm --method gut --lr 0.1 --momentum 0.9 --scaling 0.0
run qg-gutm --method gut --lr 0.1 --momentum 0.9 --scaling 0.05
