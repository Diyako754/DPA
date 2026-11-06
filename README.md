# DPA: Decentralized Primal Averaging

PyTorch implementation of **DPA** and **DPA-1G** for decentralized learning with heterogeneous data.

**DPA: Decentralized Primal Averaging with Quasi-Global Momentum for Highly Heterogeneous Data**  
Diyako Ghaderyan and Stefan Werner, NeurIPS 2026.  
[Paper](https://openreview.net/forum?id=8Tew0zEsuF)

DPA combines primal averaging with local quasi-global momentum. Both variants use a constant learning rate after linear warmup.

- **DPA** uses two communication rounds per iteration.
- **DPA-1G** uses one communication round per iteration. In our experiments, four iterations perform gradient updates, and every fifth iteration averages the output models without computing a gradient.

## Installation

The dependencies are listed in [env.yml](env.yml).

```bash
git clone https://github.com/Diyako754/DPA.git
cd DPA
conda env create -n DPA -f env.yml
conda activate DPA
```

Training requires CUDA-enabled PyTorch. Prepare the dataset in `./data`, or specify its location with `--data-dir`.

## How to run

This example trains ResNet-20 on CIFAR-10 with 16 agents on a ring, `alpha = 0.1`, and four GPUs. The mini-batch size is 32 per agent. Run it from the repository root.

```bash
mkdir -p logs

python -u trainer.py \
  --method DPA \
  --dataset cifar10 --classes 10 \
  --arch resnet --depth 20 --normtype evonorm \
  --world_size 16 --graph ring --neighbors 2 --devices 4 \
  --batch-size 512 --epochs 200 --skew 0.1 --seed 0 \
  --lr 5.0 --beta-1 0.90 --beta-2 0.80 --t-gossip-x 0 \
  --optimal-epoch 200 \
  --a-epoch-frac 1.0 --b-epoch-frac 0.15 --warmup-epoch-frac 0.25 \
  --a-schedule exp --a-start 0.005 --a-end 0.0001 --qa 1 \
  --b-schedule exp --b-start 0.3 --b-end 0.1 --qb 1 \
  --data-dir ./data --save-dir ./outputs/dpa_seed0 \
  > logs/dpa_seed0.log 2>&1
```

To run **DPA-1G**, change the method and communication setting to `--method DPA-1G --t-gossip-x 5`. Use a different log filename and output directory for each run.

Commands for the other main non-IID experiments and the DSGD, GT, QG-DSGDm, MT, and QG-GUTm baselines are in [test.sh](test.sh).

Run `SEED=0 bash test.sh` to execute all listed experiments for one seed. Use `SEED=1` and `SEED=2` for the other two seeds.

## Experiment settings

We use the following DPA and DPA-1G settings for the main ring experiments.

| Dataset / model | Epochs | Learning rate | `(beta_1, beta_2)` |
| --- | ---: | ---: | --- |
| Fashion-MNIST / LeNet-5 | 100 | 0.1 | `(0.95, 0.85)` |
| CIFAR-10 / VGG-11 | 200 | 5.0 | `(0.95, 0.85)` |
| CIFAR-10 / ResNet-20 | 200 | 5.0 | `(0.90, 0.80)` |
| CIFAR-100 / ResNet-20 | 200 | 5.0 | `(0.90, 0.80)` |

Set `--optimal-epoch` equal to `--epochs` for these experiments. The coefficient schedules and warmup settings are given in the commands.

The `--skew` argument sets the Dirichlet concentration parameter. We use `0.1` and `0.01`, with smaller values giving more heterogeneous data. For `alpha = 0.01`, replace the corresponding arguments in the example with `--skew 0.01 --b-epoch-frac 0.30 --warmup-epoch-frac 0.50`.

## Evaluation

DPA and DPA-1G evaluate the averaged output model `x`. **LE** is the test accuracy at the last training epoch. **L5E** is the average over the last five training epochs.

The final summary reports LE, L5E, and post-training accuracy, averaged across agents. For the baselines, the post-training evaluation follows additional model averaging. For DPA and DPA-1G, it repeats evaluation at `x`.

For each seed, first average accuracy across agents. We report the mean and standard deviation across seeds `0`, `1`, and `2`, using `ddof=0` for the standard deviation.

## Acknowledgments and citation

This repository builds on Global Update Tracking (GUT) by Sai Aparna Aketi, Abolfazl Hashemi, and Kaushik Roy. We thank the authors for sharing their code and Sai Aparna Aketi for permission to publish this modified fork. [GUT repository](https://github.com/aparna-aketi/global_update_tracking)

Please cite our paper and the GUT paper when using this implementation.

```bibtex
@inproceedings{ghaderyan2026dpa,
  title     = {DPA: Decentralized Primal Averaging with Quasi-Global Momentum for Highly Heterogeneous Data},
  author    = {Ghaderyan, Diyako and Werner, Stefan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=8Tew0zEsuF}
}

@inproceedings{aketi2023gut,
  title     = {Global Update Tracking: A Decentralized Learning Algorithm for Heterogeneous Data},
  author    = {Aketi, Sai Aparna and Hashemi, Abolfazl and Roy, Kaushik},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.04792}
}
```
