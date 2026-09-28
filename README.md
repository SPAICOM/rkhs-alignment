# Nonlinear Semantic Alignment via Reproducing Kernel Hilbert Spaces


<h5 align="center">
    
[![ieee](https://img.shields.io/static/v1?label=IEEE+Paper&message=ID-HERE&color=0057b7&logo=ieee)](https://ieeexplore.ieee.org/document/ID-HERE)
[![arXiv](https://img.shields.io/badge/Arxiv-ID.HERE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/CODE.HERE)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/REPO-NAME-HERE/blob/main/LICENSE)

 <br>

</h5>

> [!TIP]
> Semantic interoperability among independently trained AI-native communication devices requires aligning heterogeneous latent spaces without retraining the underlying models. Existing alignment methods typically rely on linear transformations, which may be insufficient to capture nonlinear relations between independently learned semantic representations. In this paper, we propose Residual Kernel Alignment (RKA), a novel semantic alignment method that combines a geometry-preserving Stiefel transformation with a residual in a reproducing kernel Hilbert space (RKHS) to capture nonlinear latent space mismatch. An orthogonality constraint separates the two components and, under the Stiefel isometry condition, exactly decouples their estimation. The linear component is obtained through standard Procrustes alignment, while the nonlinear residual admits a closed-form constrained kernel ridge-regression solution. The proposed alignment strategy is learned from paired latent representations, referred to as semantic pilots. We therefore also address the design of the pilot set and develop a kernel-herding selection strategy to identify informative calibration samples. Numerical results show that RKA outperforms purely linear alignment and that optimized pilot selection provides
substantial gains in the low-pilot regime.


## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management and [`just`](https://github.com/casey/just) as the task runner.

### Install prerequisites

Install the required tools:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [`just`](https://github.com/casey/just)

Follow the installation instructions from their official documentation.

### Setup the development environment

From the project root, run:

```bash
just setup
```

The `setup` recipe will:

- Create the `.venv` virtual environment (if it does not exist)
- Install all project dependencies using `uv`

After the command completes, the development environment will be ready to use. 🚀

## Reproducing the figures

Three studies, one script and one config each, plus an average of the
last two over encoder pairs. All are driven by Hydra: no experiment
parameter lives in the `justfile`, so a run is fully described by
`config/hydra/` plus the overrides on its command line. The defaults in
those configs are *not* the paper's setting — the commands below state
every override the figures were produced with.

The setting: SEMASIA CIFAR-10, `regnety_016.pycls_in1k` (888-d)
transmitting into `vit_large_patch16_224.augreg_in21k_ft_in1k` (1024-d),
a truncated-whitening chart shared by every method, kernel-herded pilots
and a linear probe on the receiver.

### 1. Lambda sweep — every (rank, budget) cell the other two figures read

**`scripts/lambda_sweep.py`** fits RKA over its regularisation grid and
Procrustes once, per cell. Figures (ii) and (iii) do not refit either
method: they read these CSVs back, matched on their columns, so every
cell they need has to exist first. Figure (ii) needs rank 32 at every
pilot budget; figure (iii) needs every rank at 8192 pilots.

```bash
# k = 32 at every budget (figure ii), including N = 8192 (figure iii)
just lambda-sweep 'data.models=[regnety_016.pycls_in1k,vit_large_patch16_224.augreg_in21k_ft_in1k]' 'charts=[{preprocess:whiten}]' 'ranks=[32]' 'pilots.counts=[128,256,512,1024,2048,4096,8192]' decoder.kind=linear

# the other ranks at N = 8192 (figure iii)
just lambda-sweep 'data.models=[regnety_016.pycls_in1k,vit_large_patch16_224.augreg_in21k_ft_in1k]' 'charts=[{preprocess:whiten}]' 'ranks=[16,64,128,256,512]' 'pilots.counts=[8192]' decoder.kind=linear
```

Kernel herding is greedy and seeded from the config, so a cell's pilots
do not depend on which other budgets share its run: splitting the grid
across commands, or running it as one, writes the same CSVs.

**Figure (i)** — RKA against Procrustes over `lambda`, with Procrustes as
its `lambda -> infinity` limit:

```
figures/lambda_sweep/cifar10/whiten-k32/lambda_cifar10_regnety_016-to-vit_large_p16_whiten-k32_k32_n8192_lam1e-10to100x25_{accuracy,mrr}.{pdf,png}
figures/lambda_sweep/cifar10/whiten-k128/lambda_cifar10_regnety_016-to-vit_large_p16_whiten-k128_k128_n8192_lam1e-10to100x25_{accuracy,mrr}.{pdf,png}
```

### 2. Pilot sweep — figure (ii)

**`scripts/pilot_sweep.py`**: RKA, Procrustes, Direct MLP and Residual MLP
against the pilot budget at 32 symbols, for kernel herding and
class-stratified random pilots (`round_robin`), mean ± sd over five seeds.
RKA's `lambda` is read back per budget from step 1; the two MLPs are
trained on the same 32-d truncated-whitened pilots.

```bash
just pilot-sweep 'data.models=[regnety_016.pycls_in1k,vit_large_patch16_224.augreg_in21k_ft_in1k]' 'charts=[{preprocess:whiten}]' 'ranks=[32]' 'pilots.counts=[128,256,512,1024,2048,4096,8192]' 'pilots.strategies=[herding,round_robin]' decoder.kind=linear
```

`ranks=[32]` rather than `symbols=32`: the lambda lookup matches the chart
tag `whiten-k32` that step 1 wrote.

```
figures/pilot_sweep/cifar10/whiten-k32/pilots_cifar10_regnety_016-to-vit_large_p16_whiten-k32_k32_N128to8192x7_herding-round_robin.{pdf,png}
```

### 3. Dimension sweep — figure (iii)

**`scripts/dimension_sweep.py`**: the field against the number of
transmitted symbols at 8192 pilots. RKA and Procrustes are read back from
step 1; CCA, SVCCA and Proto-PFE are fitted on the same herded pilots,
with their rate set by the canonical rank or the anchor count rather than
by a truncation. Before drawing, the
run refits Procrustes and stops if it disagrees with the CSVs.

```bash
just dimension-sweep 'data.models=[regnety_016.pycls_in1k,vit_large_patch16_224.augreg_in21k_ft_in1k]' 'charts=[{preprocess:whiten}]' 'ranks=[16,32,64,128,256,512]' pilots.n_pilots=8192 decoder.kind=linear
```

```
figures/dimension_sweep/cifar10/whiten/dims_cifar10_regnety_016-to-vit_large_p16_whiten_n8192_herding_k16to512x6_dec-linear.{pdf,png}
```

### 4. Figures (ii) and (iii) averaged over encoder pairs

**`scripts/pair_average.py`** fits nothing: it pools steps 2 and 3 for
every pair in `config/hydra/pair_average.yaml` and redraws both figures
with the band over pairs instead of seeds (seeds are averaged within a
pair first). The five pairs are CNN transmitters into differently
pre-trained ViT receivers, no encoder repeated:

| transmitter | receiver |
|---|---|
| `regnety_016.pycls_in1k` | `vit_large_patch16_224.augreg_in21k_ft_in1k` |
| `repvgg_b0.rvgg_in1k` | `aimv2_large_patch14_224.apple_pt` |
| `mobilenetv3_large_100.ra_in1k` | `vit_base_patch16_224.augreg_in21k` |
| `efficientvit_b0.r224_in1k` | `beit_base_patch16_224.in22k_ft_in22k` |
| `ghostnet_100.in1k` | `eva02_base_patch14_224.mim_in22k` |

Run steps 1–3 for each pair with the receiver pinned and one pilot-sweep
seed, e.g. for the second pair:

```bash
just lambda-sweep 'data.models=[repvgg_b0.rvgg_in1k,aimv2_large_patch14_224.apple_pt]' receiver=aimv2_large_patch14_224.apple_pt 'charts=[{preprocess:whiten}]' 'ranks=[32]' 'pilots.counts=[128,256,512,1024,2048,4096,8192]' decoder.kind=linear
just lambda-sweep 'data.models=[repvgg_b0.rvgg_in1k,aimv2_large_patch14_224.apple_pt]' receiver=aimv2_large_patch14_224.apple_pt 'charts=[{preprocess:whiten}]' 'ranks=[16,64,128,256,512]' 'pilots.counts=[8192]' decoder.kind=linear
just pilot-sweep 'data.models=[repvgg_b0.rvgg_in1k,aimv2_large_patch14_224.apple_pt]' receiver=aimv2_large_patch14_224.apple_pt 'charts=[{preprocess:whiten}]' 'ranks=[32]' 'pilots.counts=[128,256,512,1024,2048,4096,8192]' 'pilots.strategies=[herding,round_robin]' 'seeds=[0]' decoder.kind=linear
just dimension-sweep 'data.models=[repvgg_b0.rvgg_in1k,aimv2_large_patch14_224.apple_pt]' receiver=aimv2_large_patch14_224.apple_pt 'charts=[{preprocess:whiten}]' 'ranks=[16,32,64,128,256,512]' pilots.n_pilots=8192 decoder.kind=linear
```

then pool them:

```bash
just pair-average
```

You do not have to type the other pairs out: for any pair not yet on
disk, `just pair-average` stops and prints its four commands, already
filled in. A pair takes roughly an hour, most of it the pilot sweep's MLP
baselines.

```
figures/pair_average/cifar10/whiten/pairavg-pilots_cifar10_5pairs_whiten-k32_N128to8192x7_herding-round_robin.{pdf,png}
figures/pair_average/cifar10/whiten/pairavg-dims_cifar10_5pairs_whiten_n8192_herding_k16to512x6.{pdf,png}
```

### Where the output lands

Figures and the data behind them live in parallel trees, and every
filename carries the whole configuration that produced it — a figure
loses its path the moment it is copied into a paper.

```
figures/<study>/<dataset>/<chart>/<stem>[_<metric>].{pdf,png}
results/<study>/<dataset>/<stem>.csv
```

### Re-running

Every script writes each unit of work — a `(chart, budget)` cell, a
seed, a fitted method at one rank — as it finishes and skips what is
already on disk, so an interrupted run resumes rather than restarts, and
`just` retries automatically. Append to any command above:

```bash
resume=false       # refit everything
plot_only=true     # redraw from the CSVs, fit nothing
```

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
```

## Authors

- [Enrico Grimaldi](https://scholar.google.com/citations?user=Y-31eCwAAAAJ)
- [Gabriele D'Acunto](https://scholar.google.com/citations?user=dIVgmlUAAAAJ)
- [Sergio Barbarossa](https://scholar.google.com/citations?user=2woHFu8AAAAJ)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?user=VZYvspQAAAAJ)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-89CFF0?style=for-the-badge&logo=hyperland&logoColor=white)
![w&b](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
![Semasia](https://img.shields.io/badge/Semasia-8A2BE2?style=for-the-badge&logo=huggingface&logoColor=white)
