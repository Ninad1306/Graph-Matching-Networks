# Graph Matching Networks (GMN) Dataset Preparation

This repository contains dataset preparation scripts for reproducing the experiments from:

> **Graph Matching Networks for Learning the Similarity of Graph Structured Objects**

The repository supports three datasets used for baseline reproduction and extension experiments:

| Dataset | Task | Graph Type |
|----------|------|------------|
| Synthetic GED | Graph similarity / matching | Random graphs |
| COIL-DEL | Graph similarity / classification | Object graphs |
| FFmpeg CFG | Binary function similarity | Control Flow Graphs (CFGs) |

---

# 1. Installation

## System Requirements

Ubuntu/Linux is recommended.

Python:

```bash
Python >= 3.10
```

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Additional Dependencies for FFmpeg CFG Dataset

```bash
sudo apt update

sudo apt install \
    build-essential \
    clang \
    make \
    pkg-config \
    yasm \
    nasm
```

---

# 2. Dataset Generation

## Synthetic GED + COIL-DEL

Generate both datasets using:

```bash
python prepare_gmn_data.py
```

Output:

```text
data/
├── ged/
│   ├── n20_p02/
│   ├── n20_p05/
│   ├── n50_p02/
│   └── n50_p05/
│
└── coil_del/
    ├── raw/
    ├── processed/
    ├── splits.pt
    └── metadata.json
```

---

## FFmpeg CFG Dataset

Download FFmpeg:

```bash
wget https://ffmpeg.org/releases/ffmpeg-6.1.tar.xz
tar -xf ffmpeg-6.1.tar.xz
```

Generate CFG dataset:

```bash
python prepare_ffmpeg_cfg.py all \
    --source-dir ffmpeg-6.1 \
    --out data/ffmpeg \
    --gcc gcc \
    --clang clang \
    --jobs 8
```

Output:

```text
data/ffmpeg/
├── binaries/
│   ├── gcc_O0/
│   ├── gcc_O1/
│   ├── gcc_O2/
│   ├── gcc_O3/
│   ├── clang_O0/
│   ├── clang_O1/
│   ├── clang_O2/
│   └── clang_O3/
│
├── ffmpeg_cfg.sqlite
├── splits.json
├── build_metadata.json
└── extraction_counts.json
```

---

# 3. Dataset Summary

## A. Synthetic GED Dataset

Purpose:

```text
Graph similarity benchmark
```

Configurations:

```text
n=20, p=0.2
n=20, p=0.5
n=50, p=0.2
n=50, p=0.5
```

Positive Pair:

```text
1 edge substitution
```

Negative Pair:

```text
2 edge substitutions
```

Evaluation Data:

```text
1000 fixed pairs
1000 fixed triplets
```

Training Data:

```text
Generated on-the-fly
```

Do not store training pairs on disk.

---

## B. COIL-DEL Dataset

Purpose:

```text
Graph similarity benchmark on real object graphs
```

Statistics:

```text
100 classes
3900 graphs
```

Split:

```text
Train      2400
Validation  500
Test       1000
```

Class-wise:

```text
24 train graphs/class
5 validation graphs/class
10 test graphs/class
```

Split file:

```text
data/coil_del/splits.pt
```

Positive Pair:

```text
Same class
```

Negative Pair:

```text
Different class
```

Training pairs should be sampled dynamically during training.

---

## C. FFmpeg CFG Dataset

Purpose:

```text
Binary function similarity
```

Source Project:

```text
FFmpeg 6.1
```

Compilation Variants:

```text
GCC   : O0 O1 O2 O3
Clang : O0 O1 O2 O3
```

Maximum Variants per Function:

```text
8
```

Generated Dataset:

```text
Function-level CFGs
```

Each CFG stores:

```text
Function name
Compiler
Optimization level
Entry address
Basic blocks
CFG edges
Assembly instructions
```

Train / Validation / Test Split:

```text
80% / 10% / 10%
```

Split is performed by:

```text
Function identity
```

meaning all variants of the same function remain in the same split.

Default filtering:

```bash
--min-variants 2
```

Stricter benchmark:

```bash
python prepare_ffmpeg_cfg.py split \
    --out data/ffmpeg \
    --min-variants 8 \
    --seed 42
```

---

# 4. Training Protocol

## Synthetic GED

Create training samples dynamically:

```python
from prepare_gmn_data import GEDGenerator

generator = GEDGenerator(
    n=20,
    p=0.2,
    k_pos=1,
    k_neg=2,
    seed=42
)

g1, g2, label = generator.sample_pair()
anchor, positive, negative = generator.sample_triplet()
```

Recommended training budget:

```text
batch_size = 20
training_steps = 50000
```

---

## COIL-DEL

Positive:

```text
same class
```

Negative:

```text
different class
```

Generate training pairs online.

Keep validation/test graphs fixed.

---

## FFmpeg CFG

Positive Pair:

```text
same function
different compiler/optimization level
```

Examples:

```text
avcodec_open2 (gcc O0)
vs
avcodec_open2 (clang O3)
```

Negative Pair:

```text
different functions
```

Examples:

```text
avcodec_open2
vs
avformat_open_input
```

---

# 5. Loading Evaluation Data

## GED

```python
import torch

pairs = torch.load("data/ged/n20_p02/pairs.pt")
triplets = torch.load("data/ged/n20_p02/triplets.pt")
```

Pair format:

```text
(graph1, graph2, label)
```

where:

```text
label = 1 → positive
label = 0 → negative
```

Triplet format:

```text
(anchor, positive, negative)
```

---

## COIL-DEL

```python
import torch

splits = torch.load("data/coil_del/splits.pt")

train_idx = splits["train"]
val_idx = splits["val"]
test_idx = splits["test"]
```

---

## FFmpeg

```python
import sqlite3

conn = sqlite3.connect("data/ffmpeg/ffmpeg_cfg.sqlite")
```

Use `splits.json` for train/validation/test functions.

---

# 6. GMN Input Format

The GMN implementation expects:

```text
Node Features
Adjacency Structure
Number of Nodes
```

Structure-only baseline:

```python
x = torch.ones(num_nodes, 1)
```

Each graph should provide:

```text
node_features
adjacency_matrix
num_nodes
```

For FFmpeg CFG:

```text
Option 1:
Use constant node features
(structure-only baseline)

Option 2:
Use assembly instructions
(node attributes)
```

---

# 7. Reproducibility

Use fixed random seeds:

```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

Keep the following unchanged across all extensions:

```text
Dataset
Splits
Evaluation sets
Training budget
Random seeds
```

Only modify the component being studied.

Example:

```text
Baseline:
GMN + Original Cross-Graph Attention

Extension:
GMN + Sinkhorn Matching
```

---

# 8. Recommended Workflow

Step 1:

```text
Reproduce Synthetic GED baseline
```

Step 2:

```text
Reproduce COIL-DEL baseline
```

Step 3:

```text
Reproduce FFmpeg CFG baseline
```

Step 4:

```text
Freeze preprocessing, splits, and evaluation code
```

Step 5:

```text
Implement extension
```

Step 6:

```text
Compare against baseline
```

Recommended extensions:

```text
Multi-head Cross-Graph Attention
Sinkhorn Matching
Attention Pooling
Alternative Metric Learning Losses
Scalability Analysis
Interpretability Analysis
```

---

# 9. Notes for FFmpeg CFG Extraction

CFG extraction is significantly slower than GED or COIL-DEL generation.

Recommendations:

1. Use FFmpeg 6.1.

2. Enable progress bars in CFGFast:

```python
cfg = project.analyses.CFGFast(
    normalize=True,
    show_progressbar=True
)
```

3. Do not use:

```python
force_complete_scan=True
```

unless necessary.

4. Extraction of one FFmpeg binary may take:

```text
20–60 minutes
```

depending on CPU, angr version, and compiler output.

5. Monitor progress using:

```bash
htop
```

If Python is actively consuming CPU, CFG recovery is still running.

---

# 10. Project Structure

```text
project/
├── prepare_gmn_data.py
├── prepare_ffmpeg_cfg.py
├── requirements.txt
├── requirements-ffmpeg.txt
├── README.md
│
├── data/
│   ├── ged/
│   ├── coil_del/
│   └── ffmpeg/
│
├── src/
│   ├── datasets/
│   ├── model/
│   ├── training/
│   └── evaluation/
│
├── experiments/
│   ├── ged_baseline/
│   ├── coil_baseline/
│   ├── ffmpeg_baseline/
│   └── extensions/
│
└── results/
    ├── tables/
    └── figures/
```

This structure keeps dataset preparation, baseline reproduction, extensions, and evaluation cleanly separated.