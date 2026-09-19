# GMN Synthetic GED and COIL-DEL Dataset Preparation

This README explains how to generate and use the datasets for the
**Graph Matching Networks for Learning the Similarity of Graph Structured Objects**
baseline experiments.

## 1. Requirements

Use Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Generate the datasets

Run:

```bash
python prepare_gmn_data.py
```

The script creates:

```text
data/
├── ged/
│   ├── n20_p02/
│   │   ├── pairs.pt
│   │   ├── triplets.pt
│   │   └── metadata.json
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

# 3. Synthetic GED experiments

The synthetic experiments use the four settings:

```text
(n=20, p=0.2)
(n=20, p=0.5)
(n=50, p=0.2)
(n=50, p=0.5)
```

For each setting:

- Positive pair: graphs differ by 1 edge substitution.
- Negative pair: graphs differ by 2 edge substitutions.
- 1,000 fixed evaluation pairs are generated.
- 1,000 fixed evaluation triplets are generated.

## Important: training data

Do **not** save millions of training graphs to disk.

Generate training samples on-the-fly:

```python
from prepare_gmn_data import GEDGenerator

generator = GEDGenerator(
    n=20,
    p=0.2,
    k_pos=1,
    k_neg=2,
    seed=42
)

# Pair training sample
g1, g2, label = generator.sample_pair()

# Negative pair
g1, g2, label = generator.sample_negative_pair()

# Triplet training sample
anchor, positive, negative = generator.sample_triplet()
```

Create fresh samples for every training batch.

For the paper-style training budget, use:

```text
batch_size = 20
training_steps = 50,000
```

This gives 1,000,000 sampled training examples without storing them.

---

# 4. Loading fixed GED evaluation data

```python
import torch

pairs = torch.load("data/ged/n20_p02/pairs.pt")
triplets = torch.load("data/ged/n20_p02/triplets.pt")
```

Each pair has:

```text
(graph1, graph2, label)
```

where:

```text
label = 1 -> positive
label = 0 -> negative
```

Each triplet contains:

```text
(anchor, positive, negative)
```

Use these fixed files for validation/test evaluation so results are comparable across runs.

---

# 5. COIL-DEL experiments

The script downloads COIL-DEL through PyTorch Geometric.

The split is class-balanced:

```text
Train:  2,400 graphs
Validation: 500 graphs
Test:   1,000 graphs
```

That is:

```text
24 train graphs/class
5 validation graphs/class
10 test graphs/class
```

The exact split is saved in:

```text
data/coil_del/splits.pt
```

Load it using:

```python
import torch

splits = torch.load("data/coil_del/splits.pt")

train_idx = splits["train"]
val_idx = splits["val"]
test_idx = splits["test"]
```

## Positive and negative pairs

For COIL-DEL:

```text
same class     -> positive pair
different class -> negative pair
```

Do not precompute all possible graph pairs.

Instead, sample pairs during training:

```text
training batch
    |
    +-- choose graph A
    |
    +-- choose another graph from same class
          -> positive

or

    +-- choose graph A
    |
    +-- choose graph from another class
          -> negative
```

Keep validation/test graph identities fixed and use a fixed random seed for reproducibility.

---

# 6. Converting graphs to GMN input

The GMN implementation expects graph structure plus node features.

For the structure-only baseline, use constant node features:

```python
x = torch.ones(num_nodes, 1)
```

For each graph maintain:

```text
node_features
adjacency_matrix
num_nodes
```

A GMN pair therefore looks like:

```text
Graph 1:
    X1
    A1
    n1

Graph 2:
    X2
    A2
    n2
```

The exact tensor layout should be adapted to the
`GraphMatchingNetworks-PyTorch` repository's model interface.

---

# 7. Recommended experiment order

Run the experiments in this order:

## Experiment 1 — Synthetic baseline

Run all four:

```text
n20_p02
n20_p05
n50_p02
n50_p05
```

Record:

```text
Pair AUC
Triplet Accuracy
Training loss
Validation loss
Training time
Inference time
```

## Experiment 2 — COIL-DEL baseline

Train on:

```text
2,400 train graphs
```

Tune/check performance on:

```text
500 validation graphs
```

Report final performance on:

```text
1,000 test graphs
```

## Experiment 3 — Freeze the baseline

Do not modify GMN until the baseline results and preprocessing are fixed.

Then use the same:

```text
dataset
splits
training budget
evaluation code
random seeds
```

for every extension.

---

# 8. Reproducibility

Use fixed seeds for evaluation:

```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

For each extension, keep the dataset and test sets unchanged.

Only change the component being studied.

Example:

```text
Baseline:
GMN + original cross-graph attention

Extension:
GMN + Sinkhorn matching
```

Everything else should remain identical.

---

# 9. Suggested project structure

```text
project/
├── prepare_gmn_data.py
├── README.md
│
├── data/
│   ├── ged/
│   │   ├── n20_p02/
│   │   ├── n20_p05/
│   │   ├── n50_p02/
│   │   └── n50_p05/
│   │
│   └── coil_del/
│
├── src/
│   ├── model/
│   ├── datasets/
│   ├── training/
│   └── evaluation/
│
├── experiments/
│   ├── ged_baseline/
│   ├── coil_baseline/
│   └── extensions/
│
└── results/
    ├── tables/
    └── figures/
```

This keeps dataset preparation, baseline implementation, extensions, and results separate.
