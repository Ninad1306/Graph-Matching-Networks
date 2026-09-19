import os
import json
import random
import torch
import networkx as nx
from torch_geometric.datasets import TUDataset


# ============================================================
# SYNTHETIC GED DATASET
# ============================================================

class GEDGenerator:
    def __init__(self, n, p, k_pos=1, k_neg=2, seed=42):
        self.n = n
        self.p = p
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.rng = random.Random(seed)

    def _graph(self):
        while True:
            G = nx.erdos_renyi_graph(self.n, self.p, seed=self.rng.randint(0, 10**9))
            if nx.is_connected(G):
                return G

    def _perturb(self, G, k):
        G = G.copy()

        for _ in range(k):
            edges = list(G.edges())
            non_edges = list(nx.non_edges(G))

            # One edge substitution = delete + add
            if edges:
                G.remove_edge(*self.rng.choice(edges))
            if non_edges:
                G.add_edge(*self.rng.choice(non_edges))

        return G

    def _permute(self, G):
        perm = list(G.nodes())
        self.rng.shuffle(perm)
        mapping = {old: new for new, old in enumerate(perm)}
        return nx.relabel_nodes(G, mapping)

    def sample_pair(self):
        G1 = self._graph()

        # Positive = 1 edit
        G2 = self._perturb(G1, self.k_pos)
        G2 = self._permute(G2)

        return G1, G2, 1

    def sample_negative_pair(self):
        G1 = self._graph()

        # Negative = 2 edits
        G2 = self._perturb(G1, self.k_neg)
        G2 = self._permute(G2)

        return G1, G2, 0

    def sample_triplet(self):
        G1 = self._graph()

        G_pos = self._permute(self._perturb(G1, self.k_pos))
        G_neg = self._permute(self._perturb(G1, self.k_neg))

        return G1, G_pos, G_neg


def graph_to_dict(G):
    return {
        "num_nodes": G.number_of_nodes(),
        "edges": list(G.edges()),
        # constant node features as used for structure-only GMN
        "x": torch.ones(G.number_of_nodes(), 1)
    }


def make_fixed_ged_eval(out_dir, n, p, seed=42, num_pairs=1000, num_triplets=1000):
    os.makedirs(out_dir, exist_ok=True)

    gen = GEDGenerator(n, p, seed=seed)

    pairs = []
    for _ in range(num_pairs // 2):
        G1, G2, y = gen.sample_pair()
        pairs.append((graph_to_dict(G1), graph_to_dict(G2), y))

    for _ in range(num_pairs // 2):
        G1, G2, y = gen.sample_negative_pair()
        pairs.append((graph_to_dict(G1), graph_to_dict(G2), y))

    triplets = []
    for _ in range(num_triplets):
        G1, Gp, Gn = gen.sample_triplet()
        triplets.append(
            (graph_to_dict(G1),
             graph_to_dict(Gp),
             graph_to_dict(Gn))
        )

    torch.save(pairs, os.path.join(out_dir, "pairs.pt"))
    torch.save(triplets, os.path.join(out_dir, "triplets.pt"))

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "n": n,
            "p": p,
            "k_positive": 1,
            "k_negative": 2,
            "num_pairs": num_pairs,
            "num_triplets": num_triplets
        }, f, indent=2)


# ============================================================
# COIL-DEL
# ============================================================

def prepare_coil(out_dir="data/coil_del", seed=42):
    dataset = TUDataset(root=out_dir, name="COIL-DEL")

    # group graph indices by class
    class_to_indices = {}

    for i, data in enumerate(dataset):
        label = int(data.y.item())
        class_to_indices.setdefault(label, []).append(i)

    rng = random.Random(seed)

    train_idx, val_idx, test_idx = [], [], []

    # 24 train + 5 val + 10 test per class
    for label, indices in class_to_indices.items():
        rng.shuffle(indices)

        train_idx.extend(indices[:24])
        val_idx.extend(indices[24:29])
        test_idx.extend(indices[29:39])

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    os.makedirs(out_dir, exist_ok=True)

    torch.save(splits, os.path.join(out_dir, "splits.pt"))

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "num_graphs": len(dataset),
            "num_classes": len(class_to_indices),
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        }, f, indent=2)

    print(f"COIL-DEL: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Paper's four main synthetic settings
    configs = [
        (20, 0.2),
        (20, 0.5),
        (50, 0.2),
        (50, 0.5),
    ]

    for n, p in configs:
        name = f"n{n}_p{str(p).replace('.', '')}"
        make_fixed_ged_eval(
            f"data/ged/{name}",
            n=n,
            p=p
        )

    prepare_coil()

    print("Dataset preparation complete.")