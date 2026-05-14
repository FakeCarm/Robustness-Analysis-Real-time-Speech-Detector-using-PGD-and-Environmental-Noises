"""
balanced_training_utils.py

Utility functions for handling class imbalance during training
(too many spoof samples vs bonafide samples).

Functions:
    - create_weighted_sampler : builds a WeightedRandomSampler from a dataset
    - get_loss_function       : returns the requested loss (CE / BCE / Focal)
    - log_class_distribution  : logs predicted class distribution during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, ConcatDataset
import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Helper: extract all labels from a dataset (supports ConcatDataset)
# ---------------------------------------------------------------------------

def _get_all_labels(dataset):
    """
    Iterate over a dataset and collect every label.

    Supports:
      - ConcatDataset  (torch.utils.data.ConcatDataset)
      - Any dataset whose items are (features, filename, label)  ← your format
      - Any dataset that exposes a .labels / .label_list attribute

    Returns
    -------
    labels : list[int]
    """
    # Fast path: dataset already exposes a label list
    for attr in ("labels", "label_list", "targets"):
        if hasattr(dataset, attr):
            return list(getattr(dataset, attr))

    # ConcatDataset: recurse on each sub-dataset
    if isinstance(dataset, ConcatDataset):
        all_labels = []
        for ds in dataset.datasets:
            all_labels.extend(_get_all_labels(ds))
        return all_labels

    # Slow path: iterate item by item
    # Each item is expected to be (feat, filename, label)
    all_labels = []
    for item in dataset:
        if isinstance(item, (list, tuple)):
            # label is the last element of the tuple returned by __getitem__
            label = item[-1]
        else:
            label = item
        if isinstance(label, torch.Tensor):
            label = label.item()
        all_labels.append(int(label))
    return all_labels


# ---------------------------------------------------------------------------
# 1.  create_weighted_sampler
# ---------------------------------------------------------------------------

def create_weighted_sampler(dataset, verbose: bool = False) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that compensates for class imbalance.

    Each sample is assigned a weight inversely proportional to the frequency
    of its class, so every class is seen at approximately the same rate.

    Parameters
    ----------
    dataset : torch Dataset or ConcatDataset
        The training dataset.  Items must expose a label as the last element
        of the tuple returned by __getitem__.
    verbose : bool
        If True, print class counts and per-class weights.

    Returns
    -------
    sampler : WeightedRandomSampler
        Ready to pass to DataLoader(sampler=...).
    """
    labels = _get_all_labels(dataset)
    class_counts = Counter(labels)
    num_samples  = len(labels)

    if verbose:
        print("\n[WeightedSampler] Class distribution in the training set:")
        for cls, cnt in sorted(class_counts.items()):
            pct = 100.0 * cnt / num_samples
            tag = "bonafide" if cls == 0 else "spoof"
            print(f"   Class {cls} ({tag:>8s}): {cnt:>8d} samples  ({pct:.2f} %)")

    # Weight for each class = 1 / class_frequency
    class_weights = {
        cls: 1.0 / count for cls, count in class_counts.items()
    }

    # Assign the appropriate weight to every individual sample
    sample_weights = torch.tensor(
        [class_weights[lbl] for lbl in labels],
        dtype=torch.float32
    )

    if verbose:
        print("\n[WeightedSampler] Per-class sampling weights:")
        for cls, w in sorted(class_weights.items()):
            tag = "bonafide" if cls == 0 else "spoof"
            print(f"   Class {cls} ({tag:>8s}): weight = {w:.6f}")
        print()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True      # standard for WeightedRandomSampler
    )
    return sampler


# ---------------------------------------------------------------------------
# 2.  Focal Loss (helper class used by get_loss_function)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : list[float] or None
        Per-class weight vector.  If None, all classes are weighted equally.
    gamma : float
        Focusing parameter (default 2.0).
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha=None, gamma: float = 2.0,
                 reduction: str = "mean", num_classes: int = 2):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.register_buffer(
                "alpha",
                torch.ones(num_classes, dtype=torch.float32)
            )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs  : (N, C)  raw logits
        targets : (N,)    integer class indices
        """
        log_probs = F.log_softmax(inputs, dim=1)               # (N, C)
        probs     = torch.exp(log_probs)                        # (N, C)

        # Gather the probability of the true class for each sample
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)   # (N,)
        pt     = probs.gather(1, targets.view(-1, 1)).squeeze(1)       # (N,)

        # Alpha weight per sample
        alpha_t = self.alpha.gather(0, targets)                         # (N,)

        # Focal weight
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma

        loss = -focal_weight * log_pt                                   # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ---------------------------------------------------------------------------
# 3.  get_loss_function
# ---------------------------------------------------------------------------

def get_loss_function(
    loss_type:   str,
    model_name:  str,
    device:      torch.device,
    alpha_focal  = None,
    gamma_focal: float = 2.0,
):
    """
    Return the appropriate loss function for the current training run.

    Parameters
    ----------
    loss_type : str
        One of  'ce'  (CrossEntropy),  'bce'  (BinaryCrossEntropy),
        'focal'  (Focal Loss).
    model_name : str
        Model identifier (used only for logging).
    device : torch.device
        Target device.
    alpha_focal : list[float] or None
        Per-class weights for Focal Loss.
        e.g. [0.25, 0.75] to up-weight the bonafide class.
        If None, equal weights are used.
    gamma_focal : float
        Gamma parameter for Focal Loss (default 2.0).

    Returns
    -------
    criterion : nn.Module
        Loss function moved to `device`.
    """
    loss_type = loss_type.lower()

    print(f"[Loss] Model      : {model_name}")
    print(f"[Loss] Loss type  : {loss_type.upper()}")

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
        print("[Loss] Using standard CrossEntropyLoss (no class weighting).")
        print("[Loss] Tip: use --base_loss focal or --use_weighted_sampler for imbalance.")

    elif loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
        print("[Loss] Using BCEWithLogitsLoss (binary, single output neuron expected).")

    elif loss_type == "focal":
        if alpha_focal is not None:
            alpha_tensor = torch.tensor(alpha_focal, dtype=torch.float32)
            print(f"[Loss] Focal Loss with alpha={alpha_focal}, gamma={gamma_focal}")
        else:
            alpha_tensor = None
            print(f"[Loss] Focal Loss with uniform alpha, gamma={gamma_focal}")
            print("[Loss] Tip: pass alpha_focal=[w_spoof, w_bonafide] to boost bonafide recall.")

        criterion = FocalLoss(
            alpha=alpha_tensor,
            gamma=gamma_focal,
            reduction="mean",
            num_classes=2
        )

    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            "Choose from: 'ce', 'bce', 'focal'."
        )

    return criterion.to(device)


# ---------------------------------------------------------------------------
# 4.  log_class_distribution
# ---------------------------------------------------------------------------

def log_class_distribution(
    epoch:   int,
    phase:   str,
    outputs: torch.Tensor,
    labels:  torch.Tensor,
    topk:    int = 2,
):
    """
    Print a compact summary of predicted vs true class distribution
    for the current mini-batch.  Called periodically during training
    (e.g. every 200 batches) to monitor class-balance health.

    Parameters
    ----------
    epoch   : current epoch number (1-indexed)
    phase   : 'train' or 'val'
    outputs : (N, C) raw logits from the model
    labels  : (N,)   ground-truth class indices
    topk    : number of top classes to display (default 2)
    """
    with torch.no_grad():
        # Predicted class = argmax over logits
        preds = torch.argmax(outputs, dim=1).cpu()
        gt    = labels.cpu()

        n_total   = len(gt)
        n_bonafide_gt   = (gt == 1).sum().item()
        n_spoof_gt      = (gt == 0).sum().item()
        n_bonafide_pred = (preds == 1).sum().item()
        n_spoof_pred    = (preds == 0).sum().item()

        acc = (preds == gt).float().mean().item() * 100.0

        print(
            f"[Ep {epoch:03d} | {phase:5s}] "
            f"Batch size: {n_total} | "
            f"GT  -> bonafide: {n_bonafide_gt:4d}  spoof: {n_spoof_gt:4d} | "
            f"Pred-> bonafide: {n_bonafide_pred:4d}  spoof: {n_spoof_pred:4d} | "
            f"Batch acc: {acc:.1f}%"
        )
