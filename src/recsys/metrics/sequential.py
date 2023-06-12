import torch
import torch.nn.functional as F
import numpy as np


def masked_accuracy(masked_sequence_predictions: torch.Tensor, actual_sequence: torch.Tensor, mask: torch.Tensor):
    masked_sequence_predictions = masked_sequence_predictions.view(-1, masked_sequence_predictions.size(2))
    _, masked_sequence_predictions = torch.max(masked_sequence_predictions, 1)
    actual_sequence = torch.masked_select(actual_sequence, mask)
    masked_sequence_predictions = torch.masked_select(masked_sequence_predictions, mask)
    accuracy = (actual_sequence == masked_sequence_predictions).double().mean()
    return accuracy


def masked_cross_entropy(masked_sequence_predictions: torch.Tensor, truths: torch.Tensor, mask: torch.Tensor):
    masked_sequence_predictions = masked_sequence_predictions.view(-1, masked_sequence_predictions.size(2))
    loss_per_element = F.cross_entropy(masked_sequence_predictions, truths, reduction="none")
    masked_loss = loss_per_element * mask
    return masked_loss.sum() / (mask.sum() + 1e-8)


def hr_at_k(predictions, truths, k_values):
    top_indices = torch.argsort(predictions[:, -1], dim=1, descending=True)
    ranks = (top_indices == truths.unsqueeze(1)).long().argmax(dim=1)
    return {f'HR@{k}': (ranks < k).sum().item() for k in k_values}


def mrr(predictions, truths):
    top_indices = torch.argsort(predictions[:, -1], dim=1, descending=True)
    ranks = (top_indices == truths.unsqueeze(1)).long().argmax(dim=1) + 1
    reciprocal_ranks = 1.0 / ranks.float()
    return reciprocal_ranks.mean().item()


def ndcg_at_k(predictions, truths, k_values):
    top_indices = torch.argsort(predictions[:, -1], dim=1, descending=True)
    batch_size = predictions.shape[0]
    ndcg_values = torch.zeros(batch_size, len(k_values))

    for k_idx, k in enumerate(k_values):
        for i in range(batch_size):
            rank = 1
            dcg = 0.0
            idcg = 0.0
            truth = truths[i]
            for j in top_indices[i]:
                if rank > k:
                    break
                if j == truth:
                    dcg += 1.0 / np.log2(rank + 1)
                idcg += 1.0 / np.log2(rank + 1)
                rank += 1
            ndcg_values[i, k_idx] = dcg / idcg if idcg > 0 else 0.0

    return {f'NDCG@{k}': ndcg_values[:, k_idx].mean().item() for k_idx, k in enumerate(k_values)}