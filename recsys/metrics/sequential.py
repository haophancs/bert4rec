import torch
import torch.nn.functional as F


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


def hr_at_k(predictions, truths, k):
    top_indices = torch.argsort(predictions[:, -1], dim=1, descending=True)[:, :k]
    true_items = truths.unsqueeze(1)
    hits = torch.any(top_indices == true_items, dim=1).float()
    return hits


def ndcg_at_k(predictions, truths, k):
    top_indices = torch.argsort(predictions[:, -1], dim=1, descending=True)[:, :k]
    truths = truths.unsqueeze(1)
    positions = torch.where(top_indices == truths)[1]
    dcg = 1.0 / torch.log2(positions.float() + 2)
    idcg = 1.0 / torch.log2(torch.tensor(2.0))
    ndcg = dcg / idcg
    ndcg = ndcg if ndcg.numel() > 0 else torch.zeros(1)
    return ndcg


def precision_at_k(predictions, truths, k):
    top_indices = torch.argsort(predictions[:, -1], dim=1, descending=True)[:, :k]
    truths = truths.unsqueeze(1)
    hits = torch.sum(top_indices == truths, dim=1).float()
    precision = hits / k
    precision = precision if precision.numel() > 0 else torch.zeros(1)
    return precision


def mrr_at_k(predictions, truths, k):
    top_indices = torch.argsort(predictions[:, -1], dim=1, descending=True)[:, :k]
    truths = truths.unsqueeze(1)
    positions = torch.where(top_indices == truths)[1]
    reciprocal_ranks = 1.0 / (positions.float() + 1)
    reciprocal_ranks = reciprocal_ranks if reciprocal_ranks.numel() > 0 else torch.zeros(1)
    return reciprocal_ranks


def evaluate_ranking(predictions, truths, k_values):
    evaluation = {}

    for k in k_values:
        hr_scores = hr_at_k(predictions, truths, k)
        ndcg_scores = ndcg_at_k(predictions, truths, k)
        mrr_scores = mrr_at_k(predictions, truths, k)
        prec_scores = precision_at_k(predictions, truths, k)

        evaluation[f'HR@{k}'] = hr_scores
        evaluation[f'NDCG@{k}'] = ndcg_scores
        evaluation[f'MRR@{k}'] = mrr_scores
        evaluation[f'Prec@{k}'] = prec_scores

    return evaluation
