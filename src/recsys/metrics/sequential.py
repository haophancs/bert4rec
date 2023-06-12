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


def hr_at_k(predictions: np.array, targets: np.array, k: int):
    hit_counts = np.sum(np.isin(targets, predictions[:, :k]))
    hr_k = hit_counts / len(predictions)
    return hr_k


def ndcg_at_k(predictions: np.array, targets: np.array, k: int):
    ndcg_scores = np.zeros(len(predictions))
    for i in range(len(predictions)):
        target = targets[i]
        prediction = predictions[i][:k]
        mask = np.isin(target, prediction)
        target_ranks = np.where(prediction == target)[0] + 1
        ndcg_scores[i] = np.sum(mask / np.log2(target_ranks + 1))
    ndcg_k = np.mean(ndcg_scores) if len(ndcg_scores) > 0 else 0.0
    return ndcg_k


def mrr(predictions: np.array, targets: np.array):
    reciprocal_ranks = np.array(
        [1 / (np.where(prediction == target)[0][0] + 1) for target, prediction in zip(targets, predictions) if
         target in prediction])
    mrr = np.mean(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0.0
    return mrr
