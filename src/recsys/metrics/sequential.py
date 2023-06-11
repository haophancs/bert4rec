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


def hr_at_k(predictions, truths, k_values):
    top_indices = torch.argsort(predictions[:, -1], dim=1, descending=True)
    ranks = (top_indices == truths()).long().argmax(dim=1)
    return {f'HR@{k}': (ranks < k).sum() for k in k_values}
