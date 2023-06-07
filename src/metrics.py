import torch
import torch.nn.functional as F


def masked_accuracy(predictions: torch.Tensor, truths: torch.Tensor, mask: torch.Tensor):
    _, predictions = torch.max(predictions, 1)
    truths = torch.masked_select(truths, mask)
    predictions = torch.masked_select(predictions, mask)
    accuracy = (truths == predictions).double().mean()
    return accuracy


def masked_cross_entropy(predictions: torch.Tensor, truths: torch.Tensor, mask: torch.Tensor):
    loss_per_element = F.cross_entropy(predictions, truths, reduction="none")
    masked_loss = loss_per_element * mask
    return masked_loss.sum() / (mask.sum() + 1e-8)


def mrr_at_k(predictions, truths, k):
    _, predictions = torch.max(predictions, 1)
    predictions = torch.masked_select(predictions, mask)
    truths = torch.masked_select(truths, mask)

    scores = []
    for pred, truth in zip(predictions, truths):
        for i, p in enumerate(pred[:k]):
            if p in truth:
                scores.append(1 / (i + 1))
                break
    return torch.mean(torch.tensor(scores)) if scores else 0


def precision_at_k(predictions, truths, k):
    _, predictions = torch.max(predictions, 1)
    predictions = torch.masked_select(predictions, mask)
    truths = torch.masked_select(truths, mask)

    scores = []
    for pred, truth in zip(predictions, truths):
        pred_k = pred[:k]
        scores.append(len(set(pred_k.tolist()) & set(truth.tolist())) / len(pred_k))
    return torch.mean(torch.tensor(scores))


def dcg_at_k(r, k):
    r = r[:k]
    return torch.sum((2 ** r / torch.log2(torch.tensor(list(range(2, len(r) + 2, 1)), dtype=torch.float))))


def ndcg_at_k(predictions, truths, k):
    _, predictions = torch.max(predictions, 1)
    predictions = torch.masked_select(predictions, mask)
    truths = torch.masked_select(truths, mask)

    scores = []
    for pred, truth in zip(predictions, truths):
        pred_k = [int(p.item() in truth) for p in pred[:k]]
        truth_k = [1] * len(pred_k)
        idcg = masked_dcg_at_k(torch.tensor(truth_k, dtype=torch.float), k)
        if idcg == 0:
            scores.append(0.)
        else:
            scores.append(masked_dcg_at_k(torch.tensor(pred_k, dtype=torch.float), k) / idcg)
    return torch.mean(torch.tensor(scores))


def hr_at_k(predictions, truths, k):
    _, predictions = torch.max(predictions, 1)
    predictions = torch.masked_select(predictions, mask)
    truths = torch.masked_select(truths, mask)

    scores = []
    for pred, truth in zip(predictions, truths):
        pred_k = pred[:k]
        hit = len(set(pred_k.tolist()) & set(truth.tolist())) > 0
        scores.append(int(hit))
    return torch.mean(torch.tensor(scores, dtype=torch.float))
