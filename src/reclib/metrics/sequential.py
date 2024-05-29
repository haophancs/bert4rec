import torch
import torch.nn.functional as F  # noqa: WPS301, WPS211, N812, WPS111


def masked_accuracy(
    masked_sequence_predictions: torch.Tensor,
    actual_sequence: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the accuracy of masked sequence predictions.

    :param masked_sequence_predictions: Predictions for the masked sequence.
    :param actual_sequence: The actual sequence.
    :param mask: The mask tensor.
    :return: The accuracy of the masked sequence predictions.
    """
    masked_sequence_predictions = masked_sequence_predictions.view(
        -1,
        masked_sequence_predictions.size(2),
    )
    _, masked_sequence_predictions = torch.max(masked_sequence_predictions, 1)
    actual_sequence = torch.masked_select(actual_sequence, mask)
    masked_sequence_predictions = torch.masked_select(masked_sequence_predictions, mask)
    return (actual_sequence == masked_sequence_predictions).double().mean()


def masked_cross_entropy(
    masked_sequence_predictions: torch.Tensor,
    truths: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the masked cross-entropy loss.

    :param masked_sequence_predictions: Predictions for the masked sequence.
    :param truths: The ground truth labels.
    :param mask: The mask tensor.
    :return: The masked cross-entropy loss.
    """
    masked_sequence_predictions = masked_sequence_predictions.view(
        -1,
        masked_sequence_predictions.size(2),
    )
    loss_per_element = F.cross_entropy(
        masked_sequence_predictions,
        truths,
        reduction="none",
    )
    masked_loss = loss_per_element * mask
    return masked_loss.sum() / (mask.sum() + 1e-8)  # noqa: WPS432
