import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple


def compute_spherical_distance(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Computes the distance between two points (given as angles) on a sphere, as described in Eq. (6) in the paper.

    Args:
        y_pred (Tensor): Tensor of predicted azimuth and elevation angles.
        y_true (Tensor): Tensor of ground-truth azimuth and elevation angles.

    Returns:
        Tensor: Tensor of spherical distances.
    """
    if (y_pred.shape[-1] != 2) or (y_true.shape[-1] != 2):
        assert RuntimeError('Input tensors require a dimension of two.')

    sine_term = torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0])
    cosine_term = torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1] - y_pred[:, 1])

    return torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))


def compute_kld_to_standard_norm(covariance_matrix: Tensor) -> Tensor:
    """Computes the Kullback-Leibler divergence between two multivariate Gaussian distributions with identical mean,
    where the second distribution has an identity covariance matrix.

    Args:
        covariance_matrix (Tensor): Covariance matrix of the first distribution.

    Returns:
        Tensor: Tensor of KLD values.
    """
    matrix_dim = covariance_matrix.shape[-1]

    covariance_trace = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1).sum(-1)

    return 0.5 * (covariance_trace - matrix_dim - torch.logdet(covariance_matrix.contiguous()))


def psel_loss(predictions: Tuple[Tensor, Tensor, Tensor],
              targets: Tuple[Tensor, Tensor],
              alpha: float = 1.,
              beta: float = 1.) -> Tensor:
    """Returns the probabilistic sound event localization loss, as described in Eq. (5) in the paper.

    Args:
        predictions (tuple): Predicted source activity, direction-of-arrival and posterior covariance matrix.
        targets (Tensor): Ground-truth source activity and direction-of-arrival.
        alpha (float): Weighting factor for direction-of-arrival loss component.
        beta (float): Weighting factor for KLD loss component.

    Returns:
        Tensor: Scalar probabilistic SEL loss value.
    """
    source_activity, posterior_mean, posterior_covariance = predictions
    source_activity_target, direction_of_arrival_target = targets

    source_activity_loss = F.binary_cross_entropy(source_activity, source_activity_target)
    source_activity_mask = source_activity_target.bool()

    spherical_distance = compute_spherical_distance(posterior_mean[source_activity_mask],
                                                    direction_of_arrival_target[source_activity_mask])
    direction_of_arrival_loss = torch.mean(spherical_distance)

    kld_loss = compute_kld_to_standard_norm(posterior_covariance)
    kld_loss = torch.mean(kld_loss)

    return source_activity_loss + alpha * direction_of_arrival_loss + beta * kld_loss
