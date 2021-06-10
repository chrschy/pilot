from .losses import compute_spherical_distance
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import Tensor
from torchmetrics import Metric


class FrameRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Args:
            dist_sync_on_step:
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, source_activity_prediction: Tensor, source_activity_target: Tensor) -> None:
        """
        Args:
            source_activity_prediction (Tensor):
            source_activity_target (Tensor):
        """
        assert source_activity_prediction.shape == source_activity_target.shape

        num_active_sources_prediction = torch.sum(source_activity_prediction > 0.5, dim=1)
        num_active_sources_target = torch.sum(source_activity_target, dim=1)

        self.correct += torch.sum(num_active_sources_prediction == num_active_sources_target)
        self.total += source_activity_target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total

    @property
    def is_differentiable(self) -> bool:
        return False


class DOAError(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Args:
            dist_sync_on_step:
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('sum_doa_error', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               source_activity_prediction: Tensor,
               direction_of_arrival_prediction: Tensor,
               source_activity_target: Tensor,
               direction_of_arrival_target: Tensor,
               ) -> None:
        """
        Args:
            source_activity_prediction (Tensor):
            direction_of_arrival_prediction (Tensor):
            source_activity_target (Tensor):
            direction_of_arrival_target (Tensor):
        """
        batch_size, max_num_sources, num_steps = source_activity_prediction.shape

        for batch_idx in range(batch_size):
            for step_idx in range(num_steps):
                active_sources_prediction = source_activity_prediction[batch_idx, :, step_idx] > 0.5
                active_sources_target = source_activity_target.bool()[batch_idx, :, step_idx]
                num_predicted_sources = active_sources_prediction.sum()
                num_target_sources = active_sources_target.sum()

                if (num_predicted_sources > 0) and (num_target_sources > 0):
                    predicted_sources = direction_of_arrival_prediction[batch_idx, active_sources_prediction, step_idx, :]
                    target_sources = direction_of_arrival_target[batch_idx, active_sources_target, step_idx, :]

                    cost_matrix = np.zeros((num_predicted_sources, num_target_sources))

                    for i in range(num_predicted_sources):
                        for j in range(num_target_sources):
                            cost_matrix[i, j] = compute_spherical_distance(predicted_sources[i, :].unsqueeze(0),
                                                                           target_sources[j, :].unsqueeze(0)).cpu().numpy()

                    row_idx, col_idx = linear_sum_assignment(cost_matrix)

                    self.sum_doa_error += np.rad2deg(cost_matrix[row_idx, col_idx].mean())
                    self.total += 1

    def compute(self) -> Tensor:
        return self.sum_doa_error / self.total if self.total > 0 else 180.

    @property
    def is_differentiable(self) -> bool:
        return False
