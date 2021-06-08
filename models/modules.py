import functools
import operator
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple


class FeatureExtraction(nn.Module):
    def __init__(self,
                 chunk_length: float = 2.,
                 frame_length: float = 0.04,
                 num_fft_bins: int = 2048,
                 embedding_dim: int = 32,
                 num_sources_output: int = 3,
                 dropout_rate: float = 0.0) -> None:
        """Feature extraction stage, as described in Sec. 3.1 of the paper. The utilized convolutional neural network
        structure is based on the framework presented in [1].
        Args:
            chunk_length (float): Signal chunk (processing block) length in seconds.
            frame_length (float): Frame length within chunks in seconds.
            num_fft_bins (int): Number of frequency bins used for representing the input spectrograms.
            embedding_dim (int): Dimension of the feature embeddings.
            num_sources_output (int): Maximum number of sources that should be represented by the model.
            dropout_rate (float): Global dropout rate used within this stage.
        """
        super(FeatureExtraction, self).__init__()

        self.conv_network = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 8), ceil_mode=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True),
            nn.Dropout(p=dropout_rate)
        )

        num_frames_per_chunk = int(2 * chunk_length / frame_length)
        flatten_dim = functools.reduce(operator.mul, list(
            self.conv_network(torch.rand(1, 8, num_frames_per_chunk, num_fft_bins // 2)).shape[slice(1, 4, 2)]))

        self.fc_network = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.embedding_output = nn.Linear(128, (embedding_dim - 1) * num_sources_output)
        self.observation_noise_output = nn.Linear(128, embedding_dim * num_sources_output)

        self.embedding_dim = embedding_dim
        self.num_sources_output = num_sources_output

    def forward(self, audio_features: Tensor) -> Tuple[Tensor, Tensor]:
        """Feature extraction forward pass.
        Args:
            audio_features (Tensor): Audio feature tensor (input spectrogram representation).

        Returns:
            tuple: Feature embeddings and estimated (diagonal) observation noise covariance matrices.
        """
        batch_size, _, num_steps, _ = audio_features.shape

        output = self.conv_network(audio_features)
        output = output.permute(0, 2, 1, 3).flatten(2)

        output = self.fc_network(output)

        embeddings = self.embedding_output(output).view(batch_size, num_steps, self.num_sources_output, -1)
        embeddings = embeddings.permute(0, 2, 1, 3)

        positional_encodings = torch.linspace(0, 1, num_steps, device=audio_features.device)[None, ..., None, None]
        positional_encodings = positional_encodings.repeat((batch_size, 1, self.num_sources_output, 1))
        positional_encodings = positional_encodings.permute(0, 2, 1, 3)

        embeddings = torch.cat((embeddings, positional_encodings), dim=-1)

        observation_noise = self.observation_noise_output(output).view(batch_size, num_steps, self.num_sources_output, -1)
        observation_noise = torch.exp(observation_noise.permute(0, 2, 1, 3))

        return embeddings, observation_noise


class LinearGaussianSystem(nn.Module):
    def __init__(self,
                 state_dim: int,
                 observation_dim: int,
                 prior_mean: Tensor = None,
                 prior_covariance: Tensor = None) -> None:
        """Linear Gaussian system, as described in Sec. 3.3 in the paper.
        Args:
            state_dim (int): State dimension.
            observation_dim (int): Observation dimension.
            prior_mean (Tensor): Prior mean vector.
            prior_covariance (Tensor): Prior covariance matrix.
        """
        super(LinearGaussianSystem, self).__init__()

        self.observation_dim = observation_dim

        self.observation_matrix = nn.Parameter(1e-3 * torch.randn((observation_dim, state_dim)), requires_grad=True)
        self.observation_bias = nn.Parameter(1e-6 * torch.randn(observation_dim), requires_grad=True)

        if prior_mean is not None:
            self.prior_mean = nn.Parameter(prior_mean, requires_grad=False)
        else:
            self.prior_mean = nn.Parameter(torch.zeros(state_dim), requires_grad=False)

        if prior_covariance is not None:
            # We use the precision matrix here to avoid matrix inversion during forward pass
            self.prior_precision = nn.Parameter(torch.inverse(prior_covariance), requires_grad=False)
        else:
            self.prior_precision = nn.Parameter(torch.eye(state_dim), requires_grad=False)

    def forward(self,
                observation: Tensor,
                observation_noise: Tensor = None,
                prior_mean: Tensor = None,
                prior_covariance: Tensor = None) -> Tuple[Tensor, Tensor]:
        """Linear Gaussian system forward pass.
        Args:
            observation (Tensor): Observation vector.
            observation_noise (Tensor): Estimated (or fixed) observation noise covariance matrix (per time-step).
            prior_mean (Tensor): Prior mean vector (class-level prior mean vector will not be used).
            prior_covariance (Tensor): Prior covariance matrix (class-level prior covariance matrix will not be used).
        """
        if observation_noise is None:
            observation_noise = 1e-6 * torch.eye(self.observation_dim, device=observation.device)

        observation_noise_precision = torch.inverse(observation_noise)

        if prior_mean is None:
            prior_mean = self.prior_mean

        if prior_covariance is None:
            prior_precision = self.prior_precision
        else:
            prior_precision = torch.inverse(prior_covariance)

        innovation_covariance = self.observation_matrix.t() @ observation_noise_precision @ self.observation_matrix
        posterior_covariance = torch.inverse(prior_precision + innovation_covariance)

        residual = self.observation_matrix.t() @ observation_noise_precision @ (observation - self.observation_bias).unsqueeze(-1)
        posterior_mean = posterior_covariance @ ((prior_precision @ prior_mean).unsqueeze(-1) + residual)

        return posterior_mean, posterior_covariance
