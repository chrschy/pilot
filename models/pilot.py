import math
from .modules import FeatureExtraction, LinearGaussianSystem
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple


class PILOT(nn.Module):
    def __init__(self,
                 chunk_length: float = 2.,
                 frame_length: float = 0.04,
                 num_fft_bins: int = 2048,
                 embedding_dim: int = 32,
                 feature_extraction_dropout: float = 0.,
                 transformer_num_heads: int = 8,
                 transformer_num_layers: int = 3,
                 transformer_feedforward_dim: int = 1024,
                 transformer_dropout: float = 0.1,
                 num_sources_output: int = 3) -> None:
        """The Probabilistic Localization of Sounds with Transformers (PILOT) framework main class.
        Args:
            chunk_length (float): Signal chunk (processing block) length in seconds.
            frame_length (float): Frame length within chunks in seconds.
            num_fft_bins (int): Number of frequency bins used for representing the input spectrograms.
            embedding_dim (int): Dimension of the feature embeddings.
            feature_extraction_dropout (float): Dropout rate used within the feature extraction stage.
            transformer_num_heads (int): Number of heads in the transformer stage.
            transformer_num_layers (int): Number of layers in the transformer stage.
            transformer_feedforward_dim (int): Feedforward dimension in the transformer stage.
            transformer_dropout (float): Dropout rate used within the transformer stage.
            num_sources_output (int): Maximum number of sources that should be represented by the model.
        """
        super(PILOT, self).__init__()

        self.num_sources_output = num_sources_output
        self.embedding_dim = embedding_dim

        self.feature_extraction = FeatureExtraction(chunk_length=chunk_length,
                                                    frame_length=frame_length,
                                                    num_fft_bins=num_fft_bins,
                                                    embedding_dim=embedding_dim,
                                                    num_sources_output=num_sources_output,
                                                    dropout_rate=feature_extraction_dropout)

        transformer_layers = nn.TransformerEncoderLayer(
            embedding_dim, transformer_num_heads, transformer_feedforward_dim, transformer_dropout
        )
        self.transformer = nn.TransformerEncoder(transformer_layers, transformer_num_layers)

        self.source_activity_fc = nn.Linear(embedding_dim, 1)

        self.linear_gaussian_system = LinearGaussianSystem(state_dim=2, observation_dim=embedding_dim)
        self.prior_mean = torch.cat((
            torch.linspace(-math.pi, math.pi - 2 * math.pi / num_sources_output, num_sources_output).unsqueeze(-1),
            torch.zeros(num_sources_output).unsqueeze(-1)
        ), dim=-1)
        self.prior_covariance = torch.eye(2).unsqueeze(0).repeat((num_sources_output, 1, 1))

    def forward(self, audio_features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """PILOT forward pass.
        Args:
            audio_features: Audio feature tensor (input spectrogram representation).

        Returns:
            tuple: Estimated source activity, direction-of-arrival and posterior covariance matrix.
        """
        embeddings, observation_noise = self.feature_extraction(audio_features)

        source_activity = self.source_activity_fc(embeddings).squeeze()
        posterior_mean = []
        posterior_covariance = []

        for src_idx in range(self.num_sources_output):
            # Permutation is needed here, because the Transformer class requires sequence length in dimension zero.
            src_embeddings = self.transformer(embeddings[:, src_idx, ...].permute(1, 0, 2)).permute(1, 0, 2)
            src_observation_noise_covariance = torch.diag_embed(observation_noise[:, src_idx, ...])

            posterior_distribution = self.linear_gaussian_system(src_embeddings,
                                                                 src_observation_noise_covariance,
                                                                 prior_mean=self.prior_mean[src_idx, ...],
                                                                 prior_covariance=self.prior_covariance[src_idx, ...])

            posterior_mean.append(posterior_distribution[0])
            posterior_covariance.append(posterior_distribution[1].unsqueeze(-1))

        posterior_mean = torch.cat(posterior_mean, dim=-1).permute(0, 3, 1, 2)
        posterior_covariance = torch.cat(posterior_covariance, dim=-1).permute(0, 4, 1, 2, 3)

        return source_activity, posterior_mean, posterior_covariance
