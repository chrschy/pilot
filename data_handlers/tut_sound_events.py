import numpy as np
from numpy import ndarray
import os
import pandas as pd
from scipy.signal import stft
import soundfile as sf
from torch.utils.data import Dataset
from typing import Tuple


class TUTSoundEvents(Dataset):
    def __init__(self,
                 path_to_dataset: str,
                 train: bool = True,
                 split: Tuple[int, ...] = (1, 2, 3),
                 num_overlapping_sources: Tuple[int, ...] = (1, 2, 3),
                 chunk_length: float = 2.,
                 frame_length: float = 0.04,
                 num_fft_bins: int = 2048,
                 num_sources_output: int = 3) -> None:
        """`TUT Sound Events 2018 <https://zenodo.org/record/1237703>`_ Dataset.

        Args:
            path_to_dataset (str): Root directory of the downloaded dataset.
            train (bool): If True, creates dataset from training set, otherwise creates from test set.
            split (tuple): Indices of the splits the dataset will be created from.
            num_overlapping_sources: Number of overlapping sources that the dataset will be created from.
            chunk_length (float): Length of one chunk (signal block) in seconds.
            frame_length (float): Frame length (within one chunk) in seconds.
            num_fft_bins (int): Number of frequency bins used in the fast Fourier transform (FFT).
            num_sources_output (int): Number of sources represented in the targets.
        """
        self.split = split
        self.num_overlapping_sources = num_overlapping_sources
        self.frame_length = frame_length
        self.num_fft_bins = num_fft_bins
        self.num_sources_output = num_sources_output

        self.chunks = {}

        for audio_subfolder in os.listdir(path_to_dataset):
            if audio_subfolder.startswith('wav'):
                # Get corresponding subfolder containing the annotation files and determine split index and the number
                # of overlapping sources.
                annotation_subfolder = 'desc' + audio_subfolder[3:-5]

                subfolder_split = int(annotation_subfolder[annotation_subfolder.find('split') + 5])
                subfolder_num_overlapping_sources = int(annotation_subfolder[annotation_subfolder.find('ov') + 2])

                # If split index and number of overlapping sources match the specifications, the corresponding sequences
                # will be added to the global list.
                if subfolder_split in split and subfolder_num_overlapping_sources in num_overlapping_sources:
                    for file in os.listdir(os.path.join(path_to_dataset, audio_subfolder)):
                        file_prefix, extension = os.path.splitext(file)

                        if extension == '.wav':
                            audio_file = os.path.join(path_to_dataset, audio_subfolder, file)
                            annotation_file = os.path.join(path_to_dataset, annotation_subfolder, file_prefix + '.csv')

                            is_train_file = file_prefix.startswith('train')

                            if train and is_train_file:
                                self._append_chunks(audio_file, annotation_file, chunk_length)
                            elif not train and not is_train_file:
                                self._append_chunks(audio_file, annotation_file, chunk_length)

    def _append_chunks(self,
                       audio_file: str,
                       annotation_file: str,
                       chunk_length: float) -> None:
        """Splits an audio file into respective chunks and add chunk-level meta data to the list of chunks.

        Args:
            audio_file (str): Path to audio file in *.wav format.
            annotation_file (str): Path to the corresponding annotation file in *.csv format.
            chunk_length (float): Length of one chunk (signal block) in seconds.
        """
        file_info = sf.info(audio_file)
        num_chunks = int(np.ceil(file_info.duration / chunk_length))

        for chunk_idx in range(num_chunks):
            sequence_idx = len(self.chunks)

            start_time = chunk_idx * chunk_length
            end_time = start_time + chunk_length

            self.chunks[sequence_idx] = {
                'audio_file': audio_file,
                'annotation_file': annotation_file,
                'chunk_idx': chunk_idx,
                'start_time': start_time,
                'end_time': end_time
            }

    def _get_audio_features(self,
                            audio_file: str,
                            start_time: float = None,
                            end_time: float = None) -> ndarray:
        """Computes spectrogram audio features for a given chunk from an audio file.

        Args:
            audio_file (str): Path to audio file in *.wav format.
            start_time (float): Chunk start time in seconds.
            end_time (float): Chunk end time in seconds.

        Returns:
            ndarray: Spectrogram audio features.
        """
        file_info = sf.info(audio_file)
        start_idx = int(start_time * file_info.samplerate)
        end_idx = int(end_time * file_info.samplerate)

        audio_data, _ = sf.read(audio_file, start=start_idx, stop=end_idx)

        num_samples, num_channels = audio_data.shape
        required_num_samples = int(file_info.samplerate * (end_time - start_time))

        # Perform zero-padding if required
        if num_samples < required_num_samples:
            audio_data = np.pad(audio_data, ((0, required_num_samples - num_samples), (0, 0)), mode='constant')

        # Compute multi-channel STFT and remove first coefficient and last frame
        frame_length_samples = int(self.frame_length * file_info.samplerate)
        spectrogram = stft(audio_data,
                           fs=file_info.samplerate,
                           nperseg=frame_length_samples,
                           nfft=self.num_fft_bins,
                           axis=0)[-1]
        spectrogram = spectrogram[1:, :, :-1]
        spectrogram = spectrogram.transpose([1, 2, 0])

        audio_features = np.concatenate((np.abs(spectrogram), np.angle(spectrogram)), axis=0)

        return audio_features.astype(np.float32)

    def _get_annotation(self,
                        annotation_file: str,
                        start_time: float = None,
                        end_time: float = None) -> Tuple[ndarray, ndarray]:
        """Returns ground-truth annotations of source activity and direction of arrival.

        Args:
            annotation_file (str): Path to annotation file in *.csv format.
            start_time (float): Chunk start time in seconds.
            end_time (float): Chunk end time in seconds.

        Returns:
            tuple: Binary array of source activity and real-valued array of azimuth and elevation angles.
        """
        annotations = pd.read_csv(annotation_file,
                                  header=0,
                                  names=['sound_event', 'start_time', 'end_time', 'elevation', 'azimuth', 'distance'])
        annotations = annotations.sort_values('start_time')

        chunk_length = end_time - start_time

        event_start_time = annotations['start_time'].to_numpy()
        event_end_time = annotations['end_time'].to_numpy()

        num_frames_per_chunk = int(2 * chunk_length / self.frame_length)

        source_activity = np.zeros((self.num_sources_output, num_frames_per_chunk), dtype=np.uint8)
        direction_of_arrival = np.zeros((self.num_sources_output, num_frames_per_chunk, 2), dtype=np.float32)

        for frame_idx in range(num_frames_per_chunk):
            frame_start_time = start_time + frame_idx * (self.frame_length / 2)
            frame_end_time = frame_start_time + (self.frame_length / 2)

            event_mask = event_start_time <= frame_start_time
            event_mask = event_mask | ((event_start_time >= frame_start_time) & (event_start_time < frame_end_time))
            event_mask = event_mask & (event_end_time > frame_start_time)

            events_in_chunk = annotations[event_mask]
            num_active_sources = len(events_in_chunk)

            if num_active_sources > 0:
                source_activity[:num_active_sources, frame_idx] = 1
                direction_of_arrival[:num_active_sources, frame_idx, :] = np.deg2rad(
                    events_in_chunk[['azimuth', 'elevation']].to_numpy())

        return source_activity.astype(np.float32), direction_of_arrival.astype(np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> Tuple[ndarray, Tuple[ndarray, ndarray]]:
        sequence = self.chunks[index]

        audio_features = self._get_audio_features(sequence['audio_file'],
                                                  sequence['start_time'],
                                                  sequence['end_time'])

        source_activity, direction_of_arrival = self._get_annotation(sequence['annotation_file'],
                                                                     sequence['start_time'],
                                                                     sequence['end_time'])

        return audio_features, (source_activity, direction_of_arrival)
