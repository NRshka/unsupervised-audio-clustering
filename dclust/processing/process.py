import soundfile
import numpy as np
import torch
import os
from tqdm import tqdm
from dclust.processing import spectral


class FileProcessor:
    def __init__(self, **kwargs):
        pass


    def process(
        self, file,
        _window_width = 0.08,
        _window_overlap = 0.04,
        _mel_spectrum = 128,
        _clip_below = -60,
        _fixed_length = None,
        _center_fixed = False,
        _pre_emphasis = True,
        _mel_fbanks = {},
        _freq_points = {},
        _mean_norm = True,
        _mel_bands = 128,
        _clip_power_above = None,
        _clip_power_below = None,
        _chunk_length = None,
    ):
        # read data
        samples, fs = soundfile.read(str(file), always_2d=True)

        if _fixed_length is not None:
            fixed_length_samples = int(_fixed_length * fs)

            num_samples = samples.shape[0]

            if num_samples > fixed_length_samples:
                if _center_fixed:
                    offset = (num_samples - fixed_length_samples) // 2
                else:
                    offset = 0

                samples = samples[offset:offset + fixed_length_samples]
            else:
                if _center_fixed:
                    pad_start = (fixed_length_samples - num_samples) // 2
                    pad_end = fixed_length_samples - num_samples - pad_start
                else:
                    pad_start = 0
                    pad_end = fixed_length_samples - num_samples

                samples = np.pad(samples, pad_width=((pad_start, pad_end), (0, 0)),
                                mode="constant")


        # pre-emphasis
        if _pre_emphasis:
            samples = spectral.pre_emphasis_filter(samples)

        # power spectrum
        window_width_samples = int(fs * _window_width)
        window_overlap_samples = int(fs * _window_overlap)

        if _fixed_length is not None:
            expected_step = _window_width - _window_overlap
            expected_num_frames = int((_fixed_length - _window_overlap) / expected_step)

            actual_step = window_width_samples - window_overlap_samples
            actual_num_frames = (samples.shape[0] - window_overlap_samples) // actual_step

            if actual_num_frames > expected_num_frames:
                window_overlap_samples -= 1
            
            elif actual_num_frames < expected_num_frames:
                window_overlap_samples += 1

        window_step_samples = window_width_samples - window_overlap_samples

        f, t, sxx = spectral.power_spectrum(samples, fs, window_width_samples, window_overlap_samples)

        # convert to mel spectrum
        if _mel_bands is not None:
            if fs not in _mel_fbanks:
                freq_points, mel_fbank = spectral.mel_filter_bank(fs, window_width_samples,
                                                                _mel_bands)

                valid_banks = np.sum(mel_fbank, axis=1) > 0
                zero_count = _mel_bands - len(np.nonzero(valid_banks)[0])

                _freq_points[fs] = freq_points
                _mel_fbanks[fs] = mel_fbank

            f = _freq_points[fs]
            sxx = spectral.mel_spectrum(sxx, mel_fbank=_mel_fbanks[fs])

        # convert power scale to dB scale, optionally with amplitude clipping
        sxx = spectral.power_to_db(sxx, clip_above=_clip_power_above, clip_below=_clip_power_below)

        if _chunk_length is not None:
            chunk_length_frames = int(fs * _chunk_length)
            chunk_length_windows = chunk_length_frames // window_step_samples
            num_windows = sxx.shape[1]

            # last axis is time
            indices = np.arange(chunk_length_windows, num_windows, chunk_length_windows)
            sxx = np.split(sxx, indices_or_sections=indices, axis=1)
            t = np.split(t, indices_or_sections=indices)

            if sxx[-1].shape[1] != sxx[0].shape[1] and len(sxx) < _chunk_count + 1:
                raise ValueError("too few chunks: expected at least {}, got {}".format(_chunk_count + 1, len(sxx)))

            sxx = sxx[:_chunk_count]
            t = t[:_chunk_count]
        else:
            sxx = [sxx]
            t = [t]

        # mean normalization
        if _mean_norm:
            sxx = [x - np.mean(x) for x in sxx]
            sxx_min = [np.min(x) for x in sxx]
            sxx_max = [np.max(x) for x in sxx]

            result = []

            for x, x_min, x_max in zip(sxx, sxx_min, sxx_max):
                if abs(x_max - x_min) > 1e-4:
                    result.append(2 * (x - x_min) / (x_max - x_min) - 1)
                else:
                    result.append(x - x_min)

            return result
        else:
            return sxx


class FolderProcessor(FileProcessor):
    def __init__(self, dirpath: str):
        data = []
        for filename in os.listdir(dirpath):
            data.append(self.process(os.path.join(dirpath, filename)))
        self.data = torch.Tensor(data).squeeze_().permute(0, 2, 1) # permute to (batch_size, freq, timesteps)

    def get_dataloader(self, batch_size, shuffle):
        dataset = torch.utils.data.TensorDataset(self.data)
        return torch.utils.data.DataLoader(dataset,
                                                                            batch_size=batch_size,
                                                                            shuffle=shuffle)


def collect_data(filenames, primary_path=None):
    processor = FileProcessor()
    data = []
    print('Collecting data')
    for file_path in tqdm(filenames):
        if primary_path:
            file_path= os.path.join(primary_path, file_path)
        data.append(processor.process(file_path))

    return torch.Tensor(data).squeeze_().permute(0, 2, 1) # permute to (batch_size, freq, timesteps)