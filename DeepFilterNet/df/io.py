import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchaudio as ta
from loguru import logger
from numpy import ndarray
from torch import Tensor

from dataclasses import dataclass

@dataclass
class SimpleAudioMetaData:
    sample_rate: int
    num_channels: int
    num_frames: int

try:
    from torchaudio import AudioMetaData
    TA_RESAMPLE_SINC = "sinc_interp_hann"
    TA_RESAMPLE_KAISER = "sinc_interp_kaiser"
except Exception:
    AudioMetaData = SimpleAudioMetaData
    TA_RESAMPLE_SINC = "sinc_interpolation"
    TA_RESAMPLE_KAISER = "kaiser_window"

import soundfile as sf
from df.logger import warn_once
from df.utils import download_file, get_cache_dir, get_git_root


def load_audio(
    file: str, sr: Optional[int] = None, verbose=True, **kwargs
) -> Tuple[Tensor, Any]:
    """Loads an audio file using soundfile with torchaudio fallback."""
    try:
        data, orig_sr = sf.read(file, dtype="float32")
        if data.ndim == 1:
            audio = torch.from_numpy(data).unsqueeze(0)
            channels = 1
            frames = len(data)
        else:
            audio = torch.from_numpy(data.T)
            channels = data.shape[1]
            frames = data.shape[0]
        info = SimpleAudioMetaData(sample_rate=orig_sr, num_channels=channels, num_frames=frames)
    except Exception:
        info = ta.info(file)
        audio, orig_sr = ta.load(file, **kwargs)

    rkwargs = {}
    if "method" in kwargs:
        rkwargs["method"] = kwargs.pop("method")

    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, orig_sr, sr, **rkwargs)
    return audio.contiguous(), info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    log: bool = False,
    dtype=torch.int16,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")
    
    if isinstance(audio, Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio)
    
    if audio_np.ndim == 2:
        if audio_np.shape[0] <= 2 and audio_np.shape[1] > audio_np.shape[0]:
            audio_np = audio_np.T
        if audio_np.shape[1] == 1:
            audio_np = audio_np.squeeze(1)

    subtype = "PCM_16" if dtype == torch.int16 else "FLOAT"
    sf.write(outpath, audio_np, sr, subtype=subtype)


try:
    from torchaudio.functional import resample as ta_resample
except ImportError:
    from torchaudio.compliance.kaldi import resample_waveform as ta_resample  # type: ignore


def get_resample_params(method: str) -> Dict[str, Any]:
    params = {
        "sinc_fast": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = get_resample_params(method)
    return ta_resample(audio, orig_sr, new_sr, **params)


def get_test_sample(sr: int = 48000) -> Tensor:
    dir = get_git_root()
    file_path = os.path.join("assets", "clean_freesound_33711.wav")
    if dir is None:
        url = "https://github.com/Rikorose/DeepFilterNet/raw/main/" + file_path
        save_dir = get_cache_dir()
        path = download_file(url, save_dir)
    else:
        path = os.path.join(dir, file_path)
    sample, _ = load_audio(path, sr=sr)
    return sample
