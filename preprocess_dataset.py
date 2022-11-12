#!/usr/bin/env python3

"""
Convert a set of audio files into spectrogram slices.
"""
import argparse
import glob
import os
import tqdm
import librosa
from scipy.io import wavfile
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image
from spectral_k.mel_helpers import audio_segment_iter, audio_to_image, image_to_audio
from typing import Sequence, List, Optional
import warnings


warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str
    )
    parser.add_argument(
        "--output-dir",
        type=str
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000
    )
    return parser.parse_args()


def get_all_mp3_filenames(
        paths: Sequence[str],
        recursive: bool = True,
        output_path: Optional[str] = None
) -> List[str]:
    extensions = ["mp3", "wav"]
    filenames = []
    for ext_name in extensions:
        ext = f"**/*.{ext_name}" if recursive else f"*.{ext_name}"
        for path in paths:
            filenames.extend(glob.glob(os.path.join(path, ext), recursive=recursive))
    if output_path:
        existing_filenames = get_all_existing_filenames(output_path)
        return [filename for filename in filenames if filename.split('/')[-1] not in existing_filenames]
    return filenames


def get_all_existing_filenames(paths: Sequence[str], recursive: bool = True) -> set:
    extensions = ["png"]
    filenames = []
    for ext_name in extensions:
        ext = f"**/*.{ext_name}" if recursive else f"*.{ext_name}"
        for path in paths:
            filenames.extend(glob.glob(os.path.join(path, ext), recursive=recursive))
    names = []
    for filename in tqdm.tqdm(filenames):
        mp3_file_name = '.'.join(filename.split('/')[-1].split('.')[:-1]) + ".mp3"
        mp3_file_name = mp3_file_name.split('_')[-1]
        names.append(mp3_file_name)
    return set(names)


def process_item(
    file_path: str,
    output_dir: str,
    y_res: int = 256,
    segment_size: int = 2 ** 19,
    target_sr: int = 22050,
    save_audio: bool = False,
) -> None:
    img_file_name = '.'.join(file_path.split('/')[-1].split('.')[:-1]) + ".png"
    try:
        audio, sr = librosa.load(file_path)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        for slice_idx, audio_segment in enumerate(audio_segment_iter(audio, segment_size=segment_size)):
            image = audio_to_image(audio_segment, y_res=y_res, sr=target_sr)
            img_slice_path = os.path.join(output_dir, f"{slice_idx}_" + img_file_name)
            image.save(img_slice_path, compress_level=0)
            if save_audio:
                wav_file_name = '.'.join(file_path.split('/')[-1].split('.')[:-1]) + ".wav"
                wav_slice_path = os.path.join(output_dir, f"{slice_idx}_" + wav_file_name)
                img = Image.open(img_slice_path)
                loaded_audio = image_to_audio(img)
                wavfile.write(wav_slice_path, sr, loaded_audio)
    except:
        print(f"skipping {img_file_name}")


def process_list(
        paths: Sequence[str],
        output_dir: str,
        y_res: int = 256,
        segment_size: int = 2**19,
) -> None:
    multiprocess = True
    process_func = partial(
        process_item,
        output_dir=output_dir,
        y_res=y_res,
        segment_size=segment_size
    )
    if multiprocess:
        with Pool(max(cpu_count() - 2, 1)) as p:
            list(tqdm.tqdm(p.imap(process_func, paths), total=len(paths)))
    else:
        for path in tqdm.tqdm(paths):
            process_func(path)


def main(args) -> None:
    paths = get_all_mp3_filenames([args.input_dir])
    paths.reverse()
    process_list(paths, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
