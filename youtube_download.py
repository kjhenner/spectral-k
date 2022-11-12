#!/usr/bin/env python3

"""
Convert a set of audio files into spectrogram slices.
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Tuple, List

import tqdm
import yt_dlp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str
    )
    parser.add_argument(
        "--output-dir",
        type=str
    )
    return parser.parse_args()


def process_item(
    target: Tuple[str, str],
    output_dir: str,
) -> None:
    video_url, artist_name = target
    video_id = video_url.split('=')[-1]
    path = os.path.join(output_dir, f"{artist_name}_{video_url.split('=')[-1]}.wav")
    try:
        options = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": 192
            }],
            "outtmpl": path
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"skipping {video_id}")


def process_list(
        targets: List[Tuple[str, str]],
        output_dir: str,
) -> None:
    multiprocess = True
    process_func = partial(
        process_item,
        output_dir=output_dir,
    )
    if multiprocess:
        with Pool(max(cpu_count() - 2, 1)) as p:
            list(tqdm.tqdm(p.imap(process_func, targets), total=len(targets)))
    else:
        for path in tqdm.tqdm(targets):
            process_func(path)


def main(args) -> None:
    targets = []
    with open(args.input_path) as f:
        artist_name = None
        for line in f.readlines():
            if 'https:' in line:
                targets.append((line.strip(), artist_name))
            else:
                artist_name = line.strip()
    process_list(targets, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
