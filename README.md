# Audio Diffusion Model Tools

This repository provides tools to source audio data from YouTube, convert that data into spectrograms, and train a diffusion model based on those spectrograms. 

## Features
- Downloading audio data from YouTube
- Conversion of audio into spectrogram slices
- Training diffusion models adapted from [Karras et al. (2022) diffusion models](https://github.com/crowsonkb/k-diffusion) on the generated spectrograms.

The models and tools used are modular and could be extended to other applications related to audio processing and machine learning on audio data.

## Usage
There are three main steps to use these tools - sourcing the audio data, converting the data into spectrograms, and training a diffusion model on those spectrograms.

### Sourcing Audio Data 

To download audio data from YouTube, you would use the script in the `youtube_download.py`. This script accepts an input file with URLs of YouTube videos, then downloads those videos and saves them in the `.wav` format in a specified directory.

**Before using this script, you are responsible for ensuring that your usage adheres to the YouTube ToS and relevant copywrite law!** 

```sh
python youtube_download.py --input-path <input_file> --output-dir <output_dir>
```

### Converting Audio Data into Spectrogram Slices

Once the data has been sourced, it can be converted into spectrograms. The script `preprocess_dataset.py` will convert the audio files into overlapping fixed-length slices and save them as images in PNG format.

```sh
python preprocess_dataset.py --input-dir <input_dir> --output-dir <output_dir>
```

You can optionally specify the sample rate for the audio to be read with the `--sample-rate` flag.

### Training a Diffusion Model 

Lastly, a diffusion model can be trained on the spectrogram images that have been sourced and prepared. This is done using the script `train.py`. This script is adapted from the Karras et al. (2022) diffusion models repository and uses the `k-diffusion` and `spectral_k` libraries. 

```sh
python train.py --config ./configs/config_256x1024.json --name run_name
```
