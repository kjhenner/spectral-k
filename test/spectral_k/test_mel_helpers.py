import pytest
import os
import pathlib
import librosa
from PIL import Image
from scipy.io import wavfile
from spectral_k.mel_helpers import (
    audio_segment_iter,
    audio_to_log_mel,
    log_mel_spectrogram_to_image,
    image_to_log_mel_spectrogram,
    log_mel_to_audio,
    audio_to_image,
    image_to_audio
)


@pytest.fixture
def wav_sample():
    path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "fixtures", "crecy.wav"
    )
    return librosa.load(path, sr=48000)


@pytest.fixture
def wav_sample_segment(wav_sample):
    audio, sr = wav_sample
    return next(audio_segment_iter(audio, segment_size=2**18))


@pytest.fixture
def wav_sample_mel_spectrogram(wav_sample_segment):
    return audio_to_log_mel(wav_sample_segment, 256)


@pytest.fixture
def mp3_sample():
    path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "fixtures", "44100.mp3"
    )
    return librosa.load(path, sr=44100)


@pytest.fixture
def mp3_sample_segment(mp3_sample):
    audio, sr = mp3_sample
    return next(audio_segment_iter(audio, segment_size=2**18))


@pytest.fixture
def mp3_sample_mel_spectrogram(mp3_sample_segment):
    return audio_to_log_mel(mp3_sample_segment, 256)


def test_audio_segment_iter(wav_sample):
    segment_size = 2**18
    audio, sr = wav_sample
    assert sr == 48000
    segments = list(audio_segment_iter(audio, segment_size=segment_size))
    assert all(len(segment) == segment_size for segment in segments)


def test_audio_to_log_mel_and_back(wav_sample_segment, mp3_sample_segment):
    log_mel = audio_to_log_mel(wav_sample_segment, 256, top_db=80)
    assert log_mel.shape == (256, 513)
    recreated_segment = log_mel_to_audio(log_mel)
    assert len(recreated_segment) == len(wav_sample_segment)
    wavfile.write('wav_recreated_segment.wav', 48000, recreated_segment)
    wavfile.write('wav_sample_segment.wav', 48000, wav_sample_segment)

    log_mel = audio_to_log_mel(mp3_sample_segment, 256, top_db=80)
    assert log_mel.shape == (256, 513)
    recreated_segment = log_mel_to_audio(log_mel)
    assert len(recreated_segment) == len(mp3_sample_segment)
    wavfile.write('mp3_recreated_segment.wav', 48000, recreated_segment)
    wavfile.write('mp3_sample_segment.wav', 48000, mp3_sample_segment)


def test_log_mel_spectrogram_to_image_and_back(wav_sample_mel_spectrogram, mp3_sample_mel_spectrogram):
    image = log_mel_spectrogram_to_image(wav_sample_mel_spectrogram)
    image.save('wav_test.png')
    assert image.size == (513, 256)
    recreated_spectrogram = image_to_log_mel_spectrogram(image)
    recreated_image = log_mel_spectrogram_to_image(recreated_spectrogram)
    recreated_image.save('wav_test2.png')
    assert recreated_spectrogram.shape == wav_sample_mel_spectrogram.shape

    image = log_mel_spectrogram_to_image(mp3_sample_mel_spectrogram)
    image.save('mp3_test.png')
    assert image.size == (513, 256)
    recreated_spectrogram = image_to_log_mel_spectrogram(image)
    recreated_image = log_mel_spectrogram_to_image(recreated_spectrogram)
    recreated_image.save('mp3_test2.png')
    assert recreated_spectrogram.shape == mp3_sample_mel_spectrogram.shape


def test_audio_to_image_and_back(wav_sample_segment, mp3_sample_segment):
    image = audio_to_image(wav_sample_segment, 256)
    image.save("wav_sample_image.png")
    reloaded_image = Image.open("wav_sample_image.png")

    log_mel = image_to_log_mel_spectrogram(reloaded_image)
    two_step_audio = log_mel_to_audio(log_mel)
    wavfile.write('wav_sample_segment_full_two_step.wav', 48000, two_step_audio)

    loaded_audio = image_to_audio(reloaded_image)
    wavfile.write('wav_sample_segment_full_one_step.wav', 48000, loaded_audio)

