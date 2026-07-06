# Copyright 2023-2024 Xiaomi Corporation and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import random
from re import sub
from PIL import Image

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def aud_collate_fn(examples):
    feats = [x[0] for x in examples]
    labels = [x[1] for x in examples]
    names = [x[2] for x in examples]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    return {"audios": padded_feats, "text": labels, "names": names}
    # return {"audios": padded_feats.bfloat16(), "text": labels, "names": names} # add ".bfloat16()" by jsy
    
def img_collate_fn(examples):
    images = torch.stack([x[0] for x in examples], dim=0)
    labels = [x[1] for x in examples]
    names = [x[2] for x in examples]
    return {"images": images, "text": labels, "names": names}
    # return {"audios": padded_feats.bfloat16(), "text": labels, "names": names} # add ".bfloat16()" by jsy



import torch
import random
import librosa

def add_white_noise(data, sr=16000, rate=0.005):
    # torch.randn()을 사용하여 white noise 추가
    wn = torch.randn(data.size(), device=data.device)  # 같은 장치에서 작업
    data_wn = data + rate * wn
    return data_wn
def shift_data(data, sr=16000, rate=0.1):
    # torch.roll()을 사용하여 데이터 이동
    data_roll = torch.roll(data, int(len(data) * rate))
    return data_roll
def stretch_data(data, sr=16000, rate=0.75):
    # librosa를 사용한 타임 스트레칭 (리턴은 numpy 배열이므로 torch로 변환)
    data_np = data.cpu().numpy()  # numpy로 변환
    stretched_data = librosa.effects.time_stretch(data_np, rate=rate)
    return torch.tensor(stretched_data, device=data.device)  # 다시 torch 텐서로 변환
def minus_sound(data, sr=16000, rate=0):
    # 음성 데이터를 반전 (마이너스 곱하기)
    minus_data = (-1) * data
    return minus_data
def audio_augment(data, sr=16000, noise_rate=0.005,
                  shift_rate=0.1, stretch_rate=0.85,
                  is_shuffle=True, apply_rate=0.5):
    aug_method_list = [(add_white_noise, noise_rate), 
                       (shift_data, shift_rate),
                       (stretch_data, stretch_rate), 
                       (minus_sound, 0.0)]
    if is_shuffle:
        random.shuffle(aug_method_list)
    
    for method, rate in aug_method_list:
        is_apply = random.random()
        if is_apply < apply_rate:
            data = method(data, sr, rate)
    return data

def handle_wav(wav_file, target_rate, max_sample_length, is_aug=True):
    """
    handle one wav file.
    Return:
        waveform: Tensor(1D)
    """
    waveform, sample_rate = torchaudio.load(wav_file)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_rate
        )(waveform)
    waveform = waveform[0]  # just get one channel data
    waveform = audio_augment(waveform) if is_aug else waveform
    # if audio length is longer than max_length_sample, we randomly crop it to max length
    if waveform.shape[-1] > max_sample_length:
        max_start = waveform.shape[-1] - max_sample_length
        start = random.randint(0, max_start)
        waveform = waveform[start : start + max_sample_length]
    return waveform


def _text_preprocess(sentence):
    sentence = sentence.lower()
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")
    sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
    sentence = sentence.strip() # added by jsy
    return sentence

def text_postprocess(text):
    unwanted_tokens_pattern = r'</s>|<unk>'
    cleaned_text = sub(unwanted_tokens_pattern, '', text)
    cleaned_text = sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


class AudioDataset(Dataset):
    def __init__(self, data_file, sample_rate=16000, max_length=10, is_aug=False):
        super().__init__()
        self.lists = []
        with open(data_file, "r", encoding="utf8") as fin:
            for line in fin:
                self.lists.append(line)

        self.all_data = []
        for line in self.lists:
            obj = json.loads(line)
            self.all_data.append(obj)

        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_length_sample = self.max_length * self.sample_rate
        self.is_aug = is_aug

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        obj = self.all_data[index]
        key = obj["key"]
        wav_file = obj["wav"]
        caption = _text_preprocess(obj["label"])
        waveform = handle_wav(
            wav_file,
            target_rate=self.sample_rate,
            max_sample_length=self.max_length_sample,
            is_aug=self.is_aug
        )
        return waveform, caption, key


class ImageDataset(Dataset):
    def __init__(self, data_file, img_processor,):
        super().__init__()
        self.lists = []
        with open(data_file, "r", encoding="utf8") as fin:
            for line in fin:
                self.lists.append(line)

        self.all_data = []
        for line in self.lists:
            obj = json.loads(line)
            self.all_data.append(obj)

        self.img_processor = img_processor

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        obj = self.all_data[index]
        key = obj["key"]
        img_file = obj["img"]

        caption = _text_preprocess(obj["label"])
        image = self.img_processor(images=Image.open(img_file).convert('RGB'), return_tensors="pt")
        image = image.pixel_values[0]
        return image, caption, key
    