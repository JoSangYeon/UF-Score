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

import argparse
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from transformers import Blip2Processor

from dataset import ImageDataset, img_collate_fn
from models.loae import VisionLlama7BCaptionModel

import warnings
warnings.filterwarnings(action='ignore')


def _exec_model_test(
    model_test, data_loader, res_dir, use_nucleus_sampling=False, beam_size=3, top_p=0.9
):
    model_test.eval()
    predict_result = {}
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            output = model_test.generate(
                samples=batch_data,
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=beam_size,
                top_p=top_p,
            )

            audio_names = batch_data["names"]
            for j in range(len(audio_names)):
                predict_result[audio_names[j]] = output[j].replace("\n", " ")

    if use_nucleus_sampling:
        result_file = os.path.join(res_dir, "top-p_{}.txt".format(top_p))
    else:
        result_file = os.path.join(res_dir, "beam_{}.txt".format(beam_size))
    with open(result_file, "w+", encoding="utf8") as writer:
        for key in sorted(predict_result.keys()):
            writer.write(key + "\t" + predict_result[key] + "\n")
    return result_file


def do_beam_search():
    for beam in [3,]:
        logging.info(
            "Start test for {}, beam {}, samples {}, total batch:{}".format(
                args.test_data, beam, len(test_dataset), len(test_dataloader)
            )
        )
        _exec_model_test(
            model,
            test_dataloader,
            args.result_dir,
            use_nucleus_sampling=False,
            beam_size=beam,
        )


def do_nucleus_sampling():
    for p in [0.9,]:
        logging.info(
            "Start test for {}, top_p {}, samples {}, total batch:{}".format(
                args.test_data, p, len(test_dataset), len(test_dataloader)
            )
        )
        _exec_model_test(
            model, test_dataloader, args.result_dir, use_nucleus_sampling=True, top_p=p
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test with your special model")
    parser.add_argument("--config_path", required=True, help="config file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint file")
    parser.add_argument("--test_data", required=True, help="test data file")
    parser.add_argument("--result_dir", required=True, help="asr result file")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(args)

    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = VisionLlama7BCaptionModel(config)
    logging.info(model)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logging.info("Checkpoint: loading from checkpoint %s for GPU" % args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
    else:
        logging.info("Checkpoint: loading from checkpoint %s for CPU" % args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    logging.info("start model recognise in {}".format(device))

    sample_rate, max_length, batch_size = (
        config["dataset_conf"]["sample_rate"],
        config["dataset_conf"]["max_len"],
        config["dataset_conf"]["batch_size"],
    )

    img_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    test_dataset = ImageDataset(
        args.test_data, img_processor,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=img_collate_fn,
        drop_last=False,
    )

    do_beam_search()
    do_nucleus_sampling()

"""
CUDA_VISIBLE_DEVICES=0 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/img/valor32k_full/checkpoint-2180/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/img/valor32k_full/metric/2180

CUDA_VISIBLE_DEVICES=1 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/img/valor32k_full/checkpoint-2420/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/img/valor32k_full/metric/2420

    

    
CUDA_VISIBLE_DEVICES=2 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/img/valor32k_090/checkpoint-1962/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/img/valor32k_090/metric/1962

CUDA_VISIBLE_DEVICES=3 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/img/valor32k_090/checkpoint-2180/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/img/valor32k_090/metric/2180



    
CUDA_VISIBLE_DEVICES=2 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/img/valor32k_075/checkpoint-5278/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/img/valor32k_075/metric/5278

CUDA_VISIBLE_DEVICES=3 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/img/valor32k_075/checkpoint-5460/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/img/valor32k_075/metric/5460


    

CUDA_VISIBLE_DEVICES=2 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/img/valor32k_050/checkpoint-7032/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/img/valor32k_050/metric/7032

CUDA_VISIBLE_DEVICES=3 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/img/valor32k_050/checkpoint-7260/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/img/valor32k_050/metric/7260

"""


"""
CUDA_VISIBLE_DEVICES=0 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/clip_img/valor32k_050/checkpoint-1210/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/clip_img/valor32k_050/metric/1210

CUDA_VISIBLE_DEVICES=1 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/clip_img/valor32k_075/checkpoint-1810/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/clip_img/valor32k_075/metric/1810

CUDA_VISIBLE_DEVICES=2 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/clip_img/valor32k_090/checkpoint-2180/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/clip_img/valor32k_090/metric/2180



"""


"""Test: Valor 32K MLM Filter

[v] CUDA_VISIBLE_DEVICES=7 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/mlm_filter_img/vs_train_90/checkpoint-2180/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/mlm_filter_img/vs_train_90/metric/2180

[v] CUDA_VISIBLE_DEVICES=7 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/mlm_filter_img/image_text_matching/va_train_75/checkpoint-1810/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/mlm_filter_img/image_text_matching/va_train_75/metric/1810

[v] CUDA_VISIBLE_DEVICES=6 python test_img.py --config_path conf/finetune_clo.yaml \
    --checkpoint result/mlm_filter_img/image_text_matching/va_train_50/checkpoint-1210/pytorch_model.bin \
    --test_data data/va_test.data \
    --result_dir result/mlm_filter_img/image_text_matching/va_train_50/metric/1210


"""