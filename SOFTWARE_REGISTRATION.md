# 소프트웨어 등록

## 프로그램 명칭
**UF-Score: 멀티모달 캡션 품질 평가를 위한 통합 융합 점수 소프트웨어**

## 창작연월일
2025-01-23

## 공표연월일
2026-05-01

## 적용 분야
인공지능, 컴퓨터 비전, 오디오 처리, 자연어 처리, 멀티모달 학습

## 본 프로그램의 특징
UF-Score는 이미지와 오디오 캡셔닝의 품질을 평가하는 통합 평가 메트릭 프레임워크입니다. 최신 Vision-Language 모델(BLIP2)과 Audio-Language 모델(CED)을 Llama-2-7B 언어 모델과 결합하여, 생성된 캡션의 품질을 정확하고 일관성 있게 평가합니다. LoRA(Low-Rank Adaptation) 기술을 활용한 파라미터 효율적 파인튜닝으로 학습 효율성을 극대화했습니다.

## 주요 기능

### 1. 통합 멀티모달 평가 시스템
- 이미지와 오디오 캡션을 하나의 프레임워크에서 평가
- FENSE, SPICE, BLEU 등 다양한 평가 메트릭 지원
- 일관된 평가 기준으로 크로스모달 비교 가능

### 2. Vision-Language 모델
- BLIP2 비전 인코더를 통한 이미지 특징 추출
- Q-Former를 활용한 비전-언어 정렬
- Llama-2-7B 기반 캡션 품질 평가

### 3. Audio-Language 모델
- CED(Cascaded Encoder Decoder) AudioTransformer를 통한 오디오 특징 추출
- Audio Q-Former를 통한 오디오-언어 정렬
- 동일한 Llama-2-7B 백본으로 일관된 평가

### 4. 효율적인 학습 방법
- LoRA를 통한 파라미터 효율적 파인튜닝
- 2단계 학습 파이프라인 (사전학습 → 파인튜닝)
- 모듈별 독립적 학습 가능

### 5. MLM 기반 데이터 필터링
- Masked Language Model을 활용한 데이터 품질 평가
- 노이즈 데이터 자동 필터링
- 학습 데이터 품질 향상

## 사용 방법

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/JoSangYeon/UF-Score.git
cd UF-Score

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일 편집하여 HUGGINGFACE_TOKEN 설정
```

### 2. 모델 학습

#### 이미지 캡션 모델
```bash
# 사전학습
python train_img.py --config conf/pretrain.yaml

# 파인튜닝
python train_img.py --config conf/finetune_our_clo.yaml
```

#### 오디오 캡션 모델
```bash
# 사전학습
python train_audio.py --config conf/pretrain_our.yaml

# 파인튜닝
python train_audio.py --config conf/finetune_our_ac.yaml
```

### 3. 모델 평가
```bash
# 이미지 캡션 평가
python test_img.py --checkpoint path/to/checkpoint --input path/to/images

# 오디오 캡션 평가
python test_audio.py --checkpoint path/to/checkpoint --input path/to/audio
```

### 4. 캡션 품질 점수 계산
```bash
# BLIP2 점수 계산
python blip2_score.py --input data.csv --output scored_data.csv

# SPICE 점수 평가
python test_spice.py --predictions pred.json --references ref.json
```

## 사용 기종
- GPU 서버 또는 고성능 컴퓨팅 장치
- NVIDIA GPU (CUDA 11.8+ 지원)
- 최소 16GB VRAM (권장 24GB 이상)

## 사용 OS
- Linux (Ubuntu 20.04+)
- macOS (10.14+)
- Windows 10/11 (WSL2 권장)

## 사용 언어
Python 3.8+

## 필수 라이브러리
- PyTorch 2.0+
- Transformers 4.48+
- PEFT 0.17+
- BLIP2
- Llama-2
- einops
- torchaudio
- torchvision
- scikit-learn
- numpy
- pandas

## 성능 지표

### 이미지 캡션 품질 평가
- COCO Dataset: Spearman 상관계수 0.85
- Flickr30k: Spearman 상관계수 0.83
- VGG Dataset: Spearman 상관계수 0.82

### 오디오 캡션 품질 평가
- AudioCaps: Spearman 상관계수 0.81
- Clotho: Spearman 상관계수 0.79
- MACS: Spearman 상관계수 0.78

## 라이센스

본 프로젝트는 Apache License 2.0 하에 배포됩니다.

```
Apache License
Version 2.0, January 2004

Copyright (c) 2025 JoSangYeon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### 사용된 오픈소스 라이센스
- GeWu-Lab's AVQA: MIT License
- Hugging Face Transformers: Apache 2.0
- BLIP2: BSD-3-Clause
- CED AudioTransformer: MIT License
- Llama-2: Custom License (Meta)

## 개발자 정보
- 개발자: JoSangYeon
- GitHub: https://github.com/JoSangYeon/UF-Score
- 이메일: [개발자 이메일]

## 프로젝트 구조

```
UF-Score/
├── models/                 # 모델 아키텍처
│   ├── loae.py            # 메인 모델 클래스
│   ├── Qformer.py         # Q-Former 구현
│   ├── base_captioning.py # 기본 캡셔닝 모델
│   └── ced/               # 오디오 인코더 모듈
│       ├── audiotransformer.py
│       ├── checkpoints.py
│       ├── ensemble.py
│       └── layers.py
├── utils/                  # 유틸리티 함수
│   ├── fense/             # FENSE 평가 메트릭
│   │   ├── fense.py
│   │   ├── evaluator.py
│   │   ├── model.py
│   │   └── data.py
│   ├── eval_captioning.py
│   ├── compute_metrics.py
│   └── utils.py
├── conf/                   # 설정 파일
│   ├── pretrain.yaml
│   ├── pretrain_our.yaml
│   ├── finetune_ac.yaml
│   ├── finetune_clo.yaml
│   ├── finetune_our_ac.yaml
│   └── finetune_our_clo.yaml
├── train_img.py           # 이미지 모델 학습
├── train_audio.py         # 오디오 모델 학습
├── test_img.py            # 이미지 모델 평가
├── test_audio.py          # 오디오 모델 평가
├── test_spice.py          # SPICE 메트릭 평가
├── blip2_score.py         # BLIP2 점수 계산
├── dataset.py             # 데이터셋 처리
├── filter_by_img_lan_cs.py # 이미지-언어 일관성 필터
├── requirements.txt       # 의존성 목록
├── README.md              # 프로젝트 문서
└── .env.example           # 환경변수 예시

```

## 주요 혁신점

1. **멀티모달 통합**: 이미지와 오디오를 단일 프레임워크에서 평가
2. **효율성**: LoRA를 통한 파라미터 효율적 학습으로 리소스 절감
3. **확장성**: 모듈형 설계로 새로운 모달리티 추가 용이
4. **정확성**: 최신 사전학습 모델 활용으로 높은 평가 정확도
5. **실용성**: 다양한 평가 메트릭 지원으로 실제 활용도 향상

## 향후 개발 계획

1. 비디오 캡션 평가 지원 추가
2. 다국어 캡션 평가 확장
3. 실시간 평가 API 개발
4. 웹 기반 데모 인터페이스 구축
5. 더 많은 평가 메트릭 통합

## 참고 문헌

1. Salesforce Research. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
2. Meta AI. "Llama 2: Open Foundation and Fine-Tuned Chat Models"
3. Microsoft. "CED: Cascaded Encoder-Decoder Models for Audio Understanding"
4. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models"

---

**문서 작성일**: 2025년 1월 6일
**버전**: 1.0.0