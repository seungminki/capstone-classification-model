# capstone-classification-model

커뮤니티 게시글에 대해 자동으로 주제를 태깅하는 기능을 제공합니다.  예: `수업`, `친목`, `고민상담`, `중고거래` 등

- **유연한 태그 설정**  
  사용자가 원하는 주제 목록과 태그 개수를 동적으로 정의할 수 있습니다.

- **지식 증류 기반 학습 (Knowledge Distillation)**  
  GPT 등 대규모 언어 모델의 예측 결과를 소형 모델이 모방하도록 학습시켜, 추론 비용을 절감하면서도 높은 성능을 유지합니다.

- **OpenAI Batches 활용**  
  태깅 기준이 바뀔 때마다 GPT로부터 새로운 예측 결과를 받아야 하기 때문에, OpenAI Batches API를 활용하여 전체 API 사용 비용을 약 50% 절감했습니다.

이 프로젝트는 **확장성**과 **비용 효율성**을 모두 고려한 **텍스트 분류 자동화 파이프라인** 구축을 목표로 합니다.


## How to Use
https://platform.openai.com/docs/guides/batch


## Directory structure
```
├── train.py
├── predict.py
├── batch # OpenAI Batches API 사용을 위한 모듈
│   ├── batch.ipynb # 예제 노트북
│   ├── create_batch_jsonl.py # 입력 데이터를 토큰 길이에 맞게 분할하여 .jsonl 파일로 생성
│   ├── postprocess_batch_jsonl.py # Batches API 출력 결과를 원본 데이터와 매핑
│   └── prompt.py
├── model
│   └── bert.py
├── trained_model # 학습된 모델 및 관련 파일 저장 디렉토리
│   ├── config.json
│   ├── labels.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── utils
│   └── s3_util.py
├── preprocess.py
├── settings.py
├── .gitattributes
├── .gitignore
└── README.md