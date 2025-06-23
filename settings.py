import os
from dotenv import load_dotenv

load_dotenv()

AWS_S3_BUCKET_NAME = os.getenv("aws_s3_bucket_name")
AWS_S3_KEY_ID = os.getenv("aws_s3_id")
AWS_S3_SECRET_KEY = os.getenv("aws_s3_key")

S3_FILE_PATH = "raw_output.json"

OPENAI_TOKEN = os.getenv("openai_token")

OPENAI_BATCH_INPUT_PATH = "input_batches"
OPENAI_BATCH_OUTPUT_PATH = "output_batches"

OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.3

# 50문장씩 300라인 = 1 배치에 15000문장
prompt_batch_size = 50
jsonl_batch_size = 300

BATCHES_FILE_PATH = "output.csv"

ST_MODEL_NAME = "jhgan/ko-sroberta-multitask"

TRAINED_MODEL_LOCAL_PATH = os.getenv("aws_s3_model_path")
TRAINED_MODEL_S3_PATH = os.getenv("aws_s3_model_path")

SAVE_MODEL_DIR = "saved_model"
