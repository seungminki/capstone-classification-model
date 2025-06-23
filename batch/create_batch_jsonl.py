from batch.prompt import PROMPT
from settings import (
    OPENAI_MODEL_NAME,
    OPENAI_TEMPERATURE,
    OPENAI_BATCH_INPUT_PATH,
)

from tqdm import tqdm
import json
import os


def generate_jsonl_batches(df, prompt_batch_size=1, jsonl_batch_size=6000):
    total_num = len(df)
    # print(f"총 {total}개 데이터 처리 시작 (batch size {batch_size})")

    jsonl_data = []
    batch_counter = 1

    for i in tqdm(range(0, total_num, prompt_batch_size)):
        df_batch = df.iloc[i : i + prompt_batch_size]
        payload = make_batch_payload(df_batch, batch_counter)
        jsonl_data.append(payload)  # dict 형태로 리스트에 저장
        batch_counter += 1

    split_jsonl_batches(jsonl_data, jsonl_batch_size)


def build_prompt(text_list):
    prompt = PROMPT
    for i, t in enumerate(text_list, 1):
        prompt += f"{i}. {t}\n"
    # prompt += PROMPT2
    return prompt


def make_batch_payload(df_batch, batch_index):
    data = df_batch.to_dict(orient="records")
    prompt = build_prompt(data)

    payload = {
        "custom_id": str(batch_index),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": OPENAI_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "너는 커뮤니티 게시글을 분석해서 적절한 태그를 자동으로 추천하는 AI야.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": OPENAI_TEMPERATURE,
        },
    }
    return payload


def split_jsonl_batches(jsonl_data: list, jsonl_batch_size: int):
    print("jsonl_data length: ", len(jsonl_data))

    num_batches = (len(jsonl_data) + jsonl_batch_size - 1) // jsonl_batch_size

    for i in range(num_batches):
        batch_lines = jsonl_data[i * jsonl_batch_size : (i + 1) * jsonl_batch_size]
        batch_filename = os.path.join(
            OPENAI_BATCH_INPUT_PATH, f"batch_part_{i+1}.jsonl"
        )

        os.makedirs(os.path.dirname(batch_filename), exist_ok=True)

        with open(batch_filename, "w", encoding="utf-8") as f_out:
            for line in batch_lines:
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"✅ Saved {batch_filename} ({len(batch_lines)}개 요청)")
