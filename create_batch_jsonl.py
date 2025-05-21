import json
import os
import pandas as pd
import time
from tqdm import tqdm
from prompt import PROMPT1, PROMPT2

model_name = "gpt-4.1-mini"
temperature = 0.1  # 0.3랑 비교했을 때 더 나은 결과 예상


def build_prompt(text_list):
    prompt = PROMPT1 + "\n\n아래는 게시글 리스트이다:\n"
    for i, t in enumerate(text_list, 1):
        prompt += f"{i}. {t}\n"
    prompt += PROMPT2
    return prompt


def make_batch_payload(
    df_batch, batch_index, model_name=model_name, temperature=temperature
):
    data = df_batch.to_dict(orient="records")
    prompt = build_prompt(data)

    payload = {
        "custom_id": str(batch_index),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "너는 대학 커뮤니티 게시글을 분류하는 AI야.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        },
    }
    return payload


def generate_jsonl_batches(input_df, output_jsonl_path, batch_size=50):
    total = len(input_df)
    print(f"총 {total}개 데이터 처리 시작 (batch size {batch_size})")

    batch_counter = 1

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, total, batch_size)):
            df_batch = input_df.iloc[i : i + batch_size]
            payload = make_batch_payload(df_batch, batch_counter)
            json_line = json.dumps(payload, ensure_ascii=False)
            f.write(json_line + "\n")
            batch_counter += 1

    print(f"✅ JSONL 저장 완료: {output_jsonl_path}")


if __name__ == "__main__":
    input_path = "json/output.json"
    prefix_dir = "jsonl/gpt4.1-mini/250427"  # gpt4o-mini, gpt4.1-mini

    output_jsonl_path = f"{prefix_dir}/batch_payload.jsonl"
    prompt_batch_size = 50

    # batch용 jsonl 생성
    df = pd.read_json(input_path, lines=True)
    generate_jsonl_batches(df, output_jsonl_path, batch_size=prompt_batch_size)

    # jsonl 분할
    output_dir = f"{prefix_dir}/split_batches"
    batch_size = 300

    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"총 {total}개 요청 읽음")

    num_batches = (total + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_lines = lines[i * batch_size : (i + 1) * batch_size]
        batch_filename = os.path.join(output_dir, f"batch_part_{i+1}.jsonl")

        os.makedirs(os.path.dirname(batch_filename), exist_ok=True)

        with open(batch_filename, "w", encoding="utf-8") as f_out:
            f_out.writelines(batch_lines)

        print(f"✅ Saved {batch_filename} ({len(batch_lines)}개 요청)")
