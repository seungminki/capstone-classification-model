import os
import json
import re
from typing import List, Dict, Any
from preprocess import load_data, preprocess
from settings import S3_FILE_PATH
import pandas as pd

batch_output_dir_path = "./output_batches"


def list_files_in_directory(dir_path: str):
    return [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]


def concat_jsonl_files(file_list: List[str]) -> List[Dict[str, Any]]:
    results = []
    for file_name in file_list:
        with open(
            f"{batch_output_dir_path}/{file_name}", "r", encoding="utf-8"
        ) as file:
            for line in file:
                json_object = json.loads(line.strip())
                parsed_json = parsing_rules(file_name, json_object)
                results.append(parsed_json)

    return results


def parsing_rules(file_name: str, res: dict):
    task_id = res["custom_id"]
    try:
        result_content = res["response"]["body"]["choices"][0]["message"]["content"]
        if result_content.startswith("```json"):
            result_content = re.sub(r"^```json\n?|\n?```$", "", result_content.strip())

        result_content = re.sub(r",\s*\]$", "]", result_content.strip())

        parsed_data = json.loads(result_content)

    except Exception as e:
        # print("====== raw content start ======")
        # print(result_content)
        # print("====== raw content end ======")
        print(f"⚠️ 파싱 실패 - file: {file_name}, id: {task_id}, error: {e}")

    return parsed_data


def merge_model_output(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    output_df["post_id"] = output_df["post_id"].astype(int)
    output_df["board_id"] = output_df["board_id"].astype(int)

    merged_df = pd.merge(input_df, output_df, on=["post_id", "board_id"], how="left")
    # nan_tag_rows = merged_df[merged_df["tag"].isna()]

    return merged_df


if __name__ == "__main__":
    output_file_list = list_files_in_directory(batch_output_dir_path)

    model_output = concat_jsonl_files(output_file_list)

    output_df = pd.DataFrame(
        [
            {
                "post_id": item["post_code"],
                "board_id": item["board_code"],
                **item["probs"],
            }
            for item in model_output
        ]
    )

    df = load_data(S3_FILE_PATH)[:30]
    input_df = preprocess(df)

    df = merge_model_output(input_df, output_df)
