from soynlp.normalizer import repeat_normalize, emoticon_normalize

from utils.s3_util import download_file_from_s3

import re
import pandas as pd

pattern = re.compile(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z\s]")  # 한글, 영어

letter_ko_pattern = re.compile(r"[^가-힣\s]")  # 한글 초성 제거

url_pattern = re.compile(r"https?://\S+")
dial_pattern = re.compile(r"010-\d{4}-\d{4}|041-\d{3}-\d{4}|02-\d{3}-\d{4}")
email_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
space_pattern = re.compile(r"\s+")


def load_data(s3_file_path: str) -> pd.DataFrame:
    download_file_from_s3(
        local_dir="./",
        local_filename=s3_file_path,
        s3_key="raw/raw_output.json",
    )
    return pd.read_json(s3_file_path, lines=True)


def preprocess(df: pd.DataFrame):
    df = df.rename(columns={"post_code": "post_id"})
    df = df.rename(columns={"board_code": "board_id"})
    df = df.dropna(subset=["post_id", "board_id"])

    df["text"] = df["text"].apply(clean)
    df = df[df["text"].str.len() > 3]

    df.reset_index(drop=True)

    return df


def clean(x):
    x = url_pattern.sub("", x)
    x = dial_pattern.sub("", x)
    x = email_pattern.sub("", x)

    x = pattern.sub("", x)
    x = letter_ko_pattern.sub("", x)

    x = space_pattern.sub(" ", x)
    x = x.strip()

    x = emoticon_normalize(x, num_repeats=2)
    x = repeat_normalize(x, num_repeats=1)

    return x
