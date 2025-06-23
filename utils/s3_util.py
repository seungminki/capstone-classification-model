import os
import boto3

from settings import AWS_S3_BUCKET_NAME, AWS_S3_KEY_ID, AWS_S3_SECRET_KEY


def download_file_from_s3(local_dir, local_filename, s3_key):

    local_path = os.path.join(local_dir, local_filename)

    if os.path.exists(local_path):
        print(f"already exists in '{local_path}'")
        return

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_S3_KEY_ID,
        aws_secret_access_key=AWS_S3_SECRET_KEY,
        region_name="ap-northeast-2",
    )

    try:
        print(f"Downloading {s3_key} to {local_path} ...")
        s3.download_file(AWS_S3_BUCKET_NAME, s3_key, local_path)

    except Exception as e:
        print(f"Failed to download: {e}")
