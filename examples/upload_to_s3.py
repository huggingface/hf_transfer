# pip install boto3 hf_transfer

import boto3
from hf_transfer import multipart_upload
from math import ceil
import os
from time import time

# 10 MiB
CHUNK_SIZE = 10_485_760

s3 = boto3.client("s3")

bucket = "test-hf-transfer-multi-part-upload"
bucket_key = "some_file"

upload = s3.create_multipart_upload(
    ACL="bucket-owner-full-control",
    Bucket=bucket,
    Key=bucket_key,
)
upload_id = upload["UploadId"]
print("created multipart upload")

file_name = "some_file"
file_size = os.stat(file_name).st_size

urls = []
nb_parts = ceil(file_size / CHUNK_SIZE)
for part_number in range(1, nb_parts + 1):
    params = {
        "Bucket": bucket,
        "Key": bucket_key,
        "PartNumber": part_number,
        "UploadId": upload_id,
    }
    urls.append(
        s3.generate_presigned_url(
            ClientMethod="upload_part", Params=params, ExpiresIn=86400
        )
    )
print("prepared parts urls")

print("uploading parts...")
start = time()
responses = multipart_upload(
    file_path=file_name,
    parts_urls=urls,
    chunk_size=CHUNK_SIZE,
    max_files=64,
    parallel_failures=63,
    max_retries=5,
)
print(f"uploaded parts in {time() - start}")

etag_with_parts = []
for part_number, header in enumerate(responses):
    etag = header.get("etag")
    etag_with_parts.append({"ETag": etag, "PartNumber": part_number + 1})

parts = {"Parts": etag_with_parts}

s3.complete_multipart_upload(
    Bucket=bucket, Key=bucket_key, MultipartUpload=parts, UploadId=upload_id
)
print("upload complete")

