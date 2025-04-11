import pytest
import os
import tempfile
import random
from hf_transfer import download, multipart_upload


@pytest.mark.skip(reason="no way of currently testing this")
def test_download_basic():
    """Test basic download functionality"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Test with a small file from a public URL
        url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/depth2img/two_cats.png"
        download(url, tmp_path, max_files=4, chunk_size=1024)

        # Verify file was downloaded
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_download_invalid_url():
    """Test download with invalid URL"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with pytest.raises(Exception) as exc_info:
            download("http://invalid-url", tmp_path, max_files=4, chunk_size=1024)
        assert "invalid-url" in str(exc_info.value)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_download_invalid_chunk_size():
    """Test download with invalid chunk size"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with pytest.raises(Exception) as exc_info:
            download("https://example.com", tmp_path, max_files=4, chunk_size=0)
        assert "`chunk_size` needs to be positive" in str(exc_info.value)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_download_invalid_parallel_files():
    """Test download with invalid parallel files count"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with pytest.raises(Exception) as exc_info:
            download("https://example.com", tmp_path, max_files=0, chunk_size=1024)
        assert "`max_files` needs to be positive" in str(exc_info.value)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.skip(reason="no way of currently testing this")
def test_upload_basic():
    """Test basic upload functionality"""
    # Create a test file with some random content
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        # Write 1MB of random data
        tmp.write(random.randbytes(1024 * 1024))

    try:
        # Create mock upload URLs (in a real test, these would be actual upload URLs)
        part_urls = [
            "https://huggingface.co/upload/part1",
            "https://huggingface.co/upload/part2",
        ]

        # Test upload with 512KB chunks
        results = multipart_upload(
            file_path=tmp_path,
            parts_urls=part_urls,
            chunk_size=512 * 1024,  # 512KB
            max_files=2,
            parallel_failures=1,
            max_retries=3,
        )

        # Verify we got results for each part
        assert len(results) == len(part_urls)
        # Each result should be a dictionary
        for result in results:
            assert isinstance(result, dict)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_upload_invalid_file():
    """Test upload with non-existent file"""
    with pytest.raises(Exception) as exc_info:
        multipart_upload(
            file_path="/nonexistent/file",
            parts_urls=["https://huggingface.co/upload/part1"],
            chunk_size=1024,
            max_files=1,
        )
    assert "No such file or directory" in str(exc_info.value)


def test_upload_invalid_chunk_size():
    """Test upload with invalid chunk size"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"test")

    try:
        with pytest.raises(Exception) as exc_info:
            multipart_upload(
                file_path=tmp_path,
                parts_urls=["https://example.com/upload/part1"],
                chunk_size=0,  # Invalid chunk size
                max_files=1,
            )
        assert "`chunk_size` needs to be positive" in str(exc_info.value)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_upload_invalid_parallel_files():
    """Test upload with invalid parallel files count"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"test")

    try:
        with pytest.raises(Exception) as exc_info:
            multipart_upload(
                file_path=tmp_path,
                parts_urls=["https://example.com/upload/part1"],
                chunk_size=1024,
                max_files=0,  # Invalid max files
            )
        assert "`max_files` needs to be positive" in str(exc_info.value)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
