use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use reqwest::header::{CONTENT_RANGE, RANGE};
use std::io::SeekFrom;
use std::sync::Arc;

use tokio::io::AsyncSeekExt;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

#[pyfunction]
fn download(url: String, filename: String, max_files: usize, chunk_size: usize) -> PyResult<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async { download_async(url, filename, max_files, chunk_size).await })
}

async fn download_async(
    url: String,
    filename: String,
    max_files: usize,
    chunk_size: usize,
) -> PyResult<()> {
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header(RANGE, "bytes=0-0".to_string())
        .send()
        .await
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;
    let content_range = response
        .headers()
        .get(CONTENT_RANGE)
        .ok_or(PyException::new_err("No content length"))?
        .to_str()
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;

    let size: Vec<&str> = content_range.split('/').collect();
    // Content-Range: bytes 0-0/702517648
    // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Range
    let length: usize = size
        .last()
        .ok_or(PyException::new_err(
            "Error while downloading: No size was detected",
        ))?
        .parse()
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;

    let mut handles = vec![];
    let semaphore = Arc::new(Semaphore::new(max_files));

    let chunk_size = chunk_size;
    for start in (0..length).step_by(chunk_size) {
        let url = url.clone();
        let filename = filename.clone();
        let client = client.clone();

        let stop = std::cmp::min(start + chunk_size - 1, length);
        let permit =
            semaphore.clone().acquire_owned().await.map_err(|err| {
                PyException::new_err(format!("Error while downloading: {err:?}"))
            })?;
        handles.push(tokio::spawn(async move {
            let chunk = download_chunk(client, url, filename, start, stop).await;
            drop(permit);
            chunk
        }));
    }

    // Output the chained result
    let results: Vec<Result<PyResult<()>, tokio::task::JoinError>> =
        futures::future::join_all(handles).await;
    let results: PyResult<()> = results.into_iter().flatten().collect();
    results?;
    Ok(())
}

async fn download_chunk(
    client: reqwest::Client,
    url: String,
    filename: String,
    start: usize,
    stop: usize,
) -> PyResult<()> {
    // Process each socket concurrently.
    let range = format!("bytes={start}-{stop}");
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(filename)
        .await
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;
    file.seek(SeekFrom::Start(start as u64))
        .await
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;
    let response = client
        .get(url)
        .header(RANGE, range)
        .send()
        .await
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;
    let content = response
        .bytes()
        .await
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;
    file.write_all(&content)
        .await
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn hf_transfer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download, m)?)?;
    Ok(())
}
