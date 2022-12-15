use pyo3::prelude::*;
use reqwest::header::{CONTENT_LENGTH, RANGE};
use std::io::SeekFrom;
use std::sync::Arc;

use tokio::io::AsyncSeekExt;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn download(url: String, filename: String, max_files: usize, chunk_size: usize) -> PyResult<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { download_async(url, filename, max_files, chunk_size).await })
}

async fn download_async(
    url: String,
    filename: String,
    max_files: usize,
    chunk_size: usize,
) -> PyResult<()> {
    let start = std::time::Instant::now();
    let client = reqwest::Client::new();
    let response = client.head(&url).send().await.unwrap();
    let length = response
        .headers()
        .get(CONTENT_LENGTH)
        .unwrap()
        .to_str()
        .unwrap();
    let length: usize = length.parse().unwrap();

    let mut handles = vec![];
    let semaphore = Arc::new(Semaphore::new(max_files));

    let chunk_size = chunk_size;
    for start in (0..length).step_by(chunk_size) {
        let url = url.clone();
        let filename = filename.clone();
        let client = client.clone();

        let stop = std::cmp::min(start + chunk_size - 1, length);
        let range = format!("bytes={start}-{stop}");
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        handles.push(tokio::spawn(async move {
            // Process each socket concurrently.
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .open(filename)
                .await
                .unwrap();
            file.seek(SeekFrom::Start(start as u64)).await.unwrap();
            let response = client.get(url).header(RANGE, range).send().await.unwrap();
            let content = response.bytes().await.unwrap();
            file.write_all(&content).await.unwrap();
            drop(permit);
        }));
    }
    futures::future::join_all(handles).await;
    let size = length as f64 / 1024.0 / 1024.0;
    let speed = size / start.elapsed().as_secs_f64();
    println!(
        "Took {:?} for {:.2}Mo ({:.2} Mo/s)",
        start.elapsed(),
        size,
        speed
    );
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_dl_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download, m)?)?;
    Ok(())
}
