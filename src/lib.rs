use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rand::{thread_rng, Rng};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_LENGTH, CONTENT_RANGE, RANGE};
use std::collections::HashMap;
use std::fs::remove_file;
use std::io::SeekFrom;
use std::path::Path;
use std::sync::Arc;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tokio_util::codec::{BytesCodec, FramedRead};

const BASE_WAIT_TIME: usize = 300;
const MAX_WAIT_TIME: usize = 10_000;

/// max_files: Number of open file handles, which determines the maximum number of parallel downloads
/// parallel_failures:  Number of maximum failures of different chunks in parallel (cannot exceed max_files)
/// max_retries: Number of maximum attempts per chunk. (Retries are exponentially backed off + jitter)
///
/// The number of threads can be tuned by the environment variable `TOKIO_WORKER_THREADS` as documented in
/// https://docs.rs/tokio/latest/tokio/runtime/struct.Builder.html#method.worker_threads
#[pyfunction]
#[pyo3(signature = (url, filename, max_files, chunk_size, parallel_failures=0, max_retries=0, headers=None))]
fn download(
    url: String,
    filename: String,
    max_files: usize,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
    headers: Option<HashMap<String, String>>,
) -> PyResult<()> {
    if parallel_failures > max_files {
        return Err(PyException::new_err(
            "Error parallel_failures cannot be > max_files".to_string(),
        ));
    }
    if (parallel_failures == 0) != (max_retries == 0) {
        return Err(PyException::new_err(
            "For retry mechanism you need to set both `parallel_failures` and `max_retries`"
                .to_string(),
        ));
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            download_async(
                url,
                filename.clone(),
                max_files,
                chunk_size,
                parallel_failures,
                max_retries,
                headers,
            )
            .await
        })
        .map_err(|err| {
            let path = Path::new(&filename);
            if path.exists() {
                match remove_file(filename) {
                    Ok(_) => err,
                    Err(err) => {
                        return PyException::new_err(format!(
                            "Error while removing corrupted file: {err:?}"
                        ));
                    }
                }
            } else {
                err
            }
        })
}

/// parts_urls: Dictionary consisting of part numbers as keys and the associated url as values
/// completion_url: The url that should be called when the upload is finished
/// max_files: Number of open file handles, which determines the maximum number of parallel uploads
/// parallel_failures:  Number of maximum failures of different chunks in parallel (cannot exceed max_files)
/// max_retries: Number of maximum attempts per chunk. (Retries are exponentially backed off + jitter)
///
/// The number of threads can be tuned by the environment variable `TOKIO_WORKER_THREADS` as documented in
/// https://docs.rs/tokio/latest/tokio/runtime/struct.Builder.html#method.worker_threads
///
/// See https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html for more information
/// on the multipart upload
#[pyfunction]
#[pyo3(signature = (file_path, parts_urls, chunk_size, max_files, parallel_failures=0, max_retries=0))]
fn multipart_upload(
    file_path: String,
    parts_urls: Vec<String>,
    chunk_size: u64,
    max_files: usize,
    parallel_failures: usize,
    max_retries: usize,
) -> PyResult<Vec<HashMap<String, String>>> {
    if parallel_failures > max_files {
        return Err(PyException::new_err(
            "Error parallel_failures cannot be > max_files".to_string(),
        ));
    }
    if (parallel_failures == 0) != (max_retries == 0) {
        return Err(PyException::new_err(
            "For retry mechanism you need to set both `parallel_failures` and `max_retries`"
                .to_string(),
        ));
    }

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            upload_async(
                file_path,
                parts_urls,
                chunk_size,
                max_files,
                parallel_failures,
                max_retries,
            )
            .await
        })
}

fn jitter() -> usize {
    thread_rng().gen_range(0..=500)
}

pub fn exponential_backoff(base_wait_time: usize, n: usize, max: usize) -> usize {
    (base_wait_time + n.pow(2) + jitter()).min(max)
}

async fn download_async(
    url: String,
    filename: String,
    max_files: usize,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
    input_headers: Option<HashMap<String, String>>,
) -> PyResult<()> {
    let client = reqwest::Client::new();

    let mut headers = HeaderMap::new();
    if let Some(input_headers) = input_headers {
        for (k, v) in input_headers {
            let k: HeaderName = k
                .try_into()
                .map_err(|err| PyException::new_err(format!("Invalid header: {err:?}")))?;
            let v: HeaderValue = v
                .try_into()
                .map_err(|err| PyException::new_err(format!("Invalid header value: {err:?}")))?;
            headers.insert(k, v);
        }
    };

    let response = client
        .get(&url)
        .headers(headers.clone())
        .header(RANGE, "bytes=0-0")
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
    let parallel_failures_semaphore = Arc::new(Semaphore::new(parallel_failures));

    let chunk_size = chunk_size;
    for start in (0..length).step_by(chunk_size) {
        let url = url.clone();
        let filename = filename.clone();
        let client = client.clone();
        let headers = headers.clone();

        let stop = std::cmp::min(start + chunk_size - 1, length);
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?;
        let parallel_failures_semaphore = parallel_failures_semaphore.clone();
        handles.push(tokio::spawn(async move {
            let mut chunk = download_chunk(&client, &url, &filename, start, stop, headers.clone()).await;
            let mut i = 0;
            if parallel_failures > 0 {
                while let Err(dlerr) = chunk {
                    if i >= max_retries {
                        return Err(PyException::new_err(format!(
                            "Failed after too many retries ({max_retries:?}): {dlerr:?}"
                        )));
                    }
                    let parallel_failure_permit = parallel_failures_semaphore.clone().try_acquire_owned().map_err(|err| {
                        PyException::new_err(format!(
                            "Failed too many failures in parallel ({parallel_failures:?}): {dlerr:?} ({err:?})"
                        ))
                    })?;

                    let wait_time = exponential_backoff(BASE_WAIT_TIME, i, MAX_WAIT_TIME);
                    sleep(tokio::time::Duration::from_millis(wait_time as u64)).await;

                    chunk = download_chunk(&client, &url, &filename, start, stop, headers.clone()).await;
                    i += 1;
                    drop(parallel_failure_permit);
                }
            }
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
    client: &reqwest::Client,
    url: &str,
    filename: &str,
    start: usize,
    stop: usize,
    headers: HeaderMap,
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
        .headers(headers)
        .header(RANGE, range)
        .send()
        .await
        .map_err(|err| PyException::new_err(format!("Error while downloading: {err:?}")))?
        .error_for_status()
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

async fn upload_async(
    file_path: String,
    parts_urls: Vec<String>,
    chunk_size: u64,
    max_files: usize,
    parallel_failures: usize,
    max_retries: usize,
) -> PyResult<Vec<HashMap<String, String>>> {
    let client = reqwest::Client::new();

    let mut handles = vec![];
    let semaphore = Arc::new(Semaphore::new(max_files));
    let parallel_failures_semaphore = Arc::new(Semaphore::new(parallel_failures));

    for (part_number, part_url) in parts_urls.iter().enumerate() {
        let url = part_url.to_string();
        let path = file_path.to_owned();
        let client = client.clone();

        let start = (part_number as u64) * chunk_size;
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|err| PyException::new_err(format!("Error acquiring semaphore: {err}")))?;
        let parallel_failures_semaphore = parallel_failures_semaphore.clone();
        handles.push(tokio::spawn(async move {
                    let mut chunk = upload_chunk(&client, &url, &path, start, chunk_size).await;
                    let mut i = 0;
                    if parallel_failures > 0 {
                        while let Err(ul_err) = chunk {
                            if i >= max_retries {
                                return Err(PyException::new_err(format!(
                                    "Failed after too many retries ({max_retries:?}): {ul_err:?}"
                                )));
                            }

                            let parallel_failure_permit = parallel_failures_semaphore.clone().try_acquire_owned().map_err(|err| {
                                PyException::new_err(format!(
                                    "Failed too many failures in parallel ({parallel_failures:?}): {ul_err:?} ({err:?})"
                                ))
                            })?;

                            let wait_time = exponential_backoff(BASE_WAIT_TIME, i, MAX_WAIT_TIME);
                            sleep(tokio::time::Duration::from_millis(wait_time as u64)).await;

                            chunk = upload_chunk(&client, &url, &path, start, chunk_size).await;
                            i += 1;
                            drop(parallel_failure_permit);
                        }
                    }
                    drop(permit);
                    chunk
                }));
    }

    let results: Vec<Result<PyResult<HashMap<String, String>>, tokio::task::JoinError>> =
        futures::future::join_all(handles).await;

    let results: PyResult<Vec<HashMap<String, String>>> = results
        .into_iter()
        .flat_map(|res| match res {
            Ok(Ok(response)) => Some(Ok(response)),
            Ok(Err(err)) => Some(Err(err)),
            Err(err) => Some(Err(PyException::new_err(format!(
                "Error occurred while uploading: {err}"
            )))),
        })
        .collect();

    results
}

async fn upload_chunk(
    client: &reqwest::Client,
    url: &str,
    path: &str,
    start: u64,
    chunk_size: u64,
) -> PyResult<HashMap<String, String>> {
    let mut options = OpenOptions::new();
    let mut file = options.read(true).open(path).await?;
    let file_size = file.metadata().await?.len();
    let bytes_transfered = std::cmp::min(file_size - start, chunk_size);

    file.seek(SeekFrom::Start(start)).await?;
    let chunk = file.take(chunk_size);

    let response = client
        .put(url)
        .header(CONTENT_LENGTH, bytes_transfered)
        .body(reqwest::Body::wrap_stream(FramedRead::new(
            chunk,
            BytesCodec::new(),
        )))
        .send()
        .await
        .map_err(|err| PyException::new_err(format!("Error sending chunk: {err}")))?
        .error_for_status()
        .map_err(|err| {
            PyException::new_err(format!(
                "Server responded with error status code while upload chunk: {err}"
            ))
        })?;

    let mut headers = HashMap::new();
    for (name, value) in response.headers().into_iter() {
        headers.insert(
            name.to_string(),
            value
                .to_str()
                .map_err(|err| {
                    PyException::new_err(format!("Response header contains non ASCII chars: {err}"))
                })?
                .to_owned(),
        );
    }
    Ok(headers)
}

/// A Python module implemented in Rust.
#[pymodule]
fn hf_transfer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download, m)?)?;
    m.add_function(wrap_pyfunction!(multipart_upload, m)?)?;
    Ok(())
}
