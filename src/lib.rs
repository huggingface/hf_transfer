use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rand::{thread_rng, Rng};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_LENGTH, CONTENT_RANGE, RANGE};
use serde::Serialize;
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

/// parallel_failures:  Number of maximum failures of different chunks in parallel (cannot exceed max_files)
/// max_retries: Number of maximum attempts per chunk. (Retries are exponentially backed off + jitter)
/// number of threads can be tuned by environment variable `TOKIO_WORKER_THREADS` as documented in https://docs.rs/tokio/latest/tokio/runtime/struct.Builder.html#method.worker_threads
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

/// parallel_failures:  Number of maximum failures of different chunks in parallel (cannot exceed max_files)
/// max_retries: Number of maximum attempts per chunk. (Retries are exponentially backed off + jitter)
#[pyfunction]
#[pyo3(signature = (file_path, upload_action, verify_action, upload_info, token, max_files, parallel_failures=0, max_retries=0))]
fn upload(
    file_path: String,
    upload_action: LfsAction,
    verify_action: Option<LfsAction>,
    upload_info: UploadInfo,
    token: Option<String>,
    max_files: usize,
    parallel_failures: usize,
    max_retries: usize,
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
            upload_async(
                file_path,
                upload_action,
                verify_action,
                upload_info,
                token,
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

                    let wait_time = exponential_backoff(300, i, 10_000);
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

#[derive(Clone, Debug, FromPyObject)]
struct LfsAction {
    #[pyo3(item)]
    href: String,
    #[pyo3(item)]
    header: HashMap<String, String>,
}

#[derive(FromPyObject)]
struct UploadInfo {
    #[pyo3(attribute)]
    sha256: Vec<u8>,
    #[pyo3(attribute)]
    size: usize,
}

impl UploadInfo {
    fn get_oid(&self) -> String {
        self.sha256
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<String>>()
            .join("")
    }
}

#[derive(Serialize)]
struct UploadedObject {
    oid: String,
    size: usize,
}

impl From<UploadInfo> for UploadedObject {
    fn from(value: UploadInfo) -> Self {
        Self {
            oid: value.get_oid(),
            size: value.size,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EtagWithPart {
    etag: String,
    part_number: usize,
}

#[derive(Serialize)]
struct CompletionPayload {
    oid: String,
    parts: Vec<EtagWithPart>,
}

impl CompletionPayload {
    fn new(oid: &str, parts: Vec<EtagWithPart>) -> Self {
        Self {
            oid: oid.to_owned(),
            parts,
        }
    }
}

async fn upload_async(
    file_path: String,
    mut upload_action: LfsAction,
    verify_action: Option<LfsAction>,
    upload_info: UploadInfo,
    token: Option<String>,
    max_files: usize,
    parallel_failures: usize,
    max_retries: usize,
) -> PyResult<()> {
    let client = reqwest::Client::new();
    let chunk_size = upload_action.header.remove("chunk_size");
    let mut options = OpenOptions::new();
    let file = options.read(true).open(&file_path).await?;
    match chunk_size {
        Some(chunk_size_header) => {
            let chunk_size = chunk_size_header.parse::<u64>().map_err(|err| {
                PyException::new_err(format!("Could not parse chunk_size header: {err}"))
            })?;

            let mut handles = vec![];
            let semaphore = Arc::new(Semaphore::new(max_files));
            let parallel_failures_semaphore = Arc::new(Semaphore::new(parallel_failures));

            for (part_number, presigned_url) in &upload_action.header {
                let url = presigned_url.to_string();
                let path = file_path.to_owned();
                let client = client.clone();
                let part_number = part_number.parse::<usize>().map_err(|err| {
                    PyException::new_err(format!("Error parsing part number: {err}"))
                })?;

                let start = (part_number as u64 - 1) * chunk_size;
                let permit = semaphore.clone().acquire_owned().await.map_err(|err| {
                    PyException::new_err(format!("Error acquiring semaphore: {err}"))
                })?;
                let parallel_failures_semaphore = parallel_failures_semaphore.clone();
                handles.push(tokio::spawn(async move {
                    let mut chunk = upload_chunk(&client, &url, &path, start, upload_info.size as u64, chunk_size, part_number).await;
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

                            let wait_time = exponential_backoff(300, i, 10_000);
                            sleep(tokio::time::Duration::from_millis(wait_time as u64)).await;

                            chunk = upload_chunk(&client, &url, &path, start, upload_info.size as u64, chunk_size, part_number).await;
                            i += 1;
                            drop(parallel_failure_permit);
                        }
                    }
                    drop(permit);
                    chunk
                }));
            }

            let results: Vec<Result<PyResult<EtagWithPart>, tokio::task::JoinError>> =
                futures::future::join_all(handles).await;

            let results: PyResult<Vec<EtagWithPart>> =
                results
                    .into_iter()
                    .try_fold(vec![], |mut acc, res| match res {
                        Ok(Ok(etag_part)) => {
                            acc.push(etag_part);
                            Ok(acc)
                        }
                        Ok(Err(err)) => Err(err),
                        Err(err) => Err(PyException::new_err(format!(
                            "Error occurred while uploading: {err}"
                        ))),
                    });

            let oid = upload_info.get_oid();
            let mut parts = results?;
            parts.sort_by_key(|p| p.part_number);
            client
                .post(upload_action.href)
                .json(&CompletionPayload::new(&oid, parts))
                .send()
                .await
                .map_err(|err| {
                    PyException::new_err(format!("Error sending completion request: {err}"))
                })?
                .error_for_status()
                .map_err(|err| {
                    PyException::new_err(format!(
                        "Server responded with error status code in completion request: {err}"
                    ))
                })?;
        }
        None => {
            client
                .put(upload_action.href)
                .body(reqwest::Body::wrap_stream(FramedRead::new(
                    file,
                    BytesCodec::new(),
                )))
                .send()
                .await
                .map_err(|err| {
                    PyException::new_err(format!("Error while uploading file: {err:?}"))
                })?
                .error_for_status()
                .map_err(|err| {
                    PyException::new_err(format!(
                        "Server responsded with error status code for file upload: {err}"
                    ))
                })?;
        }
    }
    if let Some(verify_action) = verify_action {
        client
            .post(verify_action.href)
            .basic_auth("USER", token)
            .json(&UploadedObject::from(upload_info))
            .send()
            .await
            .map_err(|err| {
                PyException::new_err(format!("Error while verifying file upload: {err}"))
            })?
            .error_for_status()
            .map_err(|err| {
                PyException::new_err(format!(
                    "Server responsded with error status code for upload verification: {err}"
                ))
            })?;
    }
    Ok(())
}

async fn upload_chunk(
    client: &reqwest::Client,
    url: &str,
    path: &str,
    start: u64,
    file_size: u64,
    chunk_size: u64,
    part_number: usize,
) -> PyResult<EtagWithPart> {
    let mut options = OpenOptions::new();
    let mut file = options.read(true).open(path).await?;
    let bytes_transfered = std::cmp::min(file_size - start, chunk_size);

    file.seek(SeekFrom::Start(start as u64)).await?;
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

    let etag_part = EtagWithPart {
        etag: response
            .headers()
            .get("etag")
            .ok_or(PyException::new_err(format!(
                "Missing Etag in response header"
            )))?
            .to_str()
            .map_err(|err| {
                PyException::new_err(format!("Error deserializing etag to string: {err}"))
            })?
            .to_owned()
            .replace("\\", "")
            .replace("\"", ""),
        part_number,
    };

    Ok(etag_part)
}

/// A Python module implemented in Rust.
#[pymodule]
fn hf_transfer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download, m)?)?;
    m.add_function(wrap_pyfunction!(upload, m)?)?;
    Ok(())
}
