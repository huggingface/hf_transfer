use rand::{thread_rng, Rng};
use reqwest::{RequestBuilder, Response};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug)]
pub enum HttpError {
    RequestError { status_code: u16 },
    InvalidRequestBuilder,
}

fn jitter() -> u32 {
    thread_rng().gen_range(0..=500)
}

pub fn exponential_backoff(base_wait_time: u32, n: u32, max: u32) -> u32 {
    base_wait_time + n.pow(2) + jitter().min(max)
}

pub async fn retry_http(
    request: RequestBuilder,
    max_retries: u32,
    validate_response: impl Fn(&Response) -> bool,
) -> Result<Response, HttpError> {
    let mut result = None;
    for i in 1..=max_retries {
        let http_call = request
            .try_clone()
            .ok_or(HttpError::InvalidRequestBuilder)?;
        let r = http_call.send().await;
        match r {
            Ok(res) if validate_response(&res) => return Ok(res),
            Ok(res) => result = Some(res),
            Err(err) => (), // error!(
                            // err_msg = err.to_string(),
                            // "Probably a network error, retrying"
                            // ),
        }
        let wait_time = exponential_backoff(300, i, 2000);
        // warn!(retry_in = wait_time, "unexpected response status or error");
        sleep(Duration::from_millis(wait_time.into())).await;
    }
    Err(HttpError::RequestError {
        status_code: result.map(|res| res.status().as_u16()).unwrap_or(0),
    })
}
