use crate::{
    AppState, CompletionRequest, REQUEST_ID_COUNTER, REQUEST_SAMPLER_PARAMS, SamplerParams,
    sample_next_token, take_request_sampler_params,
};
use axum::{
    Json,
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use std::sync::atomic::Ordering;

pub(crate) async fn text_completions_route(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CompletionRequest>,
) -> Response {
    let raw_prompt = if let Some(s) = payload.prompt.as_str() {
        s.to_string()
    } else if let Some(arr) = payload.prompt.as_array() {
        arr.iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        payload.prompt.to_string()
    };

    let prompt_tokens: Vec<u32> = {
        let tok_guard = state.lock_tokenizer();
        if let Some(ref tok) = *tok_guard {
            tok.encode(&raw_prompt)
        } else {
            raw_prompt.bytes().map(|b| b as u32).collect()
        }
    };

    let max_tokens = payload.max_tokens.unwrap_or(16);
    let stream_requested = payload.stream.unwrap_or(false);
    let model_name = payload
        .model
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let req_id = REQUEST_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

    let request_params = SamplerParams {
        min_tokens: 0,
        temperature: payload.temperature,
        top_k: payload.top_k,
        top_p: payload.top_p,
        seed: payload.seed,
        repeat_penalty: payload.repeat_penalty,
    };
    if let Ok(mut reg) = REQUEST_SAMPLER_PARAMS.lock() {
        reg.insert(req_id, request_params);
    }

    let (vocab_size, eos_token_id) = {
        let tok = state.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
        let vs = tok.as_ref().map(|t| t.tokens.len()).unwrap_or(32000);
        let eos = tok.as_ref().and_then(|t| t.eos_token_id);
        (vs, eos)
    };

    let sampling = grim_core::sampler::SamplingParams {
        temperature: payload.temperature.unwrap_or(1.0),
        top_p: payload.top_p.unwrap_or(1.0),
        top_k: payload.top_k.unwrap_or(0).max(0) as u32,
        repeat_penalty: payload.repeat_penalty.unwrap_or(1.0),
        ..grim_core::sampler::SamplingParams::default()
    };
    let sampler: std::sync::Arc<dyn grim_core::sampler::Sampler> =
        std::sync::Arc::from(sampling.into_sampler(payload.seed.unwrap_or(0)));

    if stream_requested {
        let state_clone = state.clone();
        let model_clone = model_name.clone();
        let sampler_clone = sampler.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            for step in 0..max_tokens {
                let sampled = {
                    let mut engine = match state_clone.engine.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    sample_next_token(
                        &mut engine,
                        req_id,
                        step as u64,
                        sampler_clone.as_ref(),
                        if step == 0 {
                            Some(&prompt_tokens)
                        } else {
                            None
                        },
                        vocab_size,
                        Some(model_clone.clone()),
                    )
                };
                match sampled {
                    Ok(token_id) => {
                        if Some(token_id) == eos_token_id {
                            break;
                        }
                        let token_text = {
                            let tok_guard = state_clone.lock_tokenizer();
                            if let Some(ref tok) = *tok_guard {
                                tok.decode(&[token_id])
                            } else {
                                format!(" {token_id}")
                            }
                        };
                        let chunk = serde_json::json!({
                            "id": format!("cmpl-{req_id}"),
                            "object": "text_completion",
                            "created": 1700000000,
                            "model": model_clone,
                            "choices": [{
                                "text": token_text,
                                "index": 0,
                                "logprobs": null,
                                "finish_reason": null
                            }]
                        });
                        let _ = tx.send(Ok(format!(
                            "data: {}\n\n",
                            serde_json::to_string(&chunk).unwrap()
                        )));
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
            let mut engine = match state_clone.engine.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            engine.finish_request(req_id);
            take_request_sampler_params(req_id);
            drop(engine);
            let final_chunk = serde_json::json!({
                "id": format!("cmpl-{req_id}"),
                "object": "text_completion",
                "created": 1700000000,
                "model": model_clone,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": "stop"
                }]
            });
            let _ = tx.send(Ok(format!(
                "data: {}\n\n",
                serde_json::to_string(&final_chunk).unwrap()
            )));
            let _ = tx.send(Ok("data: [DONE]\n\n".to_string()));
        });

        let stream = futures::stream::unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Some(Ok(s)) => Some((Ok::<_, axum::Error>(axum::body::Bytes::from(s)), rx)),
                _ => None,
            }
        });
        axum::response::Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let mut gen_tokens = Vec::new();
        for step in 0..max_tokens {
            let sampled = {
                let mut engine = match state.engine.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                sample_next_token(
                    &mut engine,
                    req_id,
                    step as u64,
                    sampler.as_ref(),
                    if step == 0 {
                        Some(&prompt_tokens)
                    } else {
                        None
                    },
                    vocab_size,
                    Some(model_name.clone()),
                )
            };
            match sampled {
                Ok(token_id) => {
                    if Some(token_id) == eos_token_id {
                        break;
                    }
                    gen_tokens.push(token_id);
                }
                Err(e) => {
                    let mut engine = match state.engine.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    engine.finish_request(req_id);
                    take_request_sampler_params(req_id);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": { "message": e, "type": "server_error" }
                        })),
                    )
                        .into_response();
                }
            }
        }
        let mut engine = match state.engine.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        engine.finish_request(req_id);
        take_request_sampler_params(req_id);
        drop(engine);

        let gen_text = {
            let tok_guard = state.lock_tokenizer();
            if let Some(ref tok) = *tok_guard {
                tok.decode(&gen_tokens)
            } else {
                gen_tokens
                    .iter()
                    .map(|t| format!("{t}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };
        let res = serde_json::json!({
            "id": format!("cmpl-{req_id}"),
            "object": "text_completion",
            "created": 1700000000,
            "model": model_name,
            "choices": [{
                "text": gen_text,
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens.len(),
                "completion_tokens": gen_tokens.len(),
                "total_tokens": prompt_tokens.len() + gen_tokens.len()
            }
        });
        (StatusCode::OK, Json(res)).into_response()
    }
}
