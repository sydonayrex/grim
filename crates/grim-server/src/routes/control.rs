use crate::{AppState, take_cancel_token};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
};
use futures::stream::{self, Stream};
use std::sync::Arc;

pub async fn pause_request_route(
    id: u64,
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    match pause_request_inner(&state, id) {
        Ok(out) => out,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{err}")})),
        ),
    }
}

pub async fn resume_request_route(
    id: u64,
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    match resume_request_inner(&state, id) {
        Ok(out) => out,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{err}")})),
        ),
    }
}

pub async fn cancel_request_route(
    id: u64,
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    match cancel_request_inner(&state, id) {
        Ok(out) => out,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{err}")})),
        ),
    }
}

pub async fn stream_state_route(
    id: u64,
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = std::result::Result<Event, axum::Error>>> {
    let state = state.clone();
    let stream = stream::unfold(0u64, move |tick| {
        let state = state.clone();
        async move {
            let snapshot = (|| -> Option<(String, String)> {
                let engine = state.engine.lock().ok()?;
                let sched = &engine.scheduler;
                let state_str = if sched.waiting.iter().any(|r| r.id == id) {
                    "waiting".to_string()
                } else if sched.running.iter().any(|r| r.id == id) {
                    "running".to_string()
                } else if sched.paused.iter().any(|r| r.id == id) {
                    "paused".to_string()
                } else if sched.swapped.iter().any(|r| r.id == id) {
                    "swapped".to_string()
                } else {
                    return None;
                };
                Some((state_str, format!("tick={tick}")))
            })();
            let event = match snapshot {
                Some((s, note)) => Ok(Event::default().event("state").data(format!(
                    r#"{{"id": {id}, "state": "{s}", "note": "{note}"}}"#
                ))),
                None => Ok(Event::default()
                    .event("end")
                    .data(format!(r#"{{"id": {id}}}"#))),
            };
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Some((event, tick.wrapping_add(1)))
        }
    });
    Sse::new(stream)
}

fn pause_request_inner(
    state: &Arc<AppState>,
    id: u64,
) -> grim_core::Result<(StatusCode, Json<serde_json::Value>)> {
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| grim_core::Error::Config("engine mutex poisoned".into()))?;
    if engine.is_paused(id) {
        return Ok((
            StatusCode::OK,
            Json(serde_json::json!({"id": id, "state": "paused"})),
        ));
    }
    let in_flight = engine.scheduler.waiting.iter().any(|r| r.id == id)
        || engine.scheduler.running.iter().any(|r| r.id == id)
        || engine.scheduler.swapped.iter().any(|r| r.id == id);
    if !in_flight {
        return Ok((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("request {id} not active"),
                "hint": "request may have already finished or never existed"
            })),
        ));
    }
    engine.pause_request(id);
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "id": id,
            "state": "paused",
            "message": "request execution paused, KV state retained"
        })),
    ))
}

fn resume_request_inner(
    state: &Arc<AppState>,
    id: u64,
) -> grim_core::Result<(StatusCode, Json<serde_json::Value>)> {
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| grim_core::Error::Config("engine mutex poisoned".into()))?;
    let in_flight = engine.scheduler.waiting.iter().any(|r| r.id == id)
        || engine.scheduler.running.iter().any(|r| r.id == id)
        || engine.scheduler.swapped.iter().any(|r| r.id == id);
    if in_flight {
        return Ok((
            StatusCode::OK,
            Json(serde_json::json!({"id": id, "state": "active"})),
        ));
    }
    if !engine.is_paused(id) {
        return Ok((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("request {id} not paused"),
                "hint": "request was not found in the paused queue"
            })),
        ));
    }
    engine.resume_request(id);
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "id": id,
            "state": "resumed",
            "message": "request returned to the waiting queue"
        })),
    ))
}

fn cancel_request_inner(
    state: &Arc<AppState>,
    id: u64,
) -> grim_core::Result<(StatusCode, Json<serde_json::Value>)> {
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| grim_core::Error::Config("engine mutex poisoned".into()))?;

    let known = engine.scheduler.waiting.iter().any(|r| r.id == id)
        || engine.scheduler.running.iter().any(|r| r.id == id)
        || engine.scheduler.paused.iter().any(|r| r.id == id)
        || engine.scheduler.swapped.iter().any(|r| r.id == id);

    if let Some(token) = take_cancel_token(id) {
        token.cancel();
    }

    if !known {
        engine.finish_request(id);
        return Ok((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "id": id,
                "state": "cancelled",
                "error": {
                    "type": "invalid_request_error",
                    "code": "unknown_request",
                    "message": format!("request id {id} is not known to the scheduler")
                }
            })),
        ));
    }

    engine.finish_request(id);
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "id": id,
            "state": "cancelled",
        })),
    ))
}
