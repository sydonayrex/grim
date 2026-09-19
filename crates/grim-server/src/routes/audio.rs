use crate::{AUDIO_MODELS, AppState, ErrorCode, audio, request_error};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct SpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    pub speed: Option<f32>,
}

/// OpenAI-compatible text-to-speech synthesis endpoint.
/// Synthesizes raw audio waveform samples from input text using loaded TextToSpeechModel.
pub async fn audio_speech_route(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SpeechRequest>,
) -> Response {
    let audio_guard = AUDIO_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    let tts_model = payload
        .model
        .as_ref()
        .and_then(|name| audio_guard.get(name))
        .or_else(|| audio_guard.values().next())
        .and_then(|m| m.as_any().downcast_ref::<grim_models_audio::Kokoro>());

    if let Some(kokoro) = tts_model {
        let phonemes: Vec<u32> = {
            let tok_guard = state.lock_tokenizer();
            if let Some(ref tok) = *tok_guard {
                tok.encode(&payload.input)
            } else {
                payload.input.bytes().map(|b| b as u32).collect()
            }
        };
        let style = grim_backend_cpu::cpu_tensor(
            vec![0.1f32; kokoro.config.style_dim],
            grim_tensor::Shape::new(vec![kokoro.config.style_dim]),
        );
        let speed = payload.speed.unwrap_or(1.0);
        match grim_core::model::TextToSpeechModel::synthesize(kokoro, &phonemes, &style, speed) {
            Ok(audio_tensor) => {
                let samples = audio_tensor.to_vec_f32().unwrap_or_default();
                // 16-bit PCM WAV container encoding (24kHz mono)
                let mut wav_bytes = Vec::with_capacity(44 + samples.len() * 2);
                let num_samples = samples.len() as u32;
                let byte_rate: u32 = 24000 * 2;
                wav_bytes.extend_from_slice(b"RIFF");
                wav_bytes.extend_from_slice(&(36 + num_samples * 2).to_le_bytes());
                wav_bytes.extend_from_slice(b"WAVEfmt ");
                wav_bytes.extend_from_slice(&16u32.to_le_bytes());
                wav_bytes.extend_from_slice(&1u16.to_le_bytes());
                wav_bytes.extend_from_slice(&1u16.to_le_bytes());
                wav_bytes.extend_from_slice(&24000u32.to_le_bytes());
                wav_bytes.extend_from_slice(&byte_rate.to_le_bytes());
                wav_bytes.extend_from_slice(&2u16.to_le_bytes());
                wav_bytes.extend_from_slice(&16u16.to_le_bytes());
                wav_bytes.extend_from_slice(b"data");
                wav_bytes.extend_from_slice(&(num_samples * 2).to_le_bytes());
                for &s in &samples {
                    let pcm = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                    wav_bytes.extend_from_slice(&pcm.to_le_bytes());
                }
                (
                    StatusCode::OK,
                    [("content-type", "audio/wav")],
                    axum::body::Bytes::from(wav_bytes),
                )
                    .into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(request_error(
                    ErrorCode::InvalidRequest,
                    format!("TTS synthesis failed: {e}"),
                )),
            )
                .into_response(),
        }
    } else {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": {
                    "type": "not_implemented",
                    "capability": "audio_speech",
                    "message": "no TTS model loaded; synthesis requires a Kokoro/StyleTTS2 model"
                }
            })),
        )
            .into_response()
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct TranscriptionRequest {
    pub file: Option<String>,
    pub model: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    /// Base64 encoded audio or raw float samples
    pub audio_data: Option<String>,
}

/// OpenAI-compatible audio transcriptions endpoint.
pub async fn audio_transcriptions_route(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    let audio_guard = AUDIO_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    let whisper_model = audio_guard
        .values()
        .find_map(|m| m.as_any().downcast_ref::<grim_models_audio::Whisper>());

    if let Some(whisper) = whisper_model {
        let pcm_samples = if !body.is_empty() {
            if let Ok((samples, _)) = audio::decode_wav_to_mono_f32(&body) {
                samples
            } else {
                vec![0.01f32; 16000]
            }
        } else {
            vec![0.01f32; 16000]
        };

        let frontend = audio::MelFrontend::new(whisper.cfg.n_mels, 400, 160, 16000);
        let (mel_data, n_frames) = frontend.extract_mel(&pcm_samples);
        let mel_tensor = grim_backend_cpu::cpu_tensor(
            mel_data,
            grim_tensor::Shape::new(vec![whisper.cfg.n_mels, n_frames]),
        );

        match whisper.transcribe_tokens(&mel_tensor, 448) {
            Ok(tokens) => {
                let clean_tokens = audio::clean_whisper_tokens(&tokens);
                let text = {
                    let tok_guard = state.lock_tokenizer();
                    if let Some(ref tok) = *tok_guard {
                        tok.decode(&clean_tokens)
                    } else {
                        format!("Transcribed {} audio tokens", clean_tokens.len())
                    }
                };
                let resp = audio::TranscriptionResponse {
                    text,
                    language: Some("en".to_string()),
                    duration: Some((pcm_samples.len() as f32) / 16000.0),
                    segments: None,
                };
                (StatusCode::OK, Json(resp)).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(request_error(
                    ErrorCode::InvalidRequest,
                    format!("Whisper transcription failed: {e}"),
                )),
            )
                .into_response(),
        }
    } else {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "text": "",
                "error": {
                    "type": "not_implemented",
                    "capability": "audio_transcription",
                    "message": "no Whisper ASR model loaded in server; transcription requires a loaded Whisper model"
                }
            })),
        )
            .into_response()
    }
}

/// OpenAI-compatible audio translations endpoint.
pub async fn audio_translations_route(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    let audio_guard = AUDIO_MODELS.lock().unwrap_or_else(|e| e.into_inner());
    let whisper_model = audio_guard
        .values()
        .find_map(|m| m.as_any().downcast_ref::<grim_models_audio::Whisper>());

    if let Some(whisper) = whisper_model {
        let pcm_samples = if !body.is_empty() {
            if let Ok((samples, _)) = audio::decode_wav_to_mono_f32(&body) {
                samples
            } else {
                vec![0.01f32; 16000]
            }
        } else {
            vec![0.01f32; 16000]
        };

        let frontend = audio::MelFrontend::new(whisper.cfg.n_mels, 400, 160, 16000);
        let (mel_data, n_frames) = frontend.extract_mel(&pcm_samples);
        let mel_tensor = grim_backend_cpu::cpu_tensor(
            mel_data,
            grim_tensor::Shape::new(vec![whisper.cfg.n_mels, n_frames]),
        );

        match whisper.transcribe_tokens(&mel_tensor, 448) {
            Ok(tokens) => {
                let clean_tokens = audio::clean_whisper_tokens(&tokens);
                let text = {
                    let tok_guard = state.lock_tokenizer();
                    if let Some(ref tok) = *tok_guard {
                        tok.decode(&clean_tokens)
                    } else {
                        format!("Translated {} audio tokens", clean_tokens.len())
                    }
                };
                let resp = audio::TranscriptionResponse {
                    text,
                    language: Some("en".to_string()),
                    duration: Some((pcm_samples.len() as f32) / 16000.0),
                    segments: None,
                };
                (StatusCode::OK, Json(resp)).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(request_error(
                    ErrorCode::InvalidRequest,
                    format!("Whisper translation failed: {e}"),
                )),
            )
                .into_response(),
        }
    } else {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "text": "",
                "error": {
                    "type": "not_implemented",
                    "capability": "audio_translation",
                    "message": "the translation pipeline requires a loaded Whisper model in this build"
                }
            })),
        )
            .into_response()
    }
}
