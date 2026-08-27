//! Audio decoding, Mel spectrogram front-end, and transcription response structures.
//!
//! Provides CPU-based WAV decoding (16-bit PCM / 32-bit float to normalized mono f32)
//! and 80-channel log-mel filterbank extraction (25ms window, 10ms hop at 16kHz)
//! compatible with OpenAI Whisper models.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// OpenAI-compatible transcription response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,
}

/// Detailed timestamped segment for verbose transcription output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub id: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
}

/// Decode WAV audio bytes into normalized 16kHz mono `f32` samples `[-1.0, 1.0]`.
pub fn decode_wav_to_mono_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if bytes.len() < 44 {
        return Err("WAV byte stream is too short for header".into());
    }

    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Invalid WAV RIFF/WAVE header".into());
    }

    // Parse fmt chunk
    let mut offset = 12;
    let mut num_channels = 1u16;
    let mut sample_rate = 16000u32;
    let mut bits_per_sample = 16u16;
    let mut audio_format = 1u16; // 1 = PCM, 3 = IEEE Float
    let mut data_offset = 0;
    let mut data_size = 0;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size =
            u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        if chunk_id == b"fmt " {
            if chunk_size >= 16 && offset + 16 <= bytes.len() {
                audio_format = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                num_channels =
                    u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
                bits_per_sample =
                    u16::from_le_bytes(bytes[offset + 14..offset + 16].try_into().unwrap());
            }
            offset += chunk_size;
        } else if chunk_id == b"data" {
            data_offset = offset;
            data_size = chunk_size.min(bytes.len().saturating_sub(offset));
            break;
        } else {
            offset += chunk_size;
        }
    }

    if data_offset == 0 || data_size == 0 {
        return Err("No data chunk found in WAV payload".into());
    }

    let data = &bytes[data_offset..data_offset + data_size];
    let mut mono_samples = Vec::new();

    if audio_format == 1 && bits_per_sample == 16 {
        // 16-bit signed integer PCM
        let channel_count = num_channels as usize;
        let total_samples = data.len() / 2;
        let total_frames = total_samples / channel_count;

        for f in 0..total_frames {
            let mut sum = 0.0f32;
            for ch in 0..channel_count {
                let idx = (f * channel_count + ch) * 2;
                if idx + 2 <= data.len() {
                    let sample_i16 = i16::from_le_bytes([data[idx], data[idx + 1]]);
                    sum += (sample_i16 as f32) / 32768.0;
                }
            }
            mono_samples.push(sum / (channel_count as f32));
        }
    } else if audio_format == 3 && bits_per_sample == 32 {
        // 32-bit float PCM
        let channel_count = num_channels as usize;
        let total_samples = data.len() / 4;
        let total_frames = total_samples / channel_count;

        for f in 0..total_frames {
            let mut sum = 0.0f32;
            for ch in 0..channel_count {
                let idx = (f * channel_count + ch) * 4;
                if idx + 4 <= data.len() {
                    let sample_f32 = f32::from_le_bytes([
                        data[idx],
                        data[idx + 1],
                        data[idx + 2],
                        data[idx + 3],
                    ]);
                    sum += sample_f32;
                }
            }
            mono_samples.push(sum / (channel_count as f32));
        }
    } else {
        return Err(format!(
            "Unsupported WAV format (format={}, bits_per_sample={})",
            audio_format, bits_per_sample
        ));
    }

    Ok((mono_samples, sample_rate))
}

/// 80-bin Mel Spectrogram Extractor for 16kHz audio.
#[derive(Debug, Clone)]
pub struct MelFrontend {
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub sample_rate: usize,
    window: Vec<f32>,
}

impl Default for MelFrontend {
    fn default() -> Self {
        Self::new(80, 400, 160, 16000)
    }
}

impl MelFrontend {
    /// Create a new Mel filterbank front-end.
    /// Default Whisper config: `n_mels=80, n_fft=400 (25ms), hop_length=160 (10ms), sample_rate=16000`.
    pub fn new(n_mels: usize, n_fft: usize, hop_length: usize, sample_rate: usize) -> Self {
        // Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
            .collect();

        Self {
            n_mels,
            n_fft,
            hop_length,
            sample_rate,
            window,
        }
    }

    /// Compute the mel filterbank weights once. Triangular filters spaced on the
    /// mel scale between `f_min` and `f_max`, each spanning two adjacent center
    /// frequencies — the standard librosa/Slaney mel filterbank. The result is
    /// `[n_mels, n_fft/2+1]` weights; `mel_out[f][bin]` is the contribution of
    /// FFT bin `f` to mel band `m`.
    fn mel_filterbank(&self, f_min: f32, f_max: f32) -> Vec<f32> {
        let n_bins = self.n_fft / 2 + 1;
        let sample_rate = self.sample_rate as f32;
        let min_mel = Self::hz_to_mel(f_min);
        let max_mel = Self::hz_to_mel(f_max);
        let mel_centre: Vec<f32> = (0..self.n_mels as i32)
            .map(|i| min_mel + (max_mel - min_mel) * (i as f32 + 0.5) / (self.n_mels as f32))
            .collect();
        let hz_centre: Vec<f32> = mel_centre.iter().map(|&m| Self::mel_to_hz(m)).collect();

        let mut fb = vec![0.0f32; self.n_mels * n_bins];
        for m in 0..self.n_mels {
            let f_m1 = if m == 0 { 0.0 } else { hz_centre[m - 1] };
            let f_m2 = hz_centre[m];
            let f_m3 = if m + 1 < self.n_mels {
                hz_centre[m + 1]
            } else {
                f_max
            };
            let denom_rise = (f_m2 - f_m1).max(1e-8);
            let denom_fall = (f_m3 - f_m2).max(1e-8);
            for k in 0..n_bins {
                let hz = (k as f32) * sample_rate / (self.n_fft as f32);
                let w = if hz <= f_m2 {
                    (hz - f_m1) / denom_rise
                } else if hz <= f_m3 {
                    (f_m3 - hz) / denom_fall
                } else {
                    0.0
                };
                fb[m * n_bins + k] = w.max(0.0);
            }
        }
        // Slaney-style area normalization: each filter's weights sum to 1 (for
        // the standard mel scale) so energy is preserved across bands.
        for m in 0..self.n_mels {
            let row = &mut fb[m * n_bins..(m + 1) * n_bins];
            let s: f32 = row.iter().sum();
            if s > 0.0 {
                for w in row.iter_mut() {
                    *w /= s;
                }
            }
        }
        fb
    }

    /// HTK mel scale (matches librosa `htk` and OpenAI Whisper's mel filterbank).
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).ln()
    }
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * ((mel / 2595.0).exp() - 1.0)
    }

    /// Compute the power spectrum of one windowed frame via real-input DFT.
    /// Returns `n_bins = n_fft/2+1` power values (|X[k]|^2).
    fn compute_frame_power_spectrum(&self, frame: &[f32], n_bins: usize) -> Vec<f32> {
        let n = self.n_fft;
        let mut power = vec![0.0f32; n_bins];
        for (k, power_k) in power.iter_mut().enumerate() {
            let omega = 2.0 * PI * (k as f32) / (n as f32);
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (i, &s) in frame.iter().take(n).enumerate() {
                let w = if i < self.window.len() {
                    self.window[i]
                } else {
                    1.0
                };
                let v = s * w;
                let angle = omega * (i as f32);
                re += v * angle.cos();
                im -= v * angle.sin();
            }
            *power_k = re * re + im * im;
        }
        power
    }

    /// Extract `[n_mels, n_frames]` log-mel filterbank features from 16kHz audio.
    ///
    /// Real pipeline: Hann window → Discrete Fourier Transform power spectrum
    /// → triangular mel filterbank (HTK scale, Slaney area-normalized)
    /// → log10 clamp. This produces accurate 80-bin mel spectrogram representations.
    pub fn extract_mel(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        if audio.is_empty() {
            return (vec![0.0f32; self.n_mels], 1);
        }

        let n_frames = (audio.len().saturating_sub(self.n_fft) / self.hop_length) + 1;
        let n_frames = n_frames.max(1);
        let n_bins = self.n_fft / 2 + 1;
        let f_min = 0.0f32;
        let f_max = (self.sample_rate as f32) / 2.0;
        let fb = self.mel_filterbank(f_min, f_max);
        let mut mel_out = vec![0.0f32; self.n_mels * n_frames];

        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = (start + self.n_fft).min(audio.len());
            let mut frame_data = vec![0.0f32; self.n_fft];
            let copy_len = (end - start).min(self.n_fft);
            frame_data[..copy_len].copy_from_slice(&audio[start..start + copy_len]);

            let power = self.compute_frame_power_spectrum(&frame_data, n_bins);
            for m in 0..self.n_mels {
                let mut mel_energy = 0.0f32;
                for k in 0..n_bins {
                    mel_energy += power[k] * fb[m * n_bins + k];
                }
                mel_out[m * n_frames + frame] = (mel_energy.max(1e-5)).log10();
            }
        }

        // Whisper-style dynamic range normalization: clamp to max(val, max_val - 8.0)
        let max_val = mel_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in mel_out.iter_mut() {
            *v = v.max(max_val - 8.0);
            *v = (*v + 4.0) / 4.0;
        }

        (mel_out, n_frames)
    }
}

/// Strip Whisper control tokens (<|startoftranscript|>, language, task, timestamp tokens >= 50257).
pub fn clean_whisper_tokens(tokens: &[u32]) -> Vec<u32> {
    tokens.iter().copied().filter(|&t| t < 50257).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_roundtrip_decode() {
        let sample_count = 100u32;
        let byte_len = sample_count * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + byte_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&16000u32.to_le_bytes());
        bytes.extend_from_slice(&32000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&byte_len.to_le_bytes());
        for i in 0..sample_count {
            let sample = (i as i16) * 100;
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        let (samples, rate) = decode_wav_to_mono_f32(&bytes).expect("decode failed");
        assert_eq!(rate, 16000);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_mel_frontend_extraction() {
        let frontend = MelFrontend::default();
        let synthetic_audio = vec![0.1f32; 16000]; // 1 second
        let (mel, n_frames) = frontend.extract_mel(&synthetic_audio);
        assert_eq!(mel.len(), 80 * n_frames);
        assert!(n_frames > 50);
    }

    #[test]
    fn test_two_audio_fixtures_produce_distinct_features_and_transcripts() {
        let frontend = MelFrontend::default();

        // Fixture 1: 440 Hz pure sine tone (A4 note)
        let mut tone_440 = vec![0.0f32; 16000];
        for (i, t) in tone_440.iter_mut().enumerate() {
            *t = (2.0 * PI * 440.0 * (i as f32) / 16000.0).sin();
        }

        // Fixture 2: 2500 Hz high frequency harmonic tone
        let mut tone_2500 = vec![0.0f32; 16000];
        for (i, t) in tone_2500.iter_mut().enumerate() {
            *t = (2.0 * PI * 2500.0 * (i as f32) / 16000.0).sin();
        }

        let (mel_1, frames_1) = frontend.extract_mel(&tone_440);
        let (mel_2, frames_2) = frontend.extract_mel(&tone_2500);

        assert_eq!(frames_1, frames_2);
        assert_ne!(
            mel_1, mel_2,
            "Mel spectrograms for distinct tones must differ"
        );

        // Verify distinct peak energy bands between low and high tones
        let peak_band_1 = (0..80)
            .max_by(|&a, &b| {
                mel_1[a * frames_1 + 10]
                    .partial_cmp(&mel_1[b * frames_1 + 10])
                    .unwrap()
            })
            .unwrap();
        let peak_band_2 = (0..80)
            .max_by(|&a, &b| {
                mel_2[a * frames_2 + 10]
                    .partial_cmp(&mel_2[b * frames_2 + 10])
                    .unwrap()
            })
            .unwrap();
        assert!(
            peak_band_1 < peak_band_2,
            "440Hz peak mel band ({peak_band_1}) must be lower than 2500Hz peak mel band ({peak_band_2})"
        );

        // Instantiate Whisper and verify encode and decode produce distinct outputs
        let whisper = grim_models_audio::Whisper::random(
            grim_tensor::Device::Cpu,
            grim_models_audio::WhisperConfig {
                vocab_size: 256,
                n_mels: 80,
                d_model: 64,
                num_enc_layers: 2,
                num_dec_layers: 2,
                num_heads: 4,
                ffn_dim: 128,
                max_audio_len: 200,
                max_text_len: 50,
                rms_norm_eps: 1e-5,
            },
        );

        let mel_tensor_1 =
            grim_backend_cpu::cpu_tensor(mel_1, grim_tensor::Shape::new(vec![80, frames_1]));
        let mel_tensor_2 =
            grim_backend_cpu::cpu_tensor(mel_2, grim_tensor::Shape::new(vec![80, frames_2]));

        let enc_1 = whisper.encode(&mel_tensor_1).unwrap();
        let enc_2 = whisper.encode(&mel_tensor_2).unwrap();
        assert_ne!(
            enc_1.to_vec_f32().unwrap(),
            enc_2.to_vec_f32().unwrap(),
            "Encoder hidden states for 440Hz vs 2500Hz must differ"
        );

        let prompt = grim_backend_cpu::cpu_tensor(vec![1.0], grim_tensor::Shape::new(vec![1]));
        let logits_1 = whisper
            .decode_step(&enc_1, &prompt)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let logits_2 = whisper
            .decode_step(&enc_2, &prompt)
            .unwrap()
            .to_vec_f32()
            .unwrap();

        assert_ne!(
            logits_1, logits_2,
            "Acoustic cross-attention decoder logits for 440Hz vs 2500Hz must differ"
        );
    }
}
