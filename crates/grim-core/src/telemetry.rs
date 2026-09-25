//! Machine-readable fallback taxonomy (D2): every eager/host fallback emits
//! one JSON line when `GRIM_FALLBACK_LOG` is set, so silent downgrades show
//! up as countable events instead of free-form prose in stderr.
//!
//! Fire-and-forget by design: emission never fails, never panics, never
//! allocates on the hot path when disabled (one env read per call).

/// Why a fast path gave up and the caller degraded. One variant per
/// fallback class — adding a new fallback site without a variant is a
/// review failure, not a reason to reuse `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    GraphCaptureBegin,
    GraphCaptureEnd,
    GraphCaptureReplay,
    KvSeed,
    KvReseed,
    QkvArenaDevice,
    QkvStickyConfig,
    FusedQkvAttention,
    MoeDeviceDispatch,
    RopeDevBase,
    FusedFfn,
    Mxfp4Qkv,
    FusedQuantGemmDisabled,
    DeltaRuleDevice,
    MlaAbsorbedDecode,
    ShortConvDevice,
    SpeculativeDraft,
    LoraFusion,
    Other,
}

/// One fallback event. `component` is the emitting crate/module path
/// (e.g. `"grim-engine/scheduler"`); `detail` is a short human fragment.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FallbackEvent {
    pub component: &'static str,
    pub reason: FallbackReason,
    pub detail: String,
}

impl FallbackEvent {
    pub fn new(
        component: &'static str,
        reason: FallbackReason,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            component,
            reason,
            detail: detail.into(),
        }
    }

    /// Render the machine-readable line (no I/O — unit-testable).
    pub fn format(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            format!(
                "{{\"component\":\"{}\",\"reason\":\"other\",\"detail\":\"serialization failed\"}}",
                self.component,
            )
        })
    }
}

/// True when fallback logging is on (`GRIM_FALLBACK_LOG=1/json/true`).
pub fn fallback_enabled() -> bool {
    matches!(
        std::env::var("GRIM_FALLBACK_LOG").as_deref(),
        Ok("1") | Ok("json") | Ok("true")
    )
}

/// Emit one JSON line to stderr when enabled; silent no-op otherwise.
/// Never fails — a logging path must not break the path it observes.
pub fn emit_fallback(
    component: &'static str,
    reason: FallbackReason,
    detail: impl Into<String>,
) {
    if !fallback_enabled() {
        return;
    }
    eprintln!("{}", FallbackEvent::new(component, reason, detail).format());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D2: every reason serializes to a distinct snake_case token — the
    /// taxonomy is only useful if downstream `jq`/counters can split on it.
    #[test]
    fn fallback_reasons_serialize_distinct_snake_case() {
        let reasons = [
            FallbackReason::GraphCaptureBegin,
            FallbackReason::GraphCaptureEnd,
            FallbackReason::GraphCaptureReplay,
            FallbackReason::KvSeed,
            FallbackReason::KvReseed,
            FallbackReason::QkvArenaDevice,
            FallbackReason::QkvStickyConfig,
            FallbackReason::FusedQkvAttention,
            FallbackReason::MoeDeviceDispatch,
            FallbackReason::RopeDevBase,
            FallbackReason::FusedFfn,
            FallbackReason::Mxfp4Qkv,
            FallbackReason::FusedQuantGemmDisabled,
            FallbackReason::DeltaRuleDevice,
            FallbackReason::MlaAbsorbedDecode,
            FallbackReason::ShortConvDevice,
            FallbackReason::SpeculativeDraft,
            FallbackReason::LoraFusion,
            FallbackReason::Other,
        ];
        let mut seen = std::collections::HashSet::new();
        for r in reasons {
            let s = serde_json::to_string(&r).unwrap();
            assert!(
                s.starts_with('"') && !s.contains(char::is_uppercase),
                "snake_case token required, got {s}"
            );
            assert!(seen.insert(s), "duplicate reason token");
        }
    }

    /// D2: the emitted line round-trips through a JSON parser with all
    /// three fields present — proves machine-readability, not just prose.
    #[test]
    fn fallback_event_line_is_machine_readable_json() {
        let line = FallbackEvent::new(
            "grim-engine/scheduler",
            FallbackReason::GraphCaptureBegin,
            "capture_key=test ETA fallback",
        )
        .format();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["component"], "grim-engine/scheduler");
        assert_eq!(v["reason"], "graph_capture_begin");
        assert!(v["detail"].as_str().unwrap().contains("fallback"));
    }
}
