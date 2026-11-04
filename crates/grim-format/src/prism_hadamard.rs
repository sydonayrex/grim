//! prism.hadamard.* GGUF metadata contract (PrismML fork, llama-model.cpp:2090-2262).
//! Every error message mirrors the fork so a bad checkpoint fails identically.

use std::collections::HashMap;

use grim_tensor::error::{Error, Result};

use crate::gguf::GgufValue;

pub const KEY_VERSION: &str = "prism.hadamard.version";
pub const KEY_TIED_OUTPUT: &str = "prism.hadamard.tied_output";
pub const KEY_BLOCK_SIZE: &str = "prism.hadamard.block_size";
pub const KEY_TRANSFORM: &str = "prism.hadamard.transform";
pub const KEY_AXIS: &str = "prism.hadamard.axis";
pub const KEY_SIGN_MODE: &str = "prism.hadamard.sign_mode";
pub const KEY_SIGN_WIDTHS: &str = "prism.hadamard.sign_widths";
pub const KEY_SIGN_VALUES: &str = "prism.hadamard.sign_values";
pub const KEY_WEIGHT_NAMES: &str = "prism.hadamard.weight_names";
pub const KEY_INVERSE_WEIGHT_NAMES: &str = "prism.hadamard.inverse_weight_names";
pub const KEY_GDN_V_GROUPED: &str = "prism.hadamard.gdn_v_grouped";

pub const TRANSFORM_SYLVESTER: &str = "normalized-sylvester-walsh-hadamard";
pub const AXIS_INPUT_LAST: &str = "input-last-dimension";
pub const SIGN_MODE_IDENTITY: &str = "identity";
pub const SIGN_MODE_EXPLICIT: &str = "explicit";

#[derive(Debug, Clone)]
pub struct PrismHadamardMeta {
    pub version: u32,
    pub block_size: u32,
    pub sign_widths: Vec<i32>,
    pub sign_values: Vec<i32>,
    pub weight_names: Vec<String>,
    pub inverse_weight_names: Vec<String>,
    pub gdn_v_grouped: bool,
    pub tied_output: bool,
}

impl PrismHadamardMeta {
    /// Sign lookup by FULL tensor input dimension (llama-model.cpp:2395-2396:
    /// `const uint32_t width = (uint32_t) weight->ne[0];` exact match only).
    /// `None` = no group with this width; callers MUST hard-error (the fork throws
    /// `prism.hadamard has no sign vector for width %u`).
    pub fn sign_vector_for(&self, width: usize) -> Option<&[i32]> {
        if width as u32 % self.block_size != 0 {
            return None;
        }
        let mut off = 0usize;
        for &w in &self.sign_widths {
            let w = w as usize;
            if w == width {
                return Some(&self.sign_values[off..off + w]);
            }
            off += w;
        }
        None
    }
}

/// Free-function form of the per-tensor lookup (see [`PrismHadamardMeta::sign_vector_for`]).
pub fn sign_vector_for<'a>(meta: &'a PrismHadamardMeta, width: usize) -> Option<&'a [i32]> {
    meta.sign_vector_for(width)
}

fn err(msg: impl Into<String>) -> Error {
    Error::Backend(msg.into())
}

fn required_u32(meta: &HashMap<String, GgufValue>, key: &str) -> Result<u32> {
    meta.get(key)
        .and_then(GgufValue::as_u32)
        .ok_or_else(|| err(format!("missing or malformed gguf key: {key}")))
}

fn required_str<'a>(meta: &'a HashMap<String, GgufValue>, key: &str) -> Result<&'a str> {
    meta.get(key)
        .and_then(GgufValue::as_str)
        .ok_or_else(|| err(format!("missing or malformed gguf key: {key}")))
}

fn required_i32_array(meta: &HashMap<String, GgufValue>, key: &str) -> Result<Vec<i32>> {
    meta.get(key)
        .and_then(GgufValue::as_i32_array)
        .ok_or_else(|| err(format!("missing or malformed gguf key: {key}")))
}

fn required_str_array(meta: &HashMap<String, GgufValue>, key: &str) -> Result<Vec<String>> {
    match meta.get(key).and_then(GgufValue::as_array) {
        Some(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                out.push(
                    item.as_str()
                        .ok_or_else(|| err(format!("missing or malformed gguf key: {key}")))?
                        .to_string(),
                );
            }
            Ok(out)
        }
        None => Err(err(format!("missing or malformed gguf key: {key}"))),
    }
}

fn optional_str_array(meta: &HashMap<String, GgufValue>, key: &str) -> Result<Vec<String>> {
    match meta.get(key) {
        None => Ok(Vec::new()),
        Some(_) => required_str_array(meta, key),
    }
}

/// Parse + validate the `prism.hadamard.*` metadata block.
///
/// `has_weight` reports whether a tensor name exists in the checkpoint's tensor
/// table (needed because `tied_output` forbids a physical `output.weight`).
/// `Ok(None)` when no `prism.hadamard.version` key is present (plain models).
/// Every validation failure is `Err` with the fork's exact message.
pub fn parse_prism_hadamard(
    meta: &HashMap<String, GgufValue>,
    has_weight: &dyn Fn(&str) -> bool,
) -> Result<Option<PrismHadamardMeta>> {
    // ml.get_key(TIED_OUTPUT, hdmd.tied_output, false) — optional, default false.
    let tied_output = meta.get(KEY_TIED_OUTPUT).and_then(GgufValue::as_bool).unwrap_or(false);

    let version = match meta.get(KEY_VERSION).and_then(GgufValue::as_u32) {
        None => {
            // No version key: only the tied_output tail rule still applies.
            if tied_output {
                return Err(err("prism.hadamard.tied_output requires version 2"));
            }
            return Ok(None);
        }
        Some(v) => v,
    };

    if version != 1 && version != 2 {
        return Err(err(format!("unsupported {KEY_VERSION}: {version}")));
    }

    if (version == 2) != tied_output {
        return Err(err(
            "prism.hadamard version 2 requires tied_output=true; version 1 forbids it",
        ));
    }
    if tied_output && has_weight("output.weight") {
        return Err(err("prism.hadamard.tied_output requires output.weight to be absent"));
    }

    let block_size = required_u32(meta, KEY_BLOCK_SIZE)?;
    let transform = required_str(meta, KEY_TRANSFORM)?;
    let axis = required_str(meta, KEY_AXIS)?;
    let sign_mode = required_str(meta, KEY_SIGN_MODE)?;
    let weight_names = required_str_array(meta, KEY_WEIGHT_NAMES)?;

    if block_size == 0 || (block_size & (block_size - 1)) != 0 {
        return Err(err(format!("invalid {KEY_BLOCK_SIZE}: {block_size}")));
    }
    if transform != TRANSFORM_SYLVESTER {
        return Err(err(format!("unsupported {KEY_TRANSFORM}: {transform}")));
    }
    if axis != AXIS_INPUT_LAST {
        return Err(err(format!("unsupported {KEY_AXIS}: {axis}")));
    }
    if sign_mode != SIGN_MODE_IDENTITY && sign_mode != SIGN_MODE_EXPLICIT {
        return Err(err(format!("unsupported {KEY_SIGN_MODE}: {sign_mode}")));
    }
    if weight_names.is_empty() {
        return Err(err("prism.hadamard.weight_names is empty"));
    }

    let mut sign_widths: Vec<i32> = Vec::new();
    let mut sign_values: Vec<i32> = Vec::new();
    if sign_mode == SIGN_MODE_EXPLICIT {
        sign_widths = required_i32_array(meta, KEY_SIGN_WIDTHS)?;
        sign_values = required_i32_array(meta, KEY_SIGN_VALUES)?;
        // explicit mode with no widths gives an empty sign table, which acts as
        // identity and changes the model
        if sign_widths.is_empty() {
            return Err(err("prism.hadamard.sign_mode is explicit but sign_widths is empty"));
        }
        let mut off: usize = 0;
        let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for &width in &sign_widths {
            if width <= 0
                || (width as u32) % block_size != 0
                || off + width as usize > sign_values.len()
            {
                return Err(err(format!("invalid prism.hadamard sign width: {width}")));
            }
            if !seen.insert(width) {
                return Err(err(format!("duplicate prism.hadamard sign width: {width}")));
            }
            for &v in &sign_values[off..off + width as usize] {
                if v != 1 && v != -1 {
                    return Err(err("prism.hadamard sign values must be +/-1"));
                }
            }
            off += width as usize;
        }
        if off != sign_values.len() {
            return Err(err("prism.hadamard.sign_values length mismatch"));
        }
    }

    let gdn_v_grouped = meta.get(KEY_GDN_V_GROUPED).and_then(GgufValue::as_bool).unwrap_or(false);

    // tables read by row lookup store latent rows: the inverse transform goes
    // on the lookup result (optional key, default empty).
    let inverse_weight_names = optional_str_array(meta, KEY_INVERSE_WEIGHT_NAMES)?;
    let has_token_embd_inverse = inverse_weight_names.iter().any(|n| n == "token_embd.weight");

    if tied_output {
        // (version == 2) == tied_output was already enforced above.
        if !has_token_embd_inverse {
            return Err(err("prism.hadamard.tied_output requires a latent token embedding"));
        }
    } else if has_token_embd_inverse && !has_weight("output.weight") {
        return Err(err("a tied Hadamard output requires version 2 and tied_output=true"));
    }

    Ok(Some(PrismHadamardMeta {
        version,
        block_size,
        sign_widths,
        sign_values,
        weight_names,
        inverse_weight_names,
        gdn_v_grouped,
        tied_output,
    }))
}
