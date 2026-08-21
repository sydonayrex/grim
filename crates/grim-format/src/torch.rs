//! PyTorch `.pth` and TorchScript `.pt` checkpoint reader.
//!
//! Parses PyTorch ZIP containers (legacy `torch.save` archives and JIT
//! exports — both store entries uncompressed) and interprets the pickle
//! stream with a small stack VM so tensors come out with their real names,
//! shapes, strides, and dtypes. Implements [`TensorProvider`] so
//! `grim_nn::WeightSource` can load real checkpoints (e.g. the Kokoro-82M
//! `.pth` under `models/audio/`) without a Python runtime.
//!
//! Supported pickle surface is the subset PyTorch emits for state dicts:
//! `PROTO/FRAME`, marks + tuples/lists/dicts, `GLOBAL`/`STACK_GLOBAL`,
//! `REDUCE` of `_rebuild_tensor_v2` / `_rebuild_tensor` /
//! `_rebuild_parameter`, `BINPERSID` storage references, memo ops, and the
//! integer/string/bool atom opcodes. Anything else degrades to an opaque
//! value that is skipped during tensor extraction.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use grim_tensor::dtype::{ArithType, DType, QuantProvenance, Storage};
use grim_tensor::error::{Error, Result};
use grim_tensor::provider::{RawTensor, TensorMeta, TensorProvider};

/// Parsed tensor storage descriptor inside a PyTorch file.
#[derive(Debug, Clone)]
pub struct TorchTensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

/// Tensor provider for PyTorch `.pth` and `.pt` checkpoints.
pub struct PthProvider {
    tensors: HashMap<String, TorchTensorEntry>,
}

impl PthProvider {
    /// Load a PyTorch `.pth` or `.pt` checkpoint from a filesystem path.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| Error::Io(e))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| Error::Io(e))?;
        Self::load_from_bytes(&bytes)
    }

    /// Parse a PyTorch `.pth` or `.pt` checkpoint from raw memory bytes.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut tensors = HashMap::new();

        if bytes.len() >= 4 && &bytes[0..4] == b"PK\x03\x04" {
            let entries = parse_zip_entries(bytes)?;
            // Legacy saves use `<prefix>/data.pkl` with prefix varying by
            // producer ("archive/", "model/", JIT export names, ...).
            let pkl_name = match entries
                .keys()
                .map(|k| k.as_str())
                .filter(|k| k.ends_with("data.pkl"))
                .min()
            {
                Some(name) => name.to_string(),
                None => {
                    return Ok(Self { tensors });
                }
            };
            let prefix = pkl_name
                .strip_suffix("data.pkl")
                .unwrap_or_default()
                .to_string();
            let pkl_data = &entries[&pkl_name];
            for entry in parse_pickle_state_dict(pkl_data, &entries, &prefix) {
                tensors.insert(entry.name.clone(), entry);
            }
        } else {
            // Pre-1.6 torch.save streams: raw pickle with no zip container.
            for entry in parse_pickle_state_dict(bytes, &HashMap::new(), "") {
                tensors.insert(entry.name.clone(), entry);
            }
        }

        Ok(Self { tensors })
    }

    /// List all tensor names found in the PyTorch file.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

impl TensorProvider for PthProvider {
    fn get(&self, name: &str) -> Result<RawTensor> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Backend(format!("tensor not found in pth: {name}")))?;

        Ok(RawTensor {
            bytes: entry.data.clone(),
            shape: entry.shape.clone(),
            dtype: entry.dtype.clone(),
            provenance: QuantProvenance::default(),
        })
    }

    fn meta(&self, name: &str) -> Result<TensorMeta> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Backend(format!("tensor not found in pth: {name}")))?;

        Ok(TensorMeta {
            dtype: entry.dtype.clone(),
            provenance: QuantProvenance::default(),
            shape: entry.shape.clone(),
            fusion_mask: 0,
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }
}

// ---------------------------------------------------------------------------
// ZIP container (stored entries only — what torch.save / torch.jit emit)
// ---------------------------------------------------------------------------

fn parse_zip_entries(bytes: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
    // Primary path: central directory (authoritative sizes, required for
    // torch's data-descriptor archives where local header sizes are zero).
    if let Some(map) = parse_central_directory(bytes) {
        return Ok(map);
    }
    // Fallback: streamed/truncated containers with no EOCD — walk local
    // headers sequentially (only valid when sizes are inline, i.e. no
    // data-descriptor flag).
    Ok(parse_local_headers(bytes))
}

fn parse_local_headers(bytes: &[u8]) -> HashMap<String, Vec<u8>> {
    let mut map = HashMap::new();
    let mut offset = 0;

    while offset + 30 <= bytes.len() {
        if &bytes[offset..offset + 4] != b"PK\x03\x04" {
            break;
        }
        let flags = u16::from_le_bytes([bytes[offset + 6], bytes[offset + 7]]);
        let comp_method = u16::from_le_bytes([bytes[offset + 8], bytes[offset + 9]]);
        let comp_size = u32::from_le_bytes([
            bytes[offset + 18],
            bytes[offset + 19],
            bytes[offset + 20],
            bytes[offset + 21],
        ]) as usize;
        let name_len = u16::from_le_bytes([bytes[offset + 26], bytes[offset + 27]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[offset + 28], bytes[offset + 29]]) as usize;

        let name_start = offset + 30;
        if name_start + name_len > bytes.len() {
            break;
        }
        let filename =
            String::from_utf8_lossy(&bytes[name_start..name_start + name_len]).to_string();
        let data_start = name_start + name_len + extra_len;

        if flags & 0x08 != 0 || comp_method != 0 || data_start + comp_size > bytes.len() {
            break;
        }

        if !filename.ends_with('/') {
            map.insert(filename, bytes[data_start..data_start + comp_size].to_vec());
        }
        offset = data_start + comp_size;
    }

    map
}

fn parse_central_directory(bytes: &[u8]) -> Option<HashMap<String, Vec<u8>>> {
    let mut map = HashMap::new();

    let eocd = find_u32_le(bytes, 0x0605_4b50)?;
    if eocd + 22 > bytes.len() {
        return None;
    }
    let cd_count = u16::from_le_bytes([bytes[eocd + 10], bytes[eocd + 11]]) as usize;
    let cd_offset = u32::from_le_bytes([
        bytes[eocd + 16],
        bytes[eocd + 17],
        bytes[eocd + 18],
        bytes[eocd + 19],
    ]) as usize;

    let mut pos = cd_offset;
    for _ in 0..cd_count {
        if pos + 46 > bytes.len() || &bytes[pos..pos + 4] != b"PK\x01\x02" {
            break;
        }
        let method = u16::from_le_bytes([bytes[pos + 10], bytes[pos + 11]]) as usize;
        let comp_size = u32::from_le_bytes([
            bytes[pos + 20],
            bytes[pos + 21],
            bytes[pos + 22],
            bytes[pos + 23],
        ]) as usize;
        let name_len = u16::from_le_bytes([bytes[pos + 28], bytes[pos + 29]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[pos + 30], bytes[pos + 31]]) as usize;
        let comment_len = u16::from_le_bytes([bytes[pos + 32], bytes[pos + 33]]) as usize;
        let local_off = u32::from_le_bytes([
            bytes[pos + 42],
            bytes[pos + 43],
            bytes[pos + 44],
            bytes[pos + 45],
        ]) as usize;
        let name_start = pos + 46;
        if name_start + name_len > bytes.len() {
            break;
        }
        let filename =
            String::from_utf8_lossy(&bytes[name_start..name_start + name_len]).to_string();

        // Data start needs the *local* header's name/extra lengths, which can
        // differ from the central copies.
        let lh = local_off;
        if lh + 30 <= bytes.len() && &bytes[lh..lh + 4] == b"PK\x03\x04" {
            let l_name_len = u16::from_le_bytes([bytes[lh + 26], bytes[lh + 27]]) as usize;
            let l_extra_len = u16::from_le_bytes([bytes[lh + 28], bytes[lh + 29]]) as usize;
            let data_start = lh + 30 + l_name_len + l_extra_len;
            if method == 0 && !filename.ends_with('/') && data_start + comp_size <= bytes.len() {
                map.insert(filename, bytes[data_start..data_start + comp_size].to_vec());
            }
        }
        pos = name_start + name_len + extra_len + comment_len;
    }

    Some(map)
}

/// Scan backwards for the last occurrence of a little-endian u32 signature.
fn find_u32_le(bytes: &[u8], sig: u32) -> Option<usize> {
    let pat = sig.to_le_bytes();
    if bytes.len() < 4 {
        return None;
    }
    let start = bytes.len().saturating_sub(66_000);
    let mut i = bytes.len() - 4;
    loop {
        if &bytes[i..i + 4] == pat {
            return Some(i);
        }
        if i == 0 || i < start {
            return None;
        }
        i -= 1;
    }
}

// ---------------------------------------------------------------------------
// Pickle stack VM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Val {
    Mark,
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Global { module: String, name: String },
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Storage(StorageRef),
    Tensor(TensorVal),
    Opaque,
}

#[derive(Debug, Clone)]
struct StorageRef {
    key: String,
    dtype_key: String,
}

#[derive(Debug, Clone)]
struct TensorVal {
    storage: StorageRef,
    offset: usize,
    size: Vec<i64>,
    stride: Vec<i64>,
}

struct Vm<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Val>,
    memo: HashMap<usize, Val>,
}

impl<'a> Vm<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(Error::Backend(format!(
                "truncated pickle: need {n} bytes at {}",
                self.pos
            )));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32le(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64le(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(Error::Backend("unterminated pickle string".into()));
        }
        let s = String::from_utf8_lossy(&self.data[start..self.pos]).to_string();
        self.pos += 1; // consume '\n'
        Ok(s)
    }

    /// Pop values until the nearest `Mark`, mark included.
    fn pop_mark(&mut self) -> Vec<Val> {
        let mut items = Vec::new();
        while let Some(v) = self.stack.pop() {
            if matches!(v, Val::Mark) {
                items.reverse();
                return items;
            }
            items.push(v);
        }
        items.reverse();
        items
    }

    fn run(mut self) -> Result<Val> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    // PROTO
                    self.u8()?;
                }
                0x95 => {
                    // FRAME
                    self.u64le()?;
                }
                0x28 => self.stack.push(Val::Mark), // MARK
                0x2e => {
                    // STOP
                    return Ok(self.stack.pop().unwrap_or(Val::None));
                }
                0x29 => self.stack.push(Val::Tuple(Vec::new())), // EMPTY_TUPLE
                0x5d => self.stack.push(Val::List(Vec::new())),  // EMPTY_LIST
                0x7d => self.stack.push(Val::Dict(Vec::new())),  // EMPTY_DICT
                0x74 => {
                    // TUPLE (mark)
                    let items = self.pop_mark();
                    self.stack.push(Val::Tuple(items));
                }
                0x85 => {
                    // TUPLE1
                    let a = self.stack.pop().unwrap_or(Val::None);
                    self.stack.push(Val::Tuple(vec![a]));
                }
                0x86 => {
                    // TUPLE2
                    let (b, a) = (self.stack.pop(), self.stack.pop());
                    self.stack.push(Val::Tuple(vec![
                        a.unwrap_or(Val::None),
                        b.unwrap_or(Val::None),
                    ]));
                }
                0x87 => {
                    // TUPLE3
                    let (c, b, a) = (self.stack.pop(), self.stack.pop(), self.stack.pop());
                    self.stack.push(Val::Tuple(vec![
                        a.unwrap_or(Val::None),
                        b.unwrap_or(Val::None),
                        c.unwrap_or(Val::None),
                    ]));
                }
                0x61 => {
                    // APPEND
                    let item = self.stack.pop().unwrap_or(Val::None);
                    let list = self.stack.pop().unwrap_or(Val::List(Vec::new()));
                    if let Val::List(mut l) = list {
                        l.push(item);
                        self.stack.push(Val::List(l));
                    } else {
                        self.stack.push(Val::Opaque);
                    }
                }
                0x65 => {
                    // APPENDS
                    let items = self.pop_mark();
                    let list = self.stack.pop().unwrap_or(Val::List(Vec::new()));
                    if let Val::List(mut l) = list {
                        l.extend(items);
                        self.stack.push(Val::List(l));
                    } else {
                        self.stack.push(Val::Opaque);
                    }
                }
                0x75 => {
                    // SETITEMS
                    let items = self.pop_mark();
                    let dict = self.stack.pop().unwrap_or(Val::Dict(Vec::new()));
                    if let Val::Dict(mut d) = dict {
                        for pair in items.chunks(2) {
                            if pair.len() == 2 {
                                d.push((pair[0].clone(), pair[1].clone()));
                            }
                        }
                        self.stack.push(Val::Dict(d));
                    } else {
                        self.stack.push(Val::Opaque);
                    }
                }
                0x73 => {
                    // SETITEM
                    let (v, k, d) = (self.stack.pop(), self.stack.pop(), self.stack.pop());
                    if let Val::Dict(mut dd) = d.unwrap_or(Val::Dict(Vec::new())) {
                        if let (Some(key), Some(val)) = (k, v) {
                            dd.push((key, val));
                        }
                        self.stack.push(Val::Dict(dd));
                    } else {
                        self.stack.push(Val::Opaque);
                    }
                }
                0x63 => {
                    // GLOBAL: module\n name\n
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.stack.push(Val::Global { module, name });
                }
                0x93 => {
                    // STACK_GLOBAL
                    let name = self.stack.pop().unwrap_or(Val::None);
                    let module = self.stack.pop().unwrap_or(Val::None);
                    let name = match name {
                        Val::Str(s) => s,
                        _ => String::new(),
                    };
                    let module = match module {
                        Val::Str(s) => s,
                        _ => String::new(),
                    };
                    self.stack.push(Val::Global { module, name });
                }
                0x52 => {
                    // REDUCE
                    let args = self.stack.pop().unwrap_or(Val::None);
                    let func = self.stack.pop().unwrap_or(Val::Opaque);
                    let out = reduce(func, args);
                    self.stack.push(out);
                }
                0x81 | 0x84 => {
                    // NEWOBJ / NEWOBJ_EX
                    let args = if op == 0x84 {
                        let kw = self.stack.pop();
                        drop(kw);
                        self.stack.pop()
                    } else {
                        self.stack.pop()
                    };
                    let _cls = self.stack.pop();
                    let obj = match args {
                        Some(Val::Tuple(items)) => Val::Dict(
                            items
                                .chunks(2)
                                .filter(|p| p.len() == 2)
                                .map(|p| (p[0].clone(), p[1].clone()))
                                .collect(),
                        ),
                        _ => Val::Opaque,
                    };
                    self.stack.push(obj);
                }
                0x62 => {
                    // BUILD
                    let state = self.stack.pop().unwrap_or(Val::None);
                    let obj = self.stack.pop().unwrap_or(Val::Opaque);
                    let merged = match (obj, state) {
                        (Val::Dict(mut d), Val::Dict(s)) => {
                            d.extend(s);
                            Val::Dict(d)
                        }
                        (o, _) => o,
                    };
                    self.stack.push(merged);
                }
                0x51 => {
                    // BINPERSID
                    let pid = self.stack.pop().unwrap_or(Val::None);
                    self.stack.push(persistent_load(pid));
                }
                0x58 => {
                    // BINUNICODE
                    let len = self.u32le()? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(Val::Str(String::from_utf8_lossy(raw).to_string()));
                }
                0x8c => {
                    // SHORT_BINUNICODE
                    let len = self.u8()? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(Val::Str(String::from_utf8_lossy(raw).to_string()));
                }
                0x8d => {
                    // BINUNICODE8
                    let len = self.u64le()? as usize;
                    let raw = self.take(len)?;
                    self.stack
                        .push(Val::Str(String::from_utf8_lossy(raw).to_string()));
                }
                0x4a => {
                    // BININT
                    let v = self.u32le()? as i32;
                    self.stack.push(Val::Int(v as i64));
                }
                0x4b => {
                    // BININT1
                    let byte = self.u8()?;
                    self.stack.push(Val::Int(byte as i64));
                }
                0x4d => {
                    // BININT2
                    let b = self.take(2)?;
                    self.stack
                        .push(Val::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
                }
                0x4c => {
                    // LONG: decimal digits terminated by '\n'
                    let s = self.read_line()?;
                    let digits = s.trim_end_matches('L');
                    self.stack
                        .push(Val::Int(digits.parse::<i64>().unwrap_or(0)));
                }
                0x4e => self.stack.push(Val::None),
                0x88 => self.stack.push(Val::Bool(true)),
                0x89 => self.stack.push(Val::Bool(false)),
                0x42 => {
                    // BINBYTES
                    let len = self.u32le()? as usize;
                    let raw = self.take(len)?.to_vec();
                    self.stack.push(Val::Opaque);
                    let _ = raw;
                }
                0x43 => {
                    // SHORT_BINBYTES
                    let len = self.u8()? as usize;
                    self.take(len)?;
                    self.stack.push(Val::Opaque);
                }
                0x68 => {
                    // BINGET
                    let k = self.u8()? as usize;
                    if let Some(v) = self.memo.get(&k) {
                        self.stack.push(v.clone());
                    }
                }
                0x6a => {
                    // LONG_BINGET
                    let k = self.u32le()? as usize;
                    if let Some(v) = self.memo.get(&k) {
                        self.stack.push(v.clone());
                    }
                }
                0x71 => {
                    // BINPUT
                    let k = self.u8()? as usize;
                    if let Some(top) = self.stack.last() {
                        self.memo.insert(k, top.clone());
                    }
                }
                0x72 => {
                    // LONG_BINPUT
                    let k = self.u32le()? as usize;
                    if let Some(top) = self.stack.last() {
                        self.memo.insert(k, top.clone());
                    }
                }
                0x94 => {
                    // MEMOIZE
                    if let Some(top) = self.stack.last() {
                        let k = self.memo.len();
                        self.memo.insert(k, top.clone());
                    }
                }
                0x30 => {
                    self.stack.pop();
                }
                0x31 => {
                    self.pop_mark();
                }
                other => {
                    return Err(Error::Backend(format!(
                        "unsupported pickle opcode 0x{other:02x} at offset {}",
                        self.pos - 1
                    )));
                }
            }
        }
    }
}

fn reduce(func: Val, args: Val) -> Val {
    let (module, name) = match &func {
        Val::Global { module, name } => (module.as_str(), name.as_str()),
        _ => return Val::Opaque,
    };
    let items = match args {
        Val::Tuple(items) => items,
        other => {
            return match (module, name) {
                ("collections", "OrderedDict") => Val::Dict(Vec::new()),
                _ => {
                    let _ = other;
                    Val::Opaque
                }
            };
        }
    };

    match (module, name) {
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            // (storage, storage_offset, size, stride, requires_grad,
            //  backward_hooks[, metadata])
            if items.len() < 4 {
                return Val::Opaque;
            }
            let storage = match &items[0] {
                Val::Storage(s) => s.clone(),
                _ => return Val::Opaque,
            };
            let offset = match &items[1] {
                Val::Int(i) => (*i).max(0) as usize,
                _ => return Val::Opaque,
            };
            let size = ints_of(&items[2]);
            let stride = ints_of(&items[3]);
            if size.is_empty() || stride.len() != size.len() {
                return Val::Opaque;
            }
            Val::Tensor(TensorVal {
                storage,
                offset,
                size,
                stride,
            })
        }
        ("torch._utils", "_rebuild_parameter") => {
            // (tensor, requires_grad, backward_hooks)
            items.into_iter().next().unwrap_or(Val::Opaque)
        }
        ("collections", "OrderedDict") => Val::Dict(Vec::new()),
        _ => Val::Opaque,
    }
}

fn persistent_load(pid: Val) -> Val {
    // ('storage', <storage type>, key, location)
    if let Val::Tuple(items) = pid {
        if items.len() >= 3 {
            let tag_ok = matches!(&items[0], Val::Str(s) if s == "storage");
            let dtype_key = match &items[1] {
                Val::Global { name, .. } => name.clone(),
                Val::Str(s) => s.clone(),
                _ => String::new(),
            };
            let key = match &items[2] {
                Val::Str(s) => s.clone(),
                Val::Int(i) => i.to_string(),
                _ => String::new(),
            };
            if tag_ok && !key.is_empty() {
                return Val::Storage(StorageRef { key, dtype_key });
            }
        }
    }
    Val::Opaque
}

fn ints_of(v: &Val) -> Vec<i64> {
    match v {
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .map(|x| match x {
                Val::Int(i) => *i,
                _ => 0,
            })
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tensor extraction from the unpickled object graph
// ---------------------------------------------------------------------------

fn elem_size_and_dtype(dtype_key: &str) -> Option<(usize, DType)> {
    let native = |arith: ArithType| DType {
        arith,
        storage: Storage::Native,
    };
    match dtype_key {
        "FloatStorage" => Some((4, DType::F32)),
        "HalfStorage" => Some((2, DType::F16)),
        "BFloat16Storage" => Some((2, DType::BF16)),
        "LongStorage" => Some((8, native(ArithType::I64))),
        "ByteStorage" | "BoolStorage" => Some((1, native(ArithType::U8))),
        // Int/Short/Char/Double have no matching grim ArithType yet; skipping
        // beats emitting wrongly-typed bytes.
        _ => None,
    }
}

fn collect_tensors(val: &Val, prefix: &str, out: &mut Vec<(String, TensorVal)>) {
    match val {
        Val::Dict(pairs) => {
            for (k, v) in pairs {
                let key = match k {
                    Val::Str(s) => s.clone(),
                    Val::Int(i) => i.to_string(),
                    _ => continue,
                };
                collect_tensors(v, &format!("{prefix}{key}."), out);
            }
        }
        Val::List(items) => {
            for (i, v) in items.iter().enumerate() {
                collect_tensors(v, &format!("{prefix}{i}."), out);
            }
        }
        Val::Tensor(t) => {
            let name = prefix.strip_suffix('.').unwrap_or(prefix).to_string();
            out.push((name, t.clone()));
        }
        _ => {}
    }
}

/// Interpret the pickle stream into concrete tensor entries.
fn parse_pickle_state_dict(
    pkl: &[u8],
    zip_entries: &HashMap<String, Vec<u8>>,
    data_prefix: &str,
) -> Vec<TorchTensorEntry> {
    let root = match Vm::new(pkl).run() {
        Ok(root) => root,
        Err(e) => {
            eprintln!("[grim-format] torch pickle parse failed: {e}");
            return Vec::new();
        }
    };

    let mut refs = Vec::new();
    collect_tensors(&root, "", &mut refs);

    let mut results = Vec::new();
    for (name, t) in refs {
        let Some((esz, dtype)) = elem_size_and_dtype(&t.storage.dtype_key) else {
            eprintln!(
                "[grim-format] skipping tensor '{name}': unsupported storage type {}",
                t.storage.dtype_key
            );
            continue;
        };
        // Require C-contiguous layout before slicing raw storage bytes.
        let mut expected = 1i64;
        let mut contiguous = true;
        for (dim, st) in t.size.iter().zip(t.stride.iter()).rev() {
            if *st != expected {
                contiguous = false;
                break;
            }
            expected *= (*dim).max(1);
        }
        if !contiguous {
            eprintln!("[grim-format] skipping tensor '{name}': non-contiguous stride");
            continue;
        }
        let count: usize = t.size.iter().map(|d| (*d).max(0) as usize).product();
        let data_path = format!("{data_prefix}data/{}", t.storage.key);
        let Some(storage_bytes) = zip_entries.get(&data_path) else {
            eprintln!("[grim-format] skipping tensor '{name}': missing storage {data_path}");
            continue;
        };
        let start = t.offset * esz;
        let end = start + count * esz;
        if end > storage_bytes.len() {
            eprintln!("[grim-format] skipping tensor '{name}': storage overrun");
            continue;
        }
        results.push(TorchTensorEntry {
            name,
            shape: t.size.iter().map(|d| (*d).max(0) as usize).collect(),
            dtype,
            data: storage_bytes[start..end].to_vec(),
        });
    }
    results
}
