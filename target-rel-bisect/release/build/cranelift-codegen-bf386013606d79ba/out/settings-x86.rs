#[derive(Clone, PartialEq, Hash)] // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:397
/// Flags group `x86`.
pub struct Flags {
    bytes: [u8; 3], // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:400
}
impl Flags {
    /// Create flags x86 settings group.
    #[allow(unused_variables, reason = "generated code")] // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:24
    pub fn new(shared: &settings::Flags, builder: &Builder) -> Self {
        let bvec = builder.state_for("x86"); // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:29
        let mut x86 = Self { bytes: [0; 3] }; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:30
        debug_assert_eq!(bvec.len(), 3); // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:36
        x86.bytes[0..3].copy_from_slice(&bvec); // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:41
        x86 // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:48
    }
}
impl Flags {
    /// Iterates the setting values.
    pub fn iter(&self) -> impl Iterator<Item = Value> + use<> {
        let mut bytes = [0; 3]; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:58
        bytes.copy_from_slice(&self.bytes[0..3]); // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:59
        DESCRIPTORS.iter().filter_map(move |d| {
            let values = match &d.detail {
                detail::Detail::Preset => return None, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:62
                detail::Detail::Enum { last, enumerators } => Some(TEMPLATE.enums(*last, *enumerators)), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:63
                _ => None // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:64
            }
            ; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:66
            Some(Value { name: d.name, detail: d.detail, values, value: bytes[d.offset as usize] }) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:67
        }
        ) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:69
    }
}
/// User-defined settings.
#[allow(dead_code, reason = "generated code")] // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:183
impl Flags {
    /// Dynamic numbered predicate getter.
    fn numbered_predicate(&self, p: usize) -> bool {
        self.bytes[0 + p / 8] & (1 << (p % 8)) != 0 // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:188
    }
    /// Has support for SSE3.
    /// SSE3: CPUID.01H:ECX.SSE3[bit 0]
    pub fn has_sse3(&self) -> bool {
        self.numbered_predicate(0) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for SSSE3.
    /// SSSE3: CPUID.01H:ECX.SSSE3[bit 9]
    pub fn has_ssse3(&self) -> bool {
        self.numbered_predicate(1) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for CMPXCHG16b.
    /// CMPXCHG16b: CPUID.01H:ECX.CMPXCHG16B[bit 13]
    pub fn has_cmpxchg16b(&self) -> bool {
        self.numbered_predicate(2) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for SSE4.1.
    /// SSE4.1: CPUID.01H:ECX.SSE4_1[bit 19]
    pub fn has_sse41(&self) -> bool {
        self.numbered_predicate(3) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for SSE4.2.
    /// SSE4.2: CPUID.01H:ECX.SSE4_2[bit 20]
    pub fn has_sse42(&self) -> bool {
        self.numbered_predicate(4) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for AVX.
    /// AVX: CPUID.01H:ECX.AVX[bit 28]
    pub fn has_avx(&self) -> bool {
        self.numbered_predicate(5) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for AVX2.
    /// AVX2: CPUID.07H:EBX.AVX2[bit 5]
    pub fn has_avx2(&self) -> bool {
        self.numbered_predicate(6) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for FMA.
    /// FMA: CPUID.01H:ECX.FMA[bit 12]
    pub fn has_fma(&self) -> bool {
        self.numbered_predicate(7) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for AVX512BITALG.
    /// AVX512BITALG: CPUID.07H:ECX.AVX512BITALG[bit 12]
    pub fn has_avx512bitalg(&self) -> bool {
        self.numbered_predicate(8) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for AVX512DQ.
    /// AVX512DQ: CPUID.07H:EBX.AVX512DQ[bit 17]
    pub fn has_avx512dq(&self) -> bool {
        self.numbered_predicate(9) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for AVX512VL.
    /// AVX512VL: CPUID.07H:EBX.AVX512VL[bit 31]
    pub fn has_avx512vl(&self) -> bool {
        self.numbered_predicate(10) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for AVX512VMBI.
    /// AVX512VBMI: CPUID.07H:ECX.AVX512VBMI[bit 1]
    pub fn has_avx512vbmi(&self) -> bool {
        self.numbered_predicate(11) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for AVX512F.
    /// AVX512F: CPUID.07H:EBX.AVX512F[bit 16]
    pub fn has_avx512f(&self) -> bool {
        self.numbered_predicate(12) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for POPCNT.
    /// POPCNT: CPUID.01H:ECX.POPCNT[bit 23]
    pub fn has_popcnt(&self) -> bool {
        self.numbered_predicate(13) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for BMI1.
    /// BMI1: CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]
    pub fn has_bmi1(&self) -> bool {
        self.numbered_predicate(14) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for BMI2.
    /// BMI2: CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]
    pub fn has_bmi2(&self) -> bool {
        self.numbered_predicate(15) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
    /// Has support for LZCNT.
    /// LZCNT: CPUID.EAX=80000001H:ECX.LZCNT[bit 5]
    pub fn has_lzcnt(&self) -> bool {
        self.numbered_predicate(16) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:155
    }
}
static DESCRIPTORS: [detail::Descriptor; 84] = [ // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:224
    detail::Descriptor {
        name: "has_sse3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for SSE3.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 0 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_ssse3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for SSSE3.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 1 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_cmpxchg16b", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for CMPXCHG16b.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 2 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_sse41", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for SSE4.1.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 3 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_sse42", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for SSE4.2.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 4 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_avx", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for AVX.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 5 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_avx2", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for AVX2.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 6 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_fma", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for FMA.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 7 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_avx512bitalg", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for AVX512BITALG.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 0 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_avx512dq", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for AVX512DQ.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 1 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_avx512vl", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for AVX512VL.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 2 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_avx512vbmi", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for AVX512VMBI.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 3 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_avx512f", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for AVX512F.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 4 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_popcnt", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for POPCNT.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 5 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_bmi1", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for BMI1.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 6 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_bmi2", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for BMI2.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 7 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "has_lzcnt", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:232
        description: "Has support for LZCNT.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:233
        offset: 2, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:234
        detail: detail::Detail::Bool { bit: 0 }, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:237
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:259
    detail::Descriptor {
        name: "sse3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "SSE3 and earlier.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "ssse3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "SSSE3 and earlier.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 3, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "sse41", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "SSE4.1 and earlier.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 6, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "sse42", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "SSE4.2 and earlier.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 9, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "baseline", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "A baseline preset with no extensions enabled.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 12, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "nocona", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Nocona microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 15, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "core2", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Core 2 microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 18, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "penryn", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Penryn microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 21, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "atom", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Atom microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 24, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "bonnell", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Bonnell microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 27, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "silvermont", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Silvermont microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 30, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "slm", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Silvermont microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 33, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "goldmont", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Goldmont microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 36, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "goldmont-plus", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Goldmont Plus microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 39, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "tremont", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Tremont microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 42, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "alderlake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Alderlake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 45, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "sierraforest", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Sierra Forest microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 48, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "grandridge", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Grandridge microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 51, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "nehalem", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Nehalem microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 54, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "corei7", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Core i7 microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 57, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "westmere", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Westmere microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 60, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "sandybridge", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Sandy Bridge microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 63, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "corei7-avx", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Core i7 AVX microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 66, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "ivybridge", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Ivy Bridge microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 69, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "core-avx-i", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Intel Core CPU with 64-bit extensions.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 72, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "haswell", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Haswell microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 75, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "core-avx2", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Intel Core CPU with AVX2 extensions.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 78, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "broadwell", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Broadwell microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 81, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "skylake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Skylake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 84, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "knl", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Knights Landing microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 87, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "knm", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Knights Mill microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 90, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "skylake-avx512", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Skylake AVX512 microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 93, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "skx", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Skylake AVX512 microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 96, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "cascadelake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Cascade Lake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 99, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "cooperlake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Cooper Lake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 102, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "cannonlake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Canon Lake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 105, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "icelake-client", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Ice Lake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 108, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "icelake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Ice Lake microarchitecture", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 111, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "icelake-server", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Ice Lake (server) microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 114, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "tigerlake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Tiger Lake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 117, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "sapphirerapids", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Sapphire Rapids microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 120, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "raptorlake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Raptor Lake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 123, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "meteorlake", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Meteor Lake microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 126, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "graniterapids", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Granite Rapids microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 129, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "opteron", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Opteron microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 132, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "k8", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "K8 Hammer microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 135, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "athlon64", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Athlon64 microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 138, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "athlon-fx", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Athlon FX microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 141, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "opteron-sse3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Opteron microarchitecture with support for SSE3 instructions.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 144, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "k8-sse3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "K8 Hammer microarchitecture with support for SSE3 instructions.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 147, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "athlon64-sse3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Athlon 64 microarchitecture with support for SSE3 instructions.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 150, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "barcelona", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Barcelona microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 153, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "amdfam10", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "AMD Family 10h microarchitecture", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 156, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "btver1", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Bobcat microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 159, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "btver2", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Jaguar microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 162, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "bdver1", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Bulldozer microarchitecture", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 165, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "bdver2", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Piledriver microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 168, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "bdver3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Steamroller microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 171, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "bdver4", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Excavator microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 174, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "znver1", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Zen (first generation) microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 177, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "znver2", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Zen (second generation) microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 180, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "znver3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Zen (third generation) microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 183, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "znver4", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Zen (fourth generation) microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 186, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "x86-64", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Generic x86-64 microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 189, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "x86-64-v2", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Generic x86-64 (V2) microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 192, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "x86-64-v3", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Generic x86-64 (V3) microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 195, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
    detail::Descriptor {
        name: "x86-64-v4", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:264
        description: "Generic x86-64 (V4) microarchitecture.", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:265
        offset: 198, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:266
        detail: detail::Detail::Preset, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:267
    }
    , // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:269
]; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:275
static ENUMERATORS: [&str; 0] = [ // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:278
]; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:284
static HASH_TABLE: [u16; 128] = [ // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:294
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    78, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    77, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    76, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    24, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    79, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    67, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    81, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    23, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    51, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    60, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    15, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    14, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    30, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    1, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    42, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    71, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    68, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    5, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    36, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    66, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    6, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    45, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    22, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    65, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    16, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    7, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    48, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    50, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    25, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    63, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    83, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    12, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    44, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    39, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    53, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    70, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    4, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    32, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    82, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    3, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    59, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    11, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    13, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    31, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    80, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    74, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    40, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    29, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    47, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    46, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    9, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    55, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    72, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    10, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    75, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    73, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    2, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    62, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    34, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    8, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    19, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    20, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    49, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    17, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    54, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    61, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    21, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    64, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    69, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    57, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    27, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    28, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    35, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    37, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    41, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    43, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    33, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    58, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    52, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    18, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    56, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    0xffff, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:306
    26, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
    38, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:298
]; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:310
static PRESETS: [(u8, u8); 201] = [ // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:313
    // sse3: has_sse3
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // ssse3: has_sse3, has_ssse3
    (0b00000011, 0b00000011), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // sse41: has_sse3, has_ssse3, has_sse41
    (0b00001011, 0b00001011), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // sse42: has_sse3, has_ssse3, has_sse41, has_sse42
    (0b00011011, 0b00011011), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // baseline: 
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // nocona: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // core2: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // penryn: has_sse3, has_ssse3, has_sse41, has_cmpxchg16b
    (0b00001111, 0b00001111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // atom: has_sse3, has_ssse3, has_cmpxchg16b
    (0b00000111, 0b00000111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // bonnell: has_sse3, has_ssse3, has_cmpxchg16b
    (0b00000111, 0b00000111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // silvermont: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // slm: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // goldmont: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // goldmont-plus: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // tremont: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // alderlake: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // sierraforest: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // grandridge: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // nehalem: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // corei7: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // westmere: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // sandybridge: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx
    (0b00111111, 0b00111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // corei7-avx: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx
    (0b00111111, 0b00111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // ivybridge: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx
    (0b00111111, 0b00111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // core-avx-i: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx
    (0b00111111, 0b00111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // haswell: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // core-avx2: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // broadwell: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // skylake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // knl: has_popcnt, has_avx512f, has_fma, has_bmi1, has_bmi2, has_lzcnt, has_cmpxchg16b
    (0b10000100, 0b10000100), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11110000, 0b11110000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // knm: has_popcnt, has_avx512f, has_fma, has_bmi1, has_bmi2, has_lzcnt, has_cmpxchg16b
    (0b10000100, 0b10000100), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11110000, 0b11110000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // skylake-avx512: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11110110, 0b11110110), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // skx: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11110110, 0b11110110), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // cascadelake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11110110, 0b11110110), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // cooperlake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11110110, 0b11110110), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // cannonlake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11111110, 0b11111110), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // icelake-client: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // icelake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // icelake-server: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // tigerlake: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // sapphirerapids: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // raptorlake: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // meteorlake: has_sse3, has_ssse3, has_cmpxchg16b, has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // graniterapids: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_avx, has_avx2, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx512f, has_avx512dq, has_avx512vl, has_avx512vbmi, has_avx512bitalg
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // opteron: 
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // k8: 
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // athlon64: 
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // athlon-fx: 
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // opteron-sse3: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // k8-sse3: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // athlon64-sse3: has_sse3, has_cmpxchg16b
    (0b00000101, 0b00000101), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // barcelona: has_popcnt, has_lzcnt, has_cmpxchg16b
    (0b00000100, 0b00000100), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // amdfam10: has_popcnt, has_lzcnt, has_cmpxchg16b
    (0b00000100, 0b00000100), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // btver1: has_sse3, has_ssse3, has_lzcnt, has_popcnt, has_cmpxchg16b
    (0b00000111, 0b00000111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // btver2: has_sse3, has_ssse3, has_lzcnt, has_popcnt, has_cmpxchg16b, has_avx, has_bmi1
    (0b00100111, 0b00100111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b01100000, 0b01100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // bdver1: has_lzcnt, has_popcnt, has_sse3, has_ssse3, has_cmpxchg16b
    (0b00000111, 0b00000111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // bdver2: has_lzcnt, has_popcnt, has_sse3, has_ssse3, has_cmpxchg16b, has_bmi1
    (0b00000111, 0b00000111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b01100000, 0b01100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // bdver3: has_lzcnt, has_popcnt, has_sse3, has_ssse3, has_cmpxchg16b, has_bmi1
    (0b00000111, 0b00000111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b01100000, 0b01100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // bdver4: has_lzcnt, has_popcnt, has_sse3, has_ssse3, has_cmpxchg16b, has_bmi1, has_avx2, has_bmi2
    (0b01000111, 0b01000111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // znver1: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma, has_cmpxchg16b
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // znver2: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma, has_cmpxchg16b
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // znver3: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma, has_cmpxchg16b
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // znver4: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_bmi1, has_bmi2, has_lzcnt, has_fma, has_cmpxchg16b, has_avx512bitalg, has_avx512dq, has_avx512f, has_avx512vbmi, has_avx512vl
    (0b10011111, 0b10011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11111111, 0b11111111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // x86-64: 
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // x86-64-v2: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b
    (0b00011111, 0b00011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00100000, 0b00100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000000, 0b00000000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // x86-64-v3: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx2
    (0b11011111, 0b11011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100000, 0b11100000), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    // x86-64-v4: has_sse3, has_ssse3, has_sse41, has_sse42, has_popcnt, has_cmpxchg16b, has_bmi1, has_bmi2, has_fma, has_lzcnt, has_avx2, has_avx512dq, has_avx512vl
    (0b11011111, 0b11011111), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b11100110, 0b11100110), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
    (0b00000001, 0b00000001), // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:326
]; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:330
static TEMPLATE: detail::Template = detail::Template {
    name: "x86", // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:345
    descriptors: &DESCRIPTORS, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:346
    enumerators: &ENUMERATORS, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:347
    hash_table: &HASH_TABLE, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:348
    defaults: &[0x00, 0x00, 0x00], // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:349
    presets: &PRESETS, // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:350
}
; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:353
/// Create a `settings::Builder` for the x86 settings group.
pub fn builder() -> Builder {
    Builder::new(&TEMPLATE) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:360
}
impl fmt::Display for Flags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[x86]")?; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:369
        for d in &DESCRIPTORS {
            if !d.detail.is_preset() {
                write!(f, "{} = ", d.name)?; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:372
                TEMPLATE.format_toml_value(d.detail, self.bytes[d.offset as usize], f)?; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:373
                writeln!(f)?; // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:377
            }
        }
        Ok(()) // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:380
    }
}
impl Flags {
    /// Get the flag values as raw bytes for hashing.
    pub fn hash_key(&self) -> &[u8] {
        &self.bytes // /home/nelson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cranelift-codegen-meta-0.134.3/src/gen_settings.rs:390
    }
}
