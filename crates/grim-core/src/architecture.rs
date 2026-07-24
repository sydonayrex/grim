//! Centralized model architecture enumeration and tensor naming registry for Grim.
//!
//! Provides `ModelArchitecture` matching llama.cpp specifications and a unified
//! `TensorNamingRegistry` for translating tensor names across GGUF, HuggingFace,
//! and Grim internal representations.

use std::collections::HashMap;
use crate::model::ModalityHint;

/// Comprehensive enumeration of model architectures supported by llama.cpp and Grim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    Llama,
    Llama4,
    Deci,
    Falcon,
    Baichuan,
    Grok,
    Gpt2,
    GptJ,
    GptNeoX,
    Mpt,
    Starcoder,
    Refact,
    Bert,
    ModernBert,
    NomicBert,
    NomicBertMoe,
    NeoBert,
    JinaBertV2,
    JinaBertV3,
    Eurobert,
    Bloom,
    StableLm,
    Qwen,
    Qwen2,
    Qwen2Moe,
    Qwen2Vl,
    Qwen3,
    Qwen3Moe,
    Qwen3Next,
    Qwen3Vl,
    Qwen3VlMoe,
    Qwen35,
    Qwen35Moe,
    Phi2,
    Phi3,
    PhiMoe,
    Plamo,
    Plamo2,
    Plamo3,
    Codeshell,
    Orion,
    InternLm2,
    MiniCpm,
    MiniCpm3,
    Gemma,
    Gemma2,
    Gemma3,
    Gemma3n,
    Gemma4,
    Gemma4Assistant,
    GemmaEmbedding,
    Starcoder2,
    Mamba,
    Mamba2,
    Jamba,
    FalconH1,
    Xverse,
    CommandR,
    Cohere2,
    Cohere2Moe,
    Dbrx,
    Olmo,
    Olmo2,
    Olmoe,
    OpenElm,
    Arctic,
    DeepSeek,
    DeepSeek2,
    DeepSeek2Ocr,
    DeepSeek32,
    DeepSeek4,
    ChatGlm,
    Glm4,
    Glm4Moe,
    GlmDsa,
    BitNet,
    T5,
    T5Encoder,
    Jais,
    Jais2,
    Nemotron,
    NemotronH,
    NemotronHMoe,
    Exaone,
    Exaone4,
    ExaoneMoe,
    Rwkv6,
    Rwkv6Qwen2,
    Rwkv7,
    ARwkv7,
    Granite,
    GraniteMoe,
    GraniteHybrid,
    Chameleon,
    WavTokenizerDec,
    Plm,
    BailingMoe,
    BailingMoe2,
    Dots1,
    Arcee,
    AfMoe,
    Laguna,
    Ernie45,
    Ernie45Moe,
    HunyuanMoe,
    HunyuanDense,
    HunyuanVl,
    HyV3,
    SmolLm3,
    OpenAiMoe,
    Lfm2,
    Lfm2Moe,
    Dream,
    SmallThinker,
    Llada,
    LladaMoe,
    SeedOss,
    GroveMoe,
    Apertus,
    MiniMaxM2,
    CogVlm,
    Rnd1,
    PanguEmbed,
    Mistral3,
    Mistral4,
    PaddleOcr,
    Mimo2,
    Step35,
    LlamaEmbed,
    MainCoder,
    KimiLinear,
    Talkie,
    Mellum,
    Eagle3,
    DFlash,
    Unknown,
}

impl ModelArchitecture {
    /// Parse string identifier into `ModelArchitecture` enum variant.
    /// Matches GGUF `general.architecture` strings and HuggingFace `model_type` values.
    pub fn from_str(s: &str) -> Self {
        let s_lower = s.to_lowercase();
        match s_lower.as_str() {
            "llama" => Self::Llama,
            "llama4" => Self::Llama4,
            "deci" => Self::Deci,
            "falcon" => Self::Falcon,
            "baichuan" => Self::Baichuan,
            "grok" => Self::Grok,
            "gpt2" => Self::Gpt2,
            "gptj" | "gpt-j" => Self::GptJ,
            "gptneox" | "gpt-neox" => Self::GptNeoX,
            "mpt" => Self::Mpt,
            "starcoder" => Self::Starcoder,
            "refact" => Self::Refact,
            "bert" => Self::Bert,
            "modern-bert" | "modernbert" => Self::ModernBert,
            "nomic-bert" | "nomic_bert" => Self::NomicBert,
            "nomic-bert-moe" => Self::NomicBertMoe,
            "neo-bert" | "neobert" => Self::NeoBert,
            "jina-bert-v2" => Self::JinaBertV2,
            "jina-bert-v3" => Self::JinaBertV3,
            "eurobert" => Self::Eurobert,
            "bloom" => Self::Bloom,
            "stablelm" => Self::StableLm,
            "qwen" => Self::Qwen,
            "qwen2" => Self::Qwen2,
            "qwen2moe" | "qwen2_moe" => Self::Qwen2Moe,
            "qwen2vl" | "qwen2_vl" => Self::Qwen2Vl,
            "qwen3" => Self::Qwen3,
            "qwen3moe" => Self::Qwen3Moe,
            "qwen3next" => Self::Qwen3Next,
            "qwen3vl" => Self::Qwen3Vl,
            "qwen3vlmoe" => Self::Qwen3VlMoe,
            "qwen35" | "qwen3.5" => Self::Qwen35,
            "qwen35moe" => Self::Qwen35Moe,
            "phi" | "phi2" | "phi-2" => Self::Phi2,
            "phi3" | "phi-3" | "phishort" => Self::Phi3,
            "phimoe" => Self::PhiMoe,
            "plamo" => Self::Plamo,
            "plamo2" => Self::Plamo2,
            "plamo3" => Self::Plamo3,
            "codeshell" => Self::Codeshell,
            "orion" => Self::Orion,
            "internlm2" | "internlm" => Self::InternLm2,
            "minicpm" => Self::MiniCpm,
            "minicpm3" => Self::MiniCpm3,
            "gemma" => Self::Gemma,
            "gemma2" => Self::Gemma2,
            "gemma3" => Self::Gemma3,
            "gemma3n" => Self::Gemma3n,
            "gemma4" => Self::Gemma4,
            "gemma4-assistant" => Self::Gemma4Assistant,
            "gemma-embedding" => Self::GemmaEmbedding,
            "starcoder2" => Self::Starcoder2,
            "mamba" => Self::Mamba,
            "mamba2" => Self::Mamba2,
            "jamba" => Self::Jamba,
            "falcon-h1" => Self::FalconH1,
            "xverse" => Self::Xverse,
            "command-r" => Self::CommandR,
            "cohere2" => Self::Cohere2,
            "cohere2moe" => Self::Cohere2Moe,
            "dbrx" => Self::Dbrx,
            "olmo" => Self::Olmo,
            "olmo2" => Self::Olmo2,
            "olmoe" => Self::Olmoe,
            "openelm" => Self::OpenElm,
            "arctic" => Self::Arctic,
            "deepseek" => Self::DeepSeek,
            "deepseek2" | "deepseek_v2" => Self::DeepSeek2,
            "deepseek2ocr" => Self::DeepSeek2Ocr,
            "deepseek32" => Self::DeepSeek32,
            "deepseek4" | "deepseek_v3" | "deepseek_r1" => Self::DeepSeek4,
            "chatglm" => Self::ChatGlm,
            "glm4" => Self::Glm4,
            "glm4moe" | "glm4_moe" => Self::Glm4Moe,
            "glm-dsa" => Self::GlmDsa,
            "bitnet" => Self::BitNet,
            "t5" => Self::T5,
            "t5encoder" => Self::T5Encoder,
            "jais" => Self::Jais,
            "jais2" => Self::Jais2,
            "nemotron" => Self::Nemotron,
            "nemotron-h" | "nemotron_h" => Self::NemotronH,
            "nemotron-h-moe" => Self::NemotronHMoe,
            "exaone" => Self::Exaone,
            "exaone4" => Self::Exaone4,
            "exaone-moe" => Self::ExaoneMoe,
            "rwkv6" | "rwkv" => Self::Rwkv6,
            "rwkv6qwen2" => Self::Rwkv6Qwen2,
            "rwkv7" => Self::Rwkv7,
            "arwkv7" => Self::ARwkv7,
            "granite" => Self::Granite,
            "granite-moe" => Self::GraniteMoe,
            "granite-hybrid" => Self::GraniteHybrid,
            "chameleon" => Self::Chameleon,
            "wavtokenizer-dec" => Self::WavTokenizerDec,
            "plm" => Self::Plm,
            "bailingmoe" => Self::BailingMoe,
            "bailingmoe2" => Self::BailingMoe2,
            "dots1" => Self::Dots1,
            "arcee" => Self::Arcee,
            "afmoe" => Self::AfMoe,
            "laguna" => Self::Laguna,
            "ernie4-5" => Self::Ernie45,
            "ernie4-5-moe" => Self::Ernie45Moe,
            "hunyuan-moe" => Self::HunyuanMoe,
            "hunyuan-dense" => Self::HunyuanDense,
            "hunyuan-vl" => Self::HunyuanVl,
            "hy-v3" => Self::HyV3,
            "smollm3" => Self::SmolLm3,
            "openai-moe" => Self::OpenAiMoe,
            "lfm2" | "liquid" => Self::Lfm2,
            "lfm2moe" => Self::Lfm2Moe,
            "dream" => Self::Dream,
            "smallthinker" => Self::SmallThinker,
            "llada" => Self::Llada,
            "llada-moe" => Self::LladaMoe,
            "seed-oss" => Self::SeedOss,
            "grovemoe" => Self::GroveMoe,
            "apertus" => Self::Apertus,
            "minimax-m2" => Self::MiniMaxM2,
            "cogvlm" => Self::CogVlm,
            "rnd1" => Self::Rnd1,
            "pangu-embed" => Self::PanguEmbed,
            "mistral3" => Self::Mistral3,
            "mistral4" => Self::Mistral4,
            "paddleocr" => Self::PaddleOcr,
            "mimo2" => Self::Mimo2,
            "step35" => Self::Step35,
            "llama-embed" => Self::LlamaEmbed,
            "maincoder" => Self::MainCoder,
            "kimi-linear" => Self::KimiLinear,
            "talkie" => Self::Talkie,
            "mellum" => Self::Mellum,
            "eagle3" => Self::Eagle3,
            "dflash" => Self::DFlash,
            _ => Self::Unknown,
        }
    }

    /// Return canonical string representation of architecture.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Llama4 => "llama4",
            Self::Deci => "deci",
            Self::Falcon => "falcon",
            Self::Baichuan => "baichuan",
            Self::Grok => "grok",
            Self::Gpt2 => "gpt2",
            Self::GptJ => "gptj",
            Self::GptNeoX => "gptneox",
            Self::Mpt => "mpt",
            Self::Starcoder => "starcoder",
            Self::Refact => "refact",
            Self::Bert => "bert",
            Self::ModernBert => "modern-bert",
            Self::NomicBert => "nomic-bert",
            Self::NomicBertMoe => "nomic-bert-moe",
            Self::NeoBert => "neo-bert",
            Self::JinaBertV2 => "jina-bert-v2",
            Self::JinaBertV3 => "jina-bert-v3",
            Self::Eurobert => "eurobert",
            Self::Bloom => "bloom",
            Self::StableLm => "stablelm",
            Self::Qwen => "qwen",
            Self::Qwen2 => "qwen2",
            Self::Qwen2Moe => "qwen2moe",
            Self::Qwen2Vl => "qwen2vl",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3moe",
            Self::Qwen3Next => "qwen3next",
            Self::Qwen3Vl => "qwen3vl",
            Self::Qwen3VlMoe => "qwen3vlmoe",
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::Phi2 => "phi2",
            Self::Phi3 => "phi3",
            Self::PhiMoe => "phimoe",
            Self::Plamo => "plamo",
            Self::Plamo2 => "plamo2",
            Self::Plamo3 => "plamo3",
            Self::Codeshell => "codeshell",
            Self::Orion => "orion",
            Self::InternLm2 => "internlm2",
            Self::MiniCpm => "minicpm",
            Self::MiniCpm3 => "minicpm3",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Gemma3 => "gemma3",
            Self::Gemma3n => "gemma3n",
            Self::Gemma4 => "gemma4",
            Self::Gemma4Assistant => "gemma4-assistant",
            Self::GemmaEmbedding => "gemma-embedding",
            Self::Starcoder2 => "starcoder2",
            Self::Mamba => "mamba",
            Self::Mamba2 => "mamba2",
            Self::Jamba => "jamba",
            Self::FalconH1 => "falcon-h1",
            Self::Xverse => "xverse",
            Self::CommandR => "command-r",
            Self::Cohere2 => "cohere2",
            Self::Cohere2Moe => "cohere2moe",
            Self::Dbrx => "dbrx",
            Self::Olmo => "olmo",
            Self::Olmo2 => "olmo2",
            Self::Olmoe => "olmoe",
            Self::OpenElm => "openelm",
            Self::Arctic => "arctic",
            Self::DeepSeek => "deepseek",
            Self::DeepSeek2 => "deepseek2",
            Self::DeepSeek2Ocr => "deepseek2ocr",
            Self::DeepSeek32 => "deepseek32",
            Self::DeepSeek4 => "deepseek4",
            Self::ChatGlm => "chatglm",
            Self::Glm4 => "glm4",
            Self::Glm4Moe => "glm4-moe",
            Self::GlmDsa => "glm-dsa",
            Self::BitNet => "bitnet",
            Self::T5 => "t5",
            Self::T5Encoder => "t5encoder",
            Self::Jais => "jais",
            Self::Jais2 => "jais2",
            Self::Nemotron => "nemotron",
            Self::NemotronH => "nemotron-h",
            Self::NemotronHMoe => "nemotron-h-moe",
            Self::Exaone => "exaone",
            Self::Exaone4 => "exaone4",
            Self::ExaoneMoe => "exaone-moe",
            Self::Rwkv6 => "rwkv6",
            Self::Rwkv6Qwen2 => "rwkv6qwen2",
            Self::Rwkv7 => "rwkv7",
            Self::ARwkv7 => "arwkv7",
            Self::Granite => "granite",
            Self::GraniteMoe => "granite-moe",
            Self::GraniteHybrid => "granite-hybrid",
            Self::Chameleon => "chameleon",
            Self::WavTokenizerDec => "wavtokenizer-dec",
            Self::Plm => "plm",
            Self::BailingMoe => "bailingmoe",
            Self::BailingMoe2 => "bailingmoe2",
            Self::Dots1 => "dots1",
            Self::Arcee => "arcee",
            Self::AfMoe => "afmoe",
            Self::Laguna => "laguna",
            Self::Ernie45 => "ernie4-5",
            Self::Ernie45Moe => "ernie4-5-moe",
            Self::HunyuanMoe => "hunyuan-moe",
            Self::HunyuanDense => "hunyuan-dense",
            Self::HunyuanVl => "hunyuan-vl",
            Self::HyV3 => "hy-v3",
            Self::SmolLm3 => "smollm3",
            Self::OpenAiMoe => "openai-moe",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe => "lfm2moe",
            Self::Dream => "dream",
            Self::SmallThinker => "smallthinker",
            Self::Llada => "llada",
            Self::LladaMoe => "llada-moe",
            Self::SeedOss => "seed-oss",
            Self::GroveMoe => "grovemoe",
            Self::Apertus => "apertus",
            Self::MiniMaxM2 => "minimax-m2",
            Self::CogVlm => "cogvlm",
            Self::Rnd1 => "rnd1",
            Self::PanguEmbed => "pangu-embed",
            Self::Mistral3 => "mistral3",
            Self::Mistral4 => "mistral4",
            Self::PaddleOcr => "paddleocr",
            Self::Mimo2 => "mimo2",
            Self::Step35 => "step35",
            Self::LlamaEmbed => "llama-embed",
            Self::MainCoder => "maincoder",
            Self::KimiLinear => "kimi-linear",
            Self::Talkie => "talkie",
            Self::Mellum => "mellum",
            Self::Eagle3 => "eagle3",
            Self::DFlash => "dflash",
            Self::Unknown => "unknown",
        }
    }

    /// Return coarse modality hint for architecture.
    pub fn modality(&self) -> ModalityHint {
        match self {
            Self::Bert | Self::ModernBert | Self::NomicBert | Self::NeoBert | Self::JinaBertV2 | Self::JinaBertV3 | Self::GemmaEmbedding | Self::PanguEmbed | Self::LlamaEmbed => ModalityHint::VisionEncoder,
            Self::T5 | Self::T5Encoder => ModalityHint::TextInTextOut,
            _ => ModalityHint::TextInTextOut,
        }
    }

    /// Returns `true` if architecture uses Mixture of Experts (MoE).
    pub fn is_moe(&self) -> bool {
        matches!(
            self,
            Self::Qwen2Moe | Self::Qwen3Moe | Self::Qwen3VlMoe | Self::Qwen35Moe | Self::PhiMoe | Self::Cohere2Moe | Self::Dbrx | Self::Olmoe | Self::DeepSeek2 | Self::DeepSeek32 | Self::DeepSeek4 | Self::Glm4Moe | Self::NemotronHMoe | Self::ExaoneMoe | Self::GraniteMoe | Self::BailingMoe | Self::BailingMoe2 | Self::AfMoe | Self::Ernie45Moe | Self::HunyuanMoe | Self::OpenAiMoe | Self::Lfm2Moe | Self::LladaMoe | Self::GroveMoe
        )
    }

    /// Returns `true` if architecture is an SSM/Mamba variant.
    pub fn is_ssm(&self) -> bool {
        matches!(self, Self::Mamba | Self::Mamba2 | Self::Jamba | Self::NemotronH | Self::GraniteHybrid)
    }

    /// Returns `true` if architecture is an RWKV variant.
    pub fn is_rwkv(&self) -> bool {
        matches!(self, Self::Rwkv6 | Self::Rwkv6Qwen2 | Self::Rwkv7 | Self::ARwkv7)
    }

    /// Returns `true` if architecture is an encoder-only model.
    pub fn is_encoder(&self) -> bool {
        matches!(
            self,
            Self::Bert | Self::ModernBert | Self::NomicBert | Self::NomicBertMoe | Self::NeoBert | Self::JinaBertV2 | Self::JinaBertV3 | Self::Eurobert | Self::GemmaEmbedding | Self::PanguEmbed | Self::LlamaEmbed
        )
    }
}

/// Abstract representation of logical tensor roles within a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    TokenEmb,
    TokenEmbNorm,
    OutputNorm,
    Output,
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    AttnNorm,
    AttnQNorm,
    AttnKNorm,
    FfnNorm,
    FfnGate,
    FfnDown,
    FfnUp,
    MoeGate,
    MoeExperts,
    ConvIn,
    ConvWeight,
    ConvOut,
    SsmA,
    SsmB,
    SsmC,
    SsmD,
    SsmDt,
    RwkvTimeMixK,
    RwkvTimeMixV,
    RwkvTimeMixR,
    RwkvTimeMixG,
    RwkvTimeDecay,
}

/// Registry providing standard GGUF and HuggingFace tensor names per architecture.
pub struct TensorNamingRegistry;

impl TensorNamingRegistry {
    /// Return the canonical GGUF tensor name for a given role and layer index.
    pub fn gguf_name(_arch: ModelArchitecture, role: TensorRole, layer_idx: Option<usize>) -> String {
        let prefix = match layer_idx {
            Some(i) => format!("blk.{i}."),
            None => String::new(),
        };

        match role {
            TensorRole::TokenEmb => "token_embd.weight".to_string(),
            TensorRole::TokenEmbNorm => "token_embd_norm.weight".to_string(),
            TensorRole::OutputNorm => "output_norm.weight".to_string(),
            TensorRole::Output => "output.weight".to_string(),
            TensorRole::AttnQ => format!("{prefix}attn_q.weight"),
            TensorRole::AttnK => format!("{prefix}attn_k.weight"),
            TensorRole::AttnV => format!("{prefix}attn_v.weight"),
            TensorRole::AttnOutput => format!("{prefix}attn_output.weight"),
            TensorRole::AttnNorm => format!("{prefix}attn_norm.weight"),
            TensorRole::AttnQNorm => format!("{prefix}attn_q_norm.weight"),
            TensorRole::AttnKNorm => format!("{prefix}attn_k_norm.weight"),
            TensorRole::FfnNorm => format!("{prefix}ffn_norm.weight"),
            TensorRole::FfnGate => format!("{prefix}ffn_gate.weight"),
            TensorRole::FfnDown => format!("{prefix}ffn_down.weight"),
            TensorRole::FfnUp => format!("{prefix}ffn_up.weight"),
            TensorRole::MoeGate => format!("{prefix}ffn_gate_inp.weight"),
            TensorRole::MoeExperts => format!("{prefix}ffn_experts.weight"),
            TensorRole::ConvIn => format!("{prefix}shortconv.in_proj.weight"),
            TensorRole::ConvWeight => format!("{prefix}shortconv.conv.weight"),
            TensorRole::ConvOut => format!("{prefix}shortconv.out_proj.weight"),
            TensorRole::SsmA => format!("{prefix}ssm_a"),
            TensorRole::SsmB => format!("{prefix}ssm_b"),
            TensorRole::SsmC => format!("{prefix}ssm_c"),
            TensorRole::SsmD => format!("{prefix}ssm_d"),
            TensorRole::SsmDt => format!("{prefix}ssm_dt"),
            TensorRole::RwkvTimeMixK => format!("{prefix}time_mix_k"),
            TensorRole::RwkvTimeMixV => format!("{prefix}time_mix_v"),
            TensorRole::RwkvTimeMixR => format!("{prefix}time_mix_r"),
            TensorRole::RwkvTimeMixG => format!("{prefix}time_mix_g"),
            TensorRole::RwkvTimeDecay => format!("{prefix}time_decay"),
        }
    }

    /// Return a HuggingFace to GGUF tensor name translation map for a specific architecture.
    pub fn remap_hf_to_gguf(arch: ModelArchitecture, num_layers: usize) -> HashMap<String, String> {
        let mut map = HashMap::new();

        // Common default HF -> GGUF mappings
        map.insert("model.embed_tokens.weight".to_string(), "token_embd.weight".to_string());
        map.insert("model.norm.weight".to_string(), "output_norm.weight".to_string());
        map.insert("lm_head.weight".to_string(), "output.weight".to_string());

        match arch {
            ModelArchitecture::Lfm2 => {
                map.insert("model.embedding_norm.weight".to_string(), "token_embd_norm.weight".to_string());
                for i in 0..num_layers {
                    let hf_p = format!("model.layers.{i}.");
                    let gg_p = format!("blk.{i}.");
                    map.insert(format!("{hf_p}operator_norm.weight"), format!("{gg_p}attn_norm.weight"));
                    map.insert(format!("{hf_p}self_attn.q_proj.weight"), format!("{gg_p}attn_q.weight"));
                    map.insert(format!("{hf_p}self_attn.k_proj.weight"), format!("{gg_p}attn_k.weight"));
                    map.insert(format!("{hf_p}self_attn.v_proj.weight"), format!("{gg_p}attn_v.weight"));
                    map.insert(format!("{hf_p}self_attn.out_proj.weight"), format!("{gg_p}attn_output.weight"));
                    map.insert(format!("{hf_p}self_attn.q_layernorm.weight"), format!("{gg_p}attn_q_norm.weight"));
                    map.insert(format!("{hf_p}self_attn.k_layernorm.weight"), format!("{gg_p}attn_k_norm.weight"));
                    map.insert(format!("{hf_p}conv.in_proj.weight"), format!("{gg_p}shortconv.in_proj.weight"));
                    map.insert(format!("{hf_p}conv.conv.weight"), format!("{gg_p}shortconv.conv.weight"));
                    map.insert(format!("{hf_p}conv.out_proj.weight"), format!("{gg_p}shortconv.out_proj.weight"));
                    map.insert(format!("{hf_p}ffn_norm.weight"), format!("{gg_p}ffn_norm.weight"));
                    map.insert(format!("{hf_p}feed_forward.w1.weight"), format!("{gg_p}ffn_gate.weight"));
                    map.insert(format!("{hf_p}feed_forward.w3.weight"), format!("{gg_p}ffn_up.weight"));
                    map.insert(format!("{hf_p}feed_forward.w2.weight"), format!("{gg_p}ffn_down.weight"));
                }
            }
            ModelArchitecture::Falcon => {
                for i in 0..num_layers {
                    let hf_p = format!("transformer.h.{i}.");
                    let gg_p = format!("blk.{i}.");
                    map.insert(format!("{hf_p}input_layernorm.weight"), format!("{gg_p}attn_norm.weight"));
                    map.insert(format!("{hf_p}self_attention.query_key_value.weight"), format!("{gg_p}attn_qkv.weight"));
                    map.insert(format!("{hf_p}self_attention.dense.weight"), format!("{gg_p}attn_output.weight"));
                    map.insert(format!("{hf_p}mlp.dense_h_to_4h.weight"), format!("{gg_p}ffn_up.weight"));
                    map.insert(format!("{hf_p}mlp.dense_4h_to_h.weight"), format!("{gg_p}ffn_down.weight"));
                }
            }
            ModelArchitecture::Gpt2 => {
                map.insert("transformer.wte.weight".to_string(), "token_embd.weight".to_string());
                map.insert("transformer.ln_f.weight".to_string(), "output_norm.weight".to_string());
                for i in 0..num_layers {
                    let hf_p = format!("transformer.h.{i}.");
                    let gg_p = format!("blk.{i}.");
                    map.insert(format!("{hf_p}ln_1.weight"), format!("{gg_p}attn_norm.weight"));
                    map.insert(format!("{hf_p}attn.c_attn.weight"), format!("{gg_p}attn_qkv.weight"));
                    map.insert(format!("{hf_p}attn.c_proj.weight"), format!("{gg_p}attn_output.weight"));
                    map.insert(format!("{hf_p}ln_2.weight"), format!("{gg_p}ffn_norm.weight"));
                    map.insert(format!("{hf_p}mlp.c_fc.weight"), format!("{gg_p}ffn_up.weight"));
                    map.insert(format!("{hf_p}mlp.c_proj.weight"), format!("{gg_p}ffn_down.weight"));
                }
            }
            _ => {
                // Default Llama-style HF -> GGUF mappings per layer
                for i in 0..num_layers {
                    let hf_p = format!("model.layers.{i}.");
                    let gg_p = format!("blk.{i}.");
                    map.insert(format!("{hf_p}input_layernorm.weight"), format!("{gg_p}attn_norm.weight"));
                    map.insert(format!("{hf_p}self_attn.q_proj.weight"), format!("{gg_p}attn_q.weight"));
                    map.insert(format!("{hf_p}self_attn.k_proj.weight"), format!("{gg_p}attn_k.weight"));
                    map.insert(format!("{hf_p}self_attn.v_proj.weight"), format!("{gg_p}attn_v.weight"));
                    map.insert(format!("{hf_p}self_attn.o_proj.weight"), format!("{gg_p}attn_output.weight"));
                    map.insert(format!("{hf_p}post_attention_layernorm.weight"), format!("{gg_p}ffn_norm.weight"));
                    map.insert(format!("{hf_p}mlp.gate_proj.weight"), format!("{gg_p}ffn_gate.weight"));
                    map.insert(format!("{hf_p}mlp.up_proj.weight"), format!("{gg_p}ffn_up.weight"));
                    map.insert(format!("{hf_p}mlp.down_proj.weight"), format!("{gg_p}ffn_down.weight"));
                }
            }
        }

        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_architecture_parsing() {
        assert_eq!(ModelArchitecture::from_str("llama"), ModelArchitecture::Llama);
        assert_eq!(ModelArchitecture::from_str("qwen2"), ModelArchitecture::Qwen2);
        assert_eq!(ModelArchitecture::from_str("lfm2"), ModelArchitecture::Lfm2);
        assert_eq!(ModelArchitecture::from_str("rwkv6"), ModelArchitecture::Rwkv6);
        assert_eq!(ModelArchitecture::from_str("mamba2"), ModelArchitecture::Mamba2);
        assert_eq!(ModelArchitecture::from_str("modern-bert"), ModelArchitecture::ModernBert);
        assert_eq!(ModelArchitecture::from_str("unknown_arch"), ModelArchitecture::Unknown);
    }

    #[test]
    fn test_tensor_naming_registry() {
        let name = TensorNamingRegistry::gguf_name(ModelArchitecture::Llama, TensorRole::AttnQ, Some(0));
        assert_eq!(name, "blk.0.attn_q.weight");

        let remap = TensorNamingRegistry::remap_hf_to_gguf(ModelArchitecture::Lfm2, 1);
        assert_eq!(remap.get("model.layers.0.self_attn.q_proj.weight").unwrap(), "blk.0.attn_q.weight");
    }
}
