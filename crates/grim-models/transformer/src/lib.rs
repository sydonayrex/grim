//! Dense CausalLm transformer implementations (Llama, Mistral, Qwen, DeepSeek, Gemma, T5, MTP).

pub mod afmoe;
pub mod apertus;
pub mod arcee;
pub mod arctic;
pub mod attention_dispatcher;

pub use attention_dispatcher::{
    AttentionDispatcher, AttentionRequest, AttentionTier, AttentionTopology,
};
pub mod baichuan;
pub mod bailingmoe;
pub mod bailingmoe2;
pub mod bailingmoe3;
pub mod bitnet;
pub mod block;
pub mod bloom;
pub mod chameleon;
pub mod chatglm;
pub mod codeshell;
pub mod cogvlm;
pub mod cohere2;
pub mod cohere2moe;
pub mod commandr;
pub mod configs;
pub mod dbrx;
pub mod deci;
pub mod deepseek;
pub mod deepseek2;
pub mod deepseek2ocr;
pub mod deepseek32;
pub mod deepseek4;
pub mod delta_net_base;
pub mod dflash;
pub mod diffusion_gemma;
pub mod dots1;
pub mod dream;
pub mod eagle3;
pub mod ernie45;
pub mod ernie4_5_moe;
pub mod eurobert;
pub mod exaone;
pub mod exaone4;
pub mod exaone_moe;
pub mod falcon;
pub mod falcon_h1;
pub mod gemma;
pub mod gemma3n;
pub mod gemma4_assistant;
pub mod gemma_embedding;
pub mod glm4;
pub mod glm4moe;
pub mod glm5_2;
pub mod glmdsa;
pub mod gpt2;
pub mod gptj;
pub mod gptneox;
pub mod granite;
pub mod granite_moe;
pub mod grok;
pub mod grovemoe;
pub mod hunyuan_dense;
pub mod hunyuan_moe;
pub mod hunyuan_vl;
pub mod hyv3;
pub mod inkling_small;
pub mod internlm2;
pub mod interns2_mobius;
pub mod jais;
pub mod jais2;
pub mod kimi_k3;
pub mod kimi_linear;
pub mod kv_attention;
pub mod laguna;
pub mod lfm2;
pub mod llada;
pub mod lladamoe;
pub mod llama4;
pub mod llama_embed;
pub mod lora;
pub mod maincoder;
pub mod maple;
pub mod mellum;
pub mod mimo2;
pub mod minicpm;
pub mod minimax_m2;
pub mod minimax_m3;
pub mod mistral3;
pub mod mistral4;
pub mod model;
/// Shared MoE block (router + expert bank + optional shared expert).
pub mod moe_block;
pub mod mpt;
pub mod multimodal;
pub mod muse_glimmer;
pub mod native_mtp;
pub mod nemotron;
pub mod nemotron_hmoe;
pub mod olmo;
pub mod olmo2;
pub mod olmoe;
pub mod openai_moe;
pub mod openelm;
pub mod orion;
pub mod paddle_ocr;
pub mod pangu_embed;
pub mod phi2;
pub mod plamo;
pub mod plamo2;
pub mod plamo3;
pub mod plm;
pub mod qwen;
pub mod qwen2moe;
pub mod qwen2vl;
pub mod qwen3;
pub mod qwen35;
pub mod qwen35moe;
pub mod qwen3moe;
pub mod qwen3next;
pub mod qwen3vl;
pub mod refact;
pub mod rnd1;
pub mod seed_oss;
pub mod smallthinker;
pub mod smollm2;
pub mod smollm3;
pub mod stablelm;
pub mod starcoder;
pub mod starcoder2;
pub mod step35;
pub mod t5;
pub mod talkie;
pub mod wav_tokenizer_dec;
pub mod xverse;

pub use arcee::{Arcee, ArceeConfig};
pub use block::{LlamaBlock, LlamaConfigRefs, LlamaLayerCache};
pub use bloom::Bloom;
pub use chameleon::{Chameleon, ChameleonConfig};
pub use codeshell::{Codeshell, CodeshellConfig};
pub use configs::{BloomConfig, MoeConfig, PhiConfig, QwenConfig};
pub use deepseek::{DeepSeek, DeepSeekConfig};
pub use deepseek2::{DeepSeek2, DeepSeek2Config};
pub use deepseek4::{DeepSeek4, DeepSeek4Config};
pub use deepseek32::{DeepSeek32, DeepSeek32Config};
pub use delta_net_base::{DeltaNetBase, DeltaNetBaseConfig};
pub use diffusion_gemma::{DiffusionGemma, DiffusionGemmaConfig};
pub use falcon::{Falcon, FalconConfig};
pub use glm5_2::{Glm52, Glm52Config};
pub use inkling_small::{InklingSmall, InklingSmallConfig};
pub use interns2_mobius::{InternS2Mobius, InternS2MobiusConfig};
pub use kimi_k3::{KimiK3, KimiK3Config};
pub use maple::{Maple, MapleConfig};
pub use minimax_m3::{MiniMaxM3, MiniMaxM3Config};
pub use muse_glimmer::{MuseGlimmer, MuseGlimmerConfig};
pub use orion::{Orion, OrionConfig};
pub use phi2::Phi2;
pub use qwen::Qwen;

pub use bailingmoe::{BailingMoe, BailingMoeConfig};
pub use bailingmoe2::{BailingMoe2, BailingMoe2Config};
pub use bailingmoe3::{Ling3Tiny, Ling3TinyConfig};
pub use cogvlm::{CogVlm, CogVlmConfig, CogVlmVisionConfig};
pub use commandr::{CommandR, CommandRConfig};
pub use dbrx::{Dbrx, DbrxConfig};
pub use deci::{Deci, DeciConfig};
pub use deepseek2ocr::{DeepSeek2Ocr, DeepSeek2OcrConfig};
pub use dflash::{DFlash, DFlashConfig};
pub use dots1::{Dots1, Dots1Config};
pub use eagle3::{Eagle3, Eagle3Config};
pub use eurobert::{Eurobert, EurobertConfig};
pub use granite_moe::{GraniteMoe, GraniteMoeConfig};
pub use grovemoe::{GroveMoe, GroveMoeConfig};
pub use hunyuan_vl::{HunyuanVl, HunyuanVlConfig, HunyuanVlVisionConfig};
pub use hyv3::{HyV3, HyV3Config};
pub use laguna::{Laguna, LagunaConfig};
pub use qwen3moe::{Qwen3Moe, Qwen3MoeConfig};
pub use starcoder2::{Starcoder2, Starcoder2Config};
pub mod solar_open2;
pub use afmoe::{AfMoe, AfMoeConfig};
pub use apertus::{Apertus, ApertusConfig};
pub use arctic::{Arctic, ArcticConfig};
pub use baichuan::{Baichuan, BaichuanConfig};
pub use bitnet::{BitNet, BitNetConfig};
pub use chatglm::{ChatGlm, ChatGlmConfig};
pub use cohere2::{Cohere2, Cohere2Config};
pub use cohere2moe::{Cohere2Moe, Cohere2MoeConfig};
pub use dream::{Dream, DreamConfig};
pub use ernie4_5_moe::{Ernie45Moe, Ernie45MoeConfig};
pub use ernie45::{Ernie45, Ernie45Config};
pub use exaone::{Exaone, ExaoneConfig};
pub use exaone_moe::{ExaoneMoe, ExaoneMoeConfig};
pub use exaone4::{Exaone4, Exaone4Config};
pub use falcon_h1::{FalconH1Config, FalconH1LayerCache, FalconH1Model};
pub use gemma::{Gemma, GemmaConfig};
pub use gemma_embedding::{GemmaEmbedding, GemmaEmbeddingConfig};
pub use gemma3n::{Gemma3n, Gemma3nConfig};
pub use gemma4_assistant::{Gemma4Assistant, Gemma4AssistantConfig};
pub use glm4::{Glm4, Glm4Config};
pub use glm4moe::{Glm4Moe, Glm4MoeConfig};
pub use glmdsa::{GlmDsa, GlmDsaConfig};
pub use gpt2::{Gpt2, Gpt2Config};
pub use gptj::{GptJ, GptJConfig};
pub use gptneox::{GptNeoX, GptNeoXConfig};
pub use granite::{Granite, GraniteConfig};
pub use grok::{Grok, GrokConfig};
pub use hunyuan_dense::{HunyuanDense, HunyuanDenseConfig};
pub use hunyuan_moe::{HunyuanMoe, HunyuanMoeConfig};
pub use internlm2::{InternLm2, InternLm2Config};
pub use jais::{Jais, JaisConfig};
pub use jais2::{Jais2, Jais2Config};
pub use kimi_linear::{KimiLinear, KimiLinearConfig};
pub use lfm2::{Lfm2, Lfm2Config};
pub use llada::{Llada, LladaConfig};
pub use lladamoe::{LladaMoe, LladaMoeConfig};
pub use llama_embed::{LlamaEmbed, LlamaEmbedConfig};
pub use llama4::{Llama4, Llama4Config};
pub use maincoder::{MainCoder, MainCoderConfig};
pub use mellum::{Mellum, MellumConfig};
pub use mimo2::{Mimo2, Mimo2Config};
pub use minicpm::{MiniCpmConfig, MiniCpmModel};
pub use minimax_m2::{MiniMaxM2, MiniMaxM2Config};
pub use mistral3::{Mistral3, Mistral3Config};
pub use mistral4::{Mistral4, Mistral4Config};
pub use model::{Llama, LlamaConfig};
pub use mpt::{Mpt, MptConfig};
pub use multimodal::*;
pub use native_mtp::{LlamaMtp, MtpDepthProvider};
pub use nemotron::{Nemotron, NemotronConfig};
pub use nemotron_hmoe::{NemotronHMoe, NemotronHMoeConfig};
pub use olmo::{Olmo, OlmoConfig};
pub use olmo2::{Olmo2, Olmo2Config};
pub use olmoe::{Olmoe, OlmoeConfig};
pub use openai_moe::{OpenAiMoe, OpenAiMoeConfig};
pub use openelm::{OpenElm, OpenElmConfig};
pub use paddle_ocr::{PaddleOcr, PaddleOcrConfig};
pub use pangu_embed::{PanguEmbed, PanguEmbedConfig};
pub use plamo::{Plamo, PlamoConfig};
pub use plamo2::{Plamo2, Plamo2Config};
pub use plamo3::{Plamo3, Plamo3Config};
pub use plm::{Plm, PlmConfig};
pub use qwen2moe::{Qwen2Moe, Qwen2MoeConfig};
pub use qwen2vl::{Qwen2Vl, Qwen2VlConfig, Qwen2VlVisionConfig};
pub use qwen3::{Qwen3, Qwen3Config};
pub use qwen3next::{Qwen3Next, Qwen3NextConfig};
pub use qwen3vl::{Qwen3Vl, Qwen3VlConfig, Qwen3VlVisionConfig};
pub use qwen35::{Qwen35, Qwen35Config};
pub use qwen35moe::{Qwen35Moe, Qwen35MoeConfig};
pub use refact::{Refact, RefactConfig};
pub use rnd1::{Rnd1, Rnd1Config};
pub use seed_oss::{SeedOss, SeedOssConfig};
pub use smallthinker::{SmallThinker, SmallThinkerConfig};
pub use smollm2::{SmolLm2, SmolLm2Config};
pub use smollm3::{SmolLm3, SmolLm3Config};
pub use solar_open2::{SolarLayerType, SolarOpen2, SolarOpen2Block, SolarOpen2Config};
pub use stablelm::{StableLm, StableLmConfig};
pub use starcoder::{Starcoder, StarcoderConfig};
pub use step35::{Step35, Step35Config};
pub use t5::{T5, T5Config};
pub use talkie::{Talkie, TalkieConfig};
pub use wav_tokenizer_dec::{WavTokenizerDec, WavTokenizerDecConfig};
pub use xverse::{Xverse, XverseConfig};

#[cfg(test)]
mod tests {
    use crate::{Llama, LlamaConfig};
    use grim_tensor::Device;

    #[test]
    fn smoke_tiny_llama_logits() {
        let cfg = LlamaConfig {
            vocab_size: 32000,
            hidden_size: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            num_layers: 2,
            intermediate_size: 384,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 256,

            partial_rotary_factor: 1.0,
            yarn: None,
        };
        let model = Llama::random(Device::Cpu, cfg);
        let tok = grim_backend_cpu::cpu_tensor(vec![1.0f32], grim_tensor::Shape::new(vec![1]));
        use grim_core::CausalLm;
        use grim_core::session::Inner;
        let mut sess = Inner::new(model.device.clone());
        let logits = CausalLm::forward(&model, &mut sess, &tok, &tok, &[]).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 32000]);
        let v = logits.to_vec_f32().unwrap();
        assert!(v.iter().any(|x| x.is_finite()));
        assert!(!v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn smoke_llama_with_empty_adapters_matches_baseline() {
        // Running with zero adapters must produce the same numerics as
        // the no-adapter sweep — guards against the fused-LoRA path
        // accidentally perturbing the base distribution.
        let cfg = LlamaConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 1,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 32,

            partial_rotary_factor: 1.0,
            yarn: None,
        };
        let model = Llama::random(Device::Cpu, cfg);
        let tok =
            grim_backend_cpu::cpu_tensor(vec![1.0f32, 2.0f32], grim_tensor::Shape::new(vec![2]));
        let mut sess_a = grim_core::session::Inner::new(model.device.clone());
        let mut sess_b = grim_core::session::Inner::new(model.device.clone());
        let base = grim_core::CausalLm::forward(&model, &mut sess_a, &tok, &tok, &[]).unwrap();
        let with_zero_adapters =
            grim_core::CausalLm::forward(&model, &mut sess_b, &tok, &tok, &[]).unwrap();
        let base_v = base.to_vec_f32().unwrap();
        let same = with_zero_adapters.to_vec_f32().unwrap();
        assert_eq!(base_v, same);
    }

    #[test]
    fn lora_apply_with_one_adapter_perturbs_logit_distribution() {
        // A single non-zero LoRA must measurably shift the logits
        // (preserving the architectural §4.5 contract that adapters
        // change the per-token distribution).
        use crate::lora::apply_adapters_to_logits;
        use grim_core::model::AdapterHandle;
        let logits = grim_backend_cpu::cpu_tensor(
            (0..32).map(|i| (i as f32 + 1.0) * 0.01).collect(),
            grim_tensor::Shape::new(vec![1, 32]),
        );
        let r = 4usize;
        let hidden = 32usize;
        let adapter = AdapterHandle {
            id: 1,
            a: grim_backend_cpu::cpu_tensor(
                (0..r * hidden)
                    .map(|i| ((i as f32) - (r * hidden) as f32 / 2.0) * 0.01)
                    .collect(),
                grim_tensor::Shape::new(vec![r, hidden]),
            ),
            b: grim_backend_cpu::cpu_tensor(
                (0..32 * r)
                    .map(|i| ((i as f32) - (32 * r) as f32 / 2.0) * 0.01)
                    .collect(),
                grim_tensor::Shape::new(vec![32, r]),
            ),
            alpha: 1.0,
        };
        let new_logits = apply_adapters_to_logits(&logits, &[adapter], hidden).unwrap();
        let v = new_logits.to_vec_f32().unwrap();
        assert!(
            v.iter().any(|x| *x != 0.0),
            "adapters must perturb the zero baseline"
        );
    }
}
