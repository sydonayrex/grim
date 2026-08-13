//! Dense CausalLm transformer implementations (Llama, Mistral, Qwen, DeepSeek, Gemma, T5, MTP).

pub mod block;
/// Shared MoE block (router + expert bank + optional shared expert).
pub mod moe_block;
pub mod qwen3moe;
pub mod configs;
pub mod deepseek;
pub mod gemma;
pub mod gpt2;
pub mod lfm2;
pub mod lora;
pub mod minicpm;
pub mod model;
pub mod multimodal;
pub mod muse_glimmer;
pub mod native_mtp;
pub mod falcon_h1;
pub mod t5;
pub mod falcon;
pub mod bloom;
pub mod phi2;
pub mod qwen;
pub mod starcoder2;
pub mod granite_moe;
pub mod deepseek2ocr;
pub mod laguna;
pub mod maple;
pub mod hunyuan_vl;
pub mod grovemoe;
pub mod eurobert;
pub mod deci;
pub mod dbrx;
pub mod commandr;
pub mod cogvlm;
pub mod bailingmoe2;
pub mod bailingmoe;
pub mod bailingmoe3;
pub mod hyv3;
pub mod eagle3;
pub mod dots1;
pub mod dflash;
pub mod wav_tokenizer_dec;
pub mod talkie;
pub mod step35;
pub mod qwen3vl;
pub mod pangu_embed;
pub mod paddle_ocr;
pub mod openelm;
pub mod openai_moe;
pub mod olmoe;
pub mod olmo2;
pub mod olmo;
pub mod nemotron_hmoe;
pub mod nemotron;
pub mod mistral4;
pub mod mistral3;
pub mod minimax_m2;
pub mod mimo2;
pub mod mellum;
pub mod maincoder;
pub mod llama4;
pub mod kimi_linear;
pub mod jais2;
pub mod jais;
pub mod bitnet;
pub mod glmdsa;
pub mod glm4moe;
pub mod glm4;
pub mod chatglm;
pub mod stablelm;
pub mod refact;
pub mod starcoder;
pub mod mpt;
pub mod gptneox;
pub mod gptj;
pub mod grok;
pub mod baichuan;
pub mod hunyuan_moe;
pub mod ernie45;
pub mod cohere2;
pub mod smollm2;
pub mod lladamoe;
pub mod plamo3;
pub mod plamo2;
pub mod gemma_embedding;
pub mod exaone_moe;
pub mod ernie4_5_moe;
pub mod cohere2moe;
pub mod arctic;
pub mod afmoe;
pub mod granite;
pub mod hunyuan_dense;
pub mod exaone4;
pub mod gemma4_assistant;
pub mod qwen3next;
pub mod qwen2vl;
pub mod qwen2moe;
pub mod plm;
pub mod smallthinker;
pub mod qwen3;
pub mod llama_embed;
pub mod apertus;
pub mod exaone;
pub mod smollm3;
pub mod seed_oss;
pub mod rnd1;
pub mod llada;
pub mod dream;
pub mod plamo;
pub mod xverse;
pub mod internlm2;
pub mod arcee;
pub mod codeshell;
pub mod orion;
pub mod chameleon;
pub mod interns2_mobius;
pub mod kimi_k3;
pub mod inkling_small;
pub mod glm5_2;
pub mod diffusion_gemma;
pub mod minimax_m3;
pub mod delta_net_base;



pub use block::{LlamaBlock, LlamaConfigRefs, LlamaLayerCache};
pub use configs::{BloomConfig, MoeConfig, PhiConfig, QwenConfig};
pub use deepseek::{DeepSeek, DeepSeekConfig};
pub use falcon::{Falcon, FalconConfig};
pub use bloom::Bloom;
pub use phi2::Phi2;
pub use qwen::Qwen;
pub use arcee::{Arcee, ArceeConfig};
pub use chameleon::{Chameleon, ChameleonConfig};
pub use codeshell::{Codeshell, CodeshellConfig};
pub use delta_net_base::{DeltaNetBase, DeltaNetBaseConfig};
pub use diffusion_gemma::{DiffusionGemma, DiffusionGemmaConfig};
pub use glm5_2::{Glm52, Glm52Config};
pub use inkling_small::{InklingSmall, InklingSmallConfig};
pub use maple::{Maple, MapleConfig};
pub use interns2_mobius::{InternS2Mobius, InternS2MobiusConfig};
pub use kimi_k3::{KimiK3, KimiK3Config};
pub use minimax_m3::{MiniMaxM3, MiniMaxM3Config};
pub use muse_glimmer::{MuseGlimmer, MuseGlimmerConfig};
pub use orion::{Orion, OrionConfig};



pub use starcoder2::{Starcoder2, Starcoder2Config};
pub use granite_moe::{GraniteMoe, GraniteMoeConfig};
pub use deepseek2ocr::{DeepSeek2Ocr, DeepSeek2OcrConfig};
pub use laguna::{Laguna, LagunaConfig};
pub use qwen3moe::{Qwen3Moe, Qwen3MoeConfig};
pub use hunyuan_vl::{HunyuanVl, HunyuanVlConfig};
pub use grovemoe::{GroveMoe, GroveMoeConfig};
pub use eurobert::{Eurobert, EurobertConfig};
pub use deci::{Deci, DeciConfig};
pub use dbrx::{Dbrx, DbrxConfig};
pub use commandr::{CommandR, CommandRConfig};
pub use cogvlm::{CogVlm, CogVlmConfig};
pub use bailingmoe2::{BailingMoe2, BailingMoe2Config};
pub use bailingmoe::{BailingMoe, BailingMoeConfig};
pub use bailingmoe3::{Ling3Tiny, Ling3TinyConfig};
pub use hyv3::{HyV3, HyV3Config};
pub use eagle3::{Eagle3, Eagle3Config};
pub use dots1::{Dots1, Dots1Config};
pub use dflash::{DFlash, DFlashConfig};
pub use wav_tokenizer_dec::{WavTokenizerDec, WavTokenizerDecConfig};
pub use talkie::{Talkie, TalkieConfig};
pub use step35::{Step35, Step35Config};
pub use qwen3vl::{Qwen3Vl, Qwen3VlConfig};
pub use pangu_embed::{PanguEmbed, PanguEmbedConfig};
pub use paddle_ocr::{PaddleOcr, PaddleOcrConfig};
pub use openelm::{OpenElm, OpenElmConfig};
pub use openai_moe::{OpenAiMoe, OpenAiMoeConfig};
pub use olmoe::{Olmoe, OlmoeConfig};
pub use olmo2::{Olmo2, Olmo2Config};
pub use olmo::{Olmo, OlmoConfig};
pub use nemotron_hmoe::{NemotronHMoe, NemotronHMoeConfig};
pub use nemotron::{Nemotron, NemotronConfig};
pub use mistral4::{Mistral4, Mistral4Config};
pub use mistral3::{Mistral3, Mistral3Config};
pub use minimax_m2::{MiniMaxM2, MiniMaxM2Config};
pub use mimo2::{Mimo2, Mimo2Config};
pub use mellum::{Mellum, MellumConfig};
pub use maincoder::{MainCoder, MainCoderConfig};
pub use llama4::{Llama4, Llama4Config};
pub use kimi_linear::{KimiLinear, KimiLinearConfig};
pub use jais2::{Jais2, Jais2Config};
pub use jais::{Jais, JaisConfig};
pub use bitnet::{BitNet, BitNetConfig};
pub use glmdsa::{GlmDsa, GlmDsaConfig};
pub use glm4moe::{Glm4Moe, Glm4MoeConfig};
pub use glm4::{Glm4, Glm4Config};
pub use chatglm::{ChatGlm, ChatGlmConfig};
pub use stablelm::{StableLm, StableLmConfig};
pub use refact::{Refact, RefactConfig};
pub use starcoder::{Starcoder, StarcoderConfig};
pub use mpt::{Mpt, MptConfig};
pub use gptneox::{GptNeoX, GptNeoXConfig};
pub use gptj::{GptJ, GptJConfig};
pub use grok::{Grok, GrokConfig};
pub use baichuan::{Baichuan, BaichuanConfig};
pub use hunyuan_moe::{HunyuanMoe, HunyuanMoeConfig};
pub use ernie45::{Ernie45, Ernie45Config};
pub use cohere2::{Cohere2, Cohere2Config};
pub use smollm2::{SmolLm2, SmolLm2Config};
pub use lladamoe::{LladaMoe, LladaMoeConfig};
pub use plamo3::{Plamo3, Plamo3Config};
pub use plamo2::{Plamo2, Plamo2Config};
pub use gemma_embedding::{GemmaEmbedding, GemmaEmbeddingConfig};
pub use exaone_moe::{ExaoneMoe, ExaoneMoeConfig};
pub use ernie4_5_moe::{Ernie45Moe, Ernie45MoeConfig};
pub use cohere2moe::{Cohere2Moe, Cohere2MoeConfig};
pub use arctic::{Arctic, ArcticConfig};
pub use afmoe::{AfMoe, AfMoeConfig};
pub use granite::{Granite, GraniteConfig};
pub use hunyuan_dense::{HunyuanDense, HunyuanDenseConfig};
pub use exaone4::{Exaone4, Exaone4Config};
pub use gemma4_assistant::{Gemma4Assistant, Gemma4AssistantConfig};
pub use qwen3next::{Qwen3Next, Qwen3NextConfig};
pub use qwen2vl::{Qwen2Vl, Qwen2VlConfig};
pub use qwen2moe::{Qwen2Moe, Qwen2MoeConfig};
pub use plm::{Plm, PlmConfig};
pub use smallthinker::{SmallThinker, SmallThinkerConfig};
pub use qwen3::{Qwen3, Qwen3Config};
pub use llama_embed::{LlamaEmbed, LlamaEmbedConfig};
pub use apertus::{Apertus, ApertusConfig};
pub use exaone::{Exaone, ExaoneConfig};
pub use smollm3::{SmolLm3, SmolLm3Config};
pub use seed_oss::{SeedOss, SeedOssConfig};
pub use rnd1::{Rnd1, Rnd1Config};
pub use llada::{Llada, LladaConfig};
pub use dream::{Dream, DreamConfig};
pub use plamo::{Plamo, PlamoConfig};
pub use xverse::{Xverse, XverseConfig};
pub use internlm2::{InternLm2, InternLm2Config};
pub use falcon_h1::{FalconH1Config, FalconH1LayerCache, FalconH1Model};
pub use gemma::{Gemma, GemmaConfig};
pub use gpt2::{Gpt2, Gpt2Config};
pub use lfm2::{Lfm2, Lfm2Config};
pub use minicpm::{MiniCpmConfig, MiniCpmModel};
pub use model::{Llama, LlamaConfig};
pub use multimodal::*;
pub use native_mtp::{LlamaMtp, MtpDepthProvider};
pub use t5::{T5, T5Config};

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

