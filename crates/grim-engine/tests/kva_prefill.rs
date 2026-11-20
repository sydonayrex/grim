use grim_engine::kva_prefill::{KvaSplitPlan, LayerPrefillMode};

#[test]
fn astrum_p7_kva_plan_covers_all_layers_and_enforces_non_empty_exact_tail() {
    let plan = KvaSplitPlan::plan(48, 24, 0);
    assert_eq!(plan.n_layers, 48);
    assert_eq!(plan.early_layers, 24);
    assert_eq!(plan.tail_len, 1, "tail_len >= 1 must be enforced");
    assert_eq!(plan.prefix_modes.len(), 48);
    assert_eq!(plan.tail_modes.len(), 48);
    assert!(plan.prefix_modes[..24].iter().all(|&m| m == LayerPrefillMode::Full));
    assert!(plan.prefix_modes[24..].iter().all(|&m| m == LayerPrefillMode::StateOnly));
    assert!(plan.tail_modes.iter().all(|&m| m == LayerPrefillMode::Full));
}

#[test]
fn astrum_p7_kva_env_parser_and_prompt_splitter() {
    assert!(KvaSplitPlan::from_env_str(None, 48).is_none());
    assert!(KvaSplitPlan::from_env_str(Some("48/16"), 48).is_none());
    let plan = KvaSplitPlan::from_env_str(Some("20/8"), 32).expect("valid KVA spec");
    assert_eq!(plan.early_layers, 20);
    assert_eq!(plan.tail_len, 8);

    let tokens: Vec<u32> = (0..20).collect();
    let (prefix, tail) = plan.split_prompt(&tokens);
    assert_eq!(prefix.len(), 12);
    assert_eq!(tail.len(), 8);
    assert_eq!(tail, &[12, 13, 14, 15, 16, 17, 18, 19]);

    let short: Vec<u32> = (0..5).collect();
    let (p_short, t_short) = plan.split_prompt(&short);
    assert!(p_short.is_empty());
    assert_eq!(t_short.len(), 5);
}
