//! Safetensors + dequant + name-resolution tests.
use gemma_wgpu::config::Gemma4Config;
use gemma_wgpu::dequant::{dequantize, dequantize_weight, pack_int, unpack_int};
use gemma_wgpu::safetensors::{build_safetensors, RawTensor, SafetensorsFile};
use gemma_wgpu::weights::{is_text_tensor, resolve_tensor_names};

fn raw(dtype: &str, shape: &[usize], bytes: Vec<u8>) -> RawTensor {
    RawTensor { dtype: dtype.into(), shape: shape.to_vec(), bytes }
}

#[test]
fn safetensors_dtype_decode() {
    let f32b: Vec<u8> = [1.5f32, -2.25, 3.0, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let f16b: Vec<u8> = [0x3c00u16, 0x4000].iter().flat_map(|h| h.to_le_bytes()).collect();
    let bf16b: Vec<u8> = [0x3f80u16, 0xc000].iter().flat_map(|h| h.to_le_bytes()).collect();
    let blob = build_safetensors(&[
        ("a".into(), raw("F32", &[2, 2], f32b)),
        ("b".into(), raw("F16", &[2], f16b)),
        ("c".into(), raw("BF16", &[2], bf16b)),
    ]);
    let st = SafetensorsFile::from_vec(blob).unwrap();
    let mut names: Vec<&str> = st.names().collect();
    names.sort();
    assert_eq!(names, ["a", "b", "c"]);
    assert_eq!(st.tensor("a").unwrap().data, [1.5, -2.25, 3.0, 0.0]);
    assert_eq!(st.tensor("b").unwrap().data, [1.0, 2.0]);
    assert_eq!(st.tensor("c").unwrap().data, [1.0, -2.0]);
    assert_eq!(st.meta("a").unwrap().shape, [2, 2]);
    assert!(st.tensor("zzz").is_err());
}

#[test]
fn int_pack_unpack_roundtrip() {
    for bits in [2u32, 4, 8] {
        let lo = -(1i32 << (bits - 1));
        let hi = (1i32 << (bits - 1)) - 1;
        let vals: Vec<i32> = (0..33).map(|i| lo + (i % (hi - lo + 1))).collect();
        let packed = pack_int(&vals, bits);
        let back = unpack_int(&packed, bits, vals.len());
        assert_eq!(back, vals, "{bits}-bit pack/unpack exact");
    }
}

#[test]
fn dequantize_per_row() {
    let q = [1, -2, 3, 0, 2, 2, -1, 4];
    let scales = [0.5f32, 2.0];
    let out = dequantize(&q, &scales, 2, 4, 4);
    assert_eq!(out, [0.5, -1.0, 1.5, 0.0, 4.0, 4.0, -2.0, 8.0]);
    let bytes = pack_int(&q, 4);
    assert_eq!(dequantize_weight(&bytes, &scales, [2, 4], 4, Some(4)), out);
}

#[test]
fn name_resolution_and_bits() {
    let r = resolve_tensor_names("language_model.layers.3.self_attn.q_proj.weight");
    assert_eq!(r.weight, "model.language_model.layers.3.self_attn.q_proj.weight");
    assert_eq!(r.scale, "model.language_model.layers.3.self_attn.q_proj.weight_scale");
    assert_eq!(resolve_tensor_names("language_model.layers.3.layer_scalar").bare.as_deref(), Some("model.language_model.layers.3.layer_scalar"));
    assert_eq!(resolve_tensor_names("lm_head.weight").weight, "lm_head.weight");
    assert_eq!(resolve_tensor_names("language_model.embed_tokens.weight").weight, "model.language_model.embed_tokens.embedding_quantized");
    assert!(is_text_tensor("model.language_model.norm.weight") && !is_text_tensor("model.vision_tower.x"));

    let cfg = Gemma4Config::from_json_str(gemma_wgpu::GEMMA4_E2B_CONFIG_JSON).unwrap();
    assert_eq!(cfg.num_hidden_layers, 35);
    assert_eq!(cfg.bits_for("lm_head"), 2);
    assert_eq!(cfg.bits_for("language_model.layers.3.mlp.gate_proj"), 4);
    assert_eq!(cfg.bits_for("language_model.layers.15.mlp.gate_proj"), 2);
    assert_eq!(cfg.bits_for("language_model.layers.15.per_layer_input_gate"), 8);
    assert_eq!(cfg.bits_for("language_model.layers.0.self_attn.q_proj"), 4);
    assert_eq!(cfg.bits_for("language_model.norm"), 16);
    assert!(cfg.is_global(4) && !cfg.is_global(3));
    assert_eq!(cfg.kv_source_layer(34), 14);
    assert_eq!(cfg.kv_source_layer(20), 13);
}
