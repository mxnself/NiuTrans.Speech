import torch
import re
import os
from transformers import WhisperForConditionalGeneration
from whisper.model import ModelDimensions

def rename_param(name):
    # Embedding
    name = name.replace("model.encoder.embed_tokens.weight", "encoder.token_embedding.weight")
    name = name.replace("model.encoder.embed_positions.weight", "encoder.positional_embedding")
    name = name.replace("model.decoder.embed_tokens.weight", "decoder.token_embedding.weight")
    name = name.replace("model.decoder.embed_positions.weight", "decoder.positional_embedding")

    # Encoder conv
    name = name.replace("model.encoder.conv1.weight", "encoder.conv1.weight")
    name = name.replace("model.encoder.conv1.bias", "encoder.conv1.bias")
    name = name.replace("model.encoder.conv2.weight", "encoder.conv2.weight")
    name = name.replace("model.encoder.conv2.bias", "encoder.conv2.bias")

    # Encoder attention
    name = re.sub(r"model.encoder.layers.(\d+).self_attn.q_proj.weight", r"encoder.blocks.\1.attn.query.weight", name)
    name = re.sub(r"model.encoder.layers.(\d+).self_attn.q_proj.bias", r"encoder.blocks.\1.attn.query.bias", name)
    name = re.sub(r"model.encoder.layers.(\d+).self_attn.k_proj.weight", r"encoder.blocks.\1.attn.key.weight", name)
    name = re.sub(r"model.encoder.layers.(\d+).self_attn.v_proj.weight", r"encoder.blocks.\1.attn.value.weight", name)
    name = re.sub(r"model.encoder.layers.(\d+).self_attn.v_proj.bias", r"encoder.blocks.\1.attn.value.bias", name)
    name = re.sub(r"model.encoder.layers.(\d+).self_attn.out_proj.weight", r"encoder.blocks.\1.attn.out.weight", name)
    name = re.sub(r"model.encoder.layers.(\d+).self_attn.out_proj.bias", r"encoder.blocks.\1.attn.out.bias", name)

    # Encoder LayerNorm
    name = re.sub(r"model.encoder.layers.(\d+).self_attn_layer_norm.weight", r"encoder.blocks.\1.attn_ln.weight", name)
    name = re.sub(r"model.encoder.layers.(\d+).self_attn_layer_norm.bias", r"encoder.blocks.\1.attn_ln.bias", name)
    name = re.sub(r"model.encoder.layers.(\d+).final_layer_norm.weight", r"encoder.blocks.\1.mlp_ln.weight", name)
    name = re.sub(r"model.encoder.layers.(\d+).final_layer_norm.bias", r"encoder.blocks.\1.mlp_ln.bias", name)

    # Encoder FFN
    name = re.sub(r"model.encoder.layers.(\d+).fc1.weight", r"encoder.blocks.\1.mlp.0.weight", name)
    name = re.sub(r"model.encoder.layers.(\d+).fc1.bias", r"encoder.blocks.\1.mlp.0.bias", name)
    name = re.sub(r"model.encoder.layers.(\d+).fc2.weight", r"encoder.blocks.\1.mlp.2.weight", name)
    name = re.sub(r"model.encoder.layers.(\d+).fc2.bias", r"encoder.blocks.\1.mlp.2.bias", name)

    name = name.replace("model.encoder.layer_norm.weight", "encoder.ln_post.weight")
    name = name.replace("model.encoder.layer_norm.bias", "encoder.ln_post.bias")

    # Decoder self-attn
    name = re.sub(r"model.decoder.layers.(\d+).self_attn.q_proj.weight", r"decoder.blocks.\1.attn.query.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).self_attn.q_proj.bias", r"decoder.blocks.\1.attn.query.bias", name)
    name = re.sub(r"model.decoder.layers.(\d+).self_attn.k_proj.weight", r"decoder.blocks.\1.attn.key.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).self_attn.v_proj.weight", r"decoder.blocks.\1.attn.value.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).self_attn.v_proj.bias", r"decoder.blocks.\1.attn.value.bias", name)
    name = re.sub(r"model.decoder.layers.(\d+).self_attn.out_proj.weight", r"decoder.blocks.\1.attn.out.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).self_attn.out_proj.bias", r"decoder.blocks.\1.attn.out.bias", name)

    name = re.sub(r"model.decoder.layers.(\d+).self_attn_layer_norm.weight", r"decoder.blocks.\1.attn_ln.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).self_attn_layer_norm.bias", r"decoder.blocks.\1.attn_ln.bias", name)

    # Decoder cross-attn
    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn.q_proj.weight", r"decoder.blocks.\1.cross_attn.query.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn.q_proj.bias", r"decoder.blocks.\1.cross_attn.query.bias", name)
    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn.k_proj.weight", r"decoder.blocks.\1.cross_attn.key.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn.v_proj.weight", r"decoder.blocks.\1.cross_attn.value.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn.v_proj.bias", r"decoder.blocks.\1.cross_attn.value.bias", name)
    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn.out_proj.weight", r"decoder.blocks.\1.cross_attn.out.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn.out_proj.bias", r"decoder.blocks.\1.cross_attn.out.bias", name)

    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn_layer_norm.weight", r"decoder.blocks.\1.cross_attn_ln.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).encoder_attn_layer_norm.bias", r"decoder.blocks.\1.cross_attn_ln.bias", name)

    # Decoder FFN
    name = re.sub(r"model.decoder.layers.(\d+).fc1.weight", r"decoder.blocks.\1.mlp.0.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).fc1.bias", r"decoder.blocks.\1.mlp.0.bias", name)
    name = re.sub(r"model.decoder.layers.(\d+).fc2.weight", r"decoder.blocks.\1.mlp.2.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).fc2.bias", r"decoder.blocks.\1.mlp.2.bias", name)
    name = re.sub(r"model.decoder.layers.(\d+).final_layer_norm.weight", r"decoder.blocks.\1.mlp_ln.weight", name)
    name = re.sub(r"model.decoder.layers.(\d+).final_layer_norm.bias", r"decoder.blocks.\1.mlp_ln.bias", name)

    name = name.replace("model.decoder.layer_norm.weight", "decoder.ln.weight")
    name = name.replace("model.decoder.layer_norm.bias", "decoder.ln.bias")
    
    name = name.replace("model.decoder.proj_out.weight", "proj_out.weight")
    name = name.replace("model.decoder.proj_out.bias", "proj_out.bias")


    return name

def convert_whisper_model(hf_path, save_path):
    print(f"🔧 Loading HuggingFace model from: {hf_path}")
    model = WhisperForConditionalGeneration.from_pretrained(hf_path)
    state_dict = model.state_dict()

    print("🔁 Renaming parameters to OpenAI Whisper format...")
    converted_state_dict = {}
    
    for k, v in state_dict.items():
        if k == "proj_out.weight":
            continue
        new_k = rename_param(k)
        converted_state_dict[new_k] = v

    print("📐 Building dims...")
    config = model.config
    if "v3" in hf_path:
        dim = 128
    else:
        dim = 80
    dims = ModelDimensions(
        n_mels=dim,
        n_vocab=config.vocab_size,
        n_audio_state=config.d_model,
        n_audio_layer=config.encoder_layers,
        n_audio_head=config.encoder_attention_heads,
        n_text_state=config.d_model,
        n_text_layer=config.decoder_layers,
        n_text_head=config.decoder_attention_heads,
        n_audio_ctx=config.max_source_positions,
        n_text_ctx=config.max_target_positions
    )

    print(f"💾 Saving converted model to: {save_path}")
    torch.save({
        "model_state_dict": converted_state_dict,
        "dims": dict(vars(dims))  # 👈 转换为 dict 防止 key 报错
    }, save_path)

    print("✅ Done. Now you can run WhisperModelConverter.py on it.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", help="Path to HuggingFace Whisper model", default="")
    parser.add_argument("--save_path", help="Path to save .pt file", default="")
    args = parser.parse_args()

    convert_whisper_model(args.hf_path, args.save_path)
