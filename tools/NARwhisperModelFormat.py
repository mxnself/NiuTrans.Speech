import yaml
from transformers import WhisperConfig
import whisper
import argparse
import torch

key_map = {
    "attention_heads": "n_text_head",
    "num_blocks": "n_text_layer",
    "idim": "n_pre_state",
    "linear_units": "decFFNHiddenDim",
    "add_layers": "adapter_layer",
}

# 读取 YAML 文件
def get_config_from_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        encoder_conf = data["encoder_conf"]
        decoder_conf = data["decoder_conf"]
        predictor_conf = data["predictor_conf"]
        model_conf = data["model_conf"]
        token_list = data["token_list"]
    return model_conf, encoder_conf, decoder_conf, predictor_conf, token_list

def get_config(args):
    model_conf, encoder_conf, decoder_conf, predictor_conf, token_list = get_config_from_file(args.c)
    config_dicts = {}
    # get encoder config
    model_type = encoder_conf["whisper_model"]
    model = whisper.load_model("{}".format(model_type))
    config = model.dims
    for keys, values in config.__dict__.items():
        config_dicts[keys] = values
    for keys, values in model_conf.items():
        if keys in key_map.keys():
            config_dicts[key_map[keys]] = values
    for keys, values in decoder_conf.items():
        if keys in key_map.keys():
            config_dicts[key_map[keys]] = values
    for keys, values in predictor_conf.items():
        if keys in key_map.keys():
            config_dicts[key_map[keys]] = values
        else:
            config_dicts[keys] = values
    config_dicts["n_vocab"] = len(token_list)
    print(config_dicts)
    return config_dicts

replace_map = {
        "encoders.": "",
        "decoders": "blocks",
        "self_attn": "attn",
        "linear_v": "value",
        "linear_q": "query",
        "linear_k": "key",
        "linear_out": "out",
        "norm1": "attn_ln",
        "src_attn": "cross_attn",
        "norm2": "cross_attn_ln",
        "feed_forward": "mlp",
        "w_1": "0",
        "w_2": "2",
        "norm3": "mlp_ln",
        "embed.0": "token_embedding",
        "after_norm": "ln",
        "output_layer": "output",
        "cif_output": "output",
        "add_layers1": "adapter.blocks"
    }

def get_model_state(input_model):
    model_state = {}
    model = torch.load(input_model, map_location='cpu')
    for keys, values in model['state_dict'].items():
        if "add_layers1" in keys:
            keys = keys.replace("norm2", "norm3")
        if keys == "after_norm.weight":
            keys = "adapter.ln.weight"
        if keys == "after_norm.bias":
            keys = "adapter.ln.bias"
        for replace in replace_map.keys():
            if replace in keys:
                keys = keys.replace(replace, replace_map[replace]) 
        model_state[keys] = values

    return model_state
    
def main(args):
    input_model = args.i
    output_model = args.o
    model_state = get_model_state(input_model)
    config = get_config(args)
    model = {}
    model['dims'] = config
    model['model_state_dict']=model_state
    #print(model)
    torch.save(model, args.o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Tool to convert fairseq checkpoint to NiuTrans.SMT model',
    )
    parser.add_argument('-i', type=str, default='./model.pt',
                        help='Input model path.')
    parser.add_argument('-o', type=str, default='./narwhisper.pt',
                        help='Output model path.')
    parser.add_argument('-c', type=str, default='/mnt/maxiangnan/NiuTrans.Speech-main/tools/model/config.yaml',
                        help='Output config path.')
    args = parser.parse_args()
    main(args)
