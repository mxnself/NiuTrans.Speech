'''
Convert a whisper model to NiuTrans.Speech model (FP32).
Usage: python3 WhisperModelConverter.py -i $whisperCheckpoint -o $niutransModel -scale $modelScale -type $dataType
* `i` - Path of the whisper checkpoint.
* `o` - Path to save the converted model parameters. All parameters are stored in a binary format. 
* `scale` - Scale of the whisper model, must match the whisper checkpoint scale. Default: "large-v3".
* `type (optional)` - Save the parameters with 32-bit data type. Default: "fp32".
'''


import torch
import argparse
import argparse
import numpy as np
from glob import glob
from struct import pack
import os
from collections import OrderedDict
import copy

ffnHiddenDim = {"tiny": 1536,
		        "base": 2048,
		        "small": 3072,
                "medium": 4096,
                "large": 5120,
                "large-v2": 5120,
		        "large-v3": 5120}

def checkByteFile(file, x, Info):
    from struct import pack, unpack
    print("!!!!!!!!! Now is Checking {} Data !!!!!!!!!!".format(Info))
    ## check 
    with open(file, 'rb') as f:
        x_check = unpack("f" * x.numel(), f.read())
        x_check = np.array(x_check)

        print(x_check, x_check.shape)
        x_true = x.contiguous().view(-1).cpu().numpy()
        print(x_true, x_true.shape)
        gap = np.abs(x_true - x_check)
        print("max: {}, min: {}, mean: {}".format(np.max(gap), np.min(gap), np.mean(gap)))
        greather = gap > 1e-3
        positions = np.where(greather)[0]
        print(positions, positions.shape, (positions.shape[0] / x_true.shape[0]) * 100)
    print("--------- Finish Checking {} Data ---------".format(Info))

def get_params_list(config, scale):

    l = []

    if not config['bool']['decoderOnly']:
        # adapter
        for i in range(config['int']['adapterLayerNum']):
            l.append('adapter.blocks.{}.attn.key.weight'.format(i))
            l.append('adapter.blocks.{}.attn.key.bias'.format(i))
            l.append('adapter.blocks.{}.attn.value.weight'.format(i))
            l.append('adapter.blocks.{}.attn.value.bias'.format(i))
            l.append('adapter.blocks.{}.attn.query.weight'.format(i))
            l.append('adapter.blocks.{}.attn.query.bias'.format(i))
            l.append('adapter.blocks.{}.attn.out.weight'.format(i))
            l.append('adapter.blocks.{}.attn.out.bias'.format(i))
            l.append('adapter.blocks.{}.attn_ln.weight'.format(i))
            l.append('adapter.blocks.{}.attn_ln.bias'.format(i))
            l.append('adapter.blocks.{}.mlp.0.weight'.format(i))
            l.append('adapter.blocks.{}.mlp.0.bias'.format(i))
            l.append('adapter.blocks.{}.mlp.2.weight'.format(i))
            l.append('adapter.blocks.{}.mlp.2.bias'.format(i))
            l.append('adapter.blocks.{}.mlp_ln.weight'.format(i))
            l.append('adapter.blocks.{}.mlp_ln.bias'.format(i)) 
        l.append('adapter.ln.weight')
        l.append('adapter.ln.bias')

        # conv1d
        for i in range(1,3):
            l.append('encoder.conv{}.weight'.format(i))
            l.append('encoder.conv{}.bias'.format(i))

        # encoder block
        for i in range(config['int']['encLayerNum']):
            l.append('encoder.blocks.{}.attn.key.weight'.format(i))
            l.append('encoder.blocks.{}.attn.value.weight'.format(i))
            l.append('encoder.blocks.{}.attn.value.bias'.format(i))
            l.append('encoder.blocks.{}.attn.query.weight'.format(i))
            l.append('encoder.blocks.{}.attn.query.bias'.format(i))
            l.append('encoder.blocks.{}.attn.out.weight'.format(i))
            l.append('encoder.blocks.{}.attn.out.bias'.format(i))
            l.append('encoder.blocks.{}.attn_ln.weight'.format(i))
            l.append('encoder.blocks.{}.attn_ln.bias'.format(i))
            l.append('encoder.blocks.{}.mlp.0.weight'.format(i))
            l.append('encoder.blocks.{}.mlp.0.bias'.format(i))
            l.append('encoder.blocks.{}.mlp.2.weight'.format(i))
            l.append('encoder.blocks.{}.mlp.2.bias'.format(i))
            l.append('encoder.blocks.{}.mlp_ln.weight'.format(i))
            l.append('encoder.blocks.{}.mlp_ln.bias'.format(i))
        if config['bool']['encFinalNorm']:
            l.append('encoder.ln_post.weight')
            l.append('encoder.ln_post.bias')
            
        # for i in range(config['int']['decLayerNum']):
            

        # decoder block
        for i in range(config['int']['decLayerNum']):
            l.append('decoder.blocks.{}.attn.key.weight'.format(i))
            l.append('decoder.blocks.{}.attn.key.bias'.format(i))
            l.append('decoder.blocks.{}.attn.value.weight'.format(i))
            l.append('decoder.blocks.{}.attn.value.bias'.format(i))
            l.append('decoder.blocks.{}.attn.query.weight'.format(i))
            l.append('decoder.blocks.{}.attn.query.bias'.format(i))
            l.append('decoder.blocks.{}.attn.out.weight'.format(i))
            l.append('decoder.blocks.{}.attn.out.bias'.format(i))
            l.append('decoder.blocks.{}.attn_ln.weight'.format(i))
            l.append('decoder.blocks.{}.attn_ln.bias'.format(i))
            if not config['bool']['decoderOnly']:
                # cross attention
                l.append('decoder.blocks.{}.cross_attn.key.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn.key.bias'.format(i))
                l.append('decoder.blocks.{}.cross_attn.value.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn.value.bias'.format(i))
                l.append('decoder.blocks.{}.cross_attn.query.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn.query.bias'.format(i))
                l.append('decoder.blocks.{}.cross_attn.out.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn.out.bias'.format(i))
                l.append('decoder.blocks.{}.cross_attn_ln.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn_ln.bias'.format(i))

            l.append('decoder.blocks.{}.mlp.0.weight'.format(i))
            l.append('decoder.blocks.{}.mlp.0.bias'.format(i))
            l.append('decoder.blocks.{}.mlp.2.weight'.format(i))
            l.append('decoder.blocks.{}.mlp.2.bias'.format(i))
            l.append('decoder.blocks.{}.mlp_ln.weight'.format(i))
            l.append('decoder.blocks.{}.mlp_ln.bias'.format(i))
        if config['bool']['decFinalNorm']:
            l.append('decoder.ln.weight')
            l.append('decoder.ln.bias')

        if not config['bool']['shareEncDecEmb']:
            l.append('decoder.token_embedding.weight')

        #l.append('decoder.positional_embedding')
        l.append('decoder.output.weight')
        l.append('decoder.output.bias')

        #l.append('ctc.ctc_lo.weight')
        #l.append('ctc.ctc_lo.bias')

        l.append('predictor.cif_conv1d.weight')
        l.append('predictor.cif_conv1d.bias')
        l.append('predictor.output.weight')
        l.append('predictor.output.bias')

    return l

def get_whisper_parameters(params, config, scale):
    '''
    get flattend transformer model parameters
    '''

    l = get_params_list(config, scale)

    p = []

    for k in params:
        assert k in l, "{} should be include".format(k)

    for name in l:
        name_keys = name.split('.')
        if "weight" in name_keys and ("attn" in name_keys or "cross_attn" in name_keys or "mlp" in name_keys or "output" in name_keys):
            
            # print(name, params[name].shape)
            p.append(params[name].t())
            # if "output" in name_keys:
            #     x = params[name].t()
            #     print('aaaaa',x.size())
        else:
            # print(name, params[name].shape)
            p.append(params[name])
    return p, l

def get_whisper_config(config, scale):
    # 12 booleans
    config_dict_bool = OrderedDict({
        "encoderL1Norm": False,
        "decoderL1Norm": False,
        "useBigAtt": False,
        "decoderOnly": False,
        "encFinalNorm": True,
        "decFinalNorm": True,
        "encPreLN": True,
        "decPreLN": True,
        #  norm
        "useEncHistory": False,
        "useDecHistory": False,
        "shareEncDecEmb": False,
        "shareDecInputOutputEmb": True,
    })
    # 19 integers 
    config_dict_int = OrderedDict({
        "srcVocabSize": -1,
        "tgtVocabSize":config['n_vocab'],
        "sos": 1,
        "eos": 2,
        "pad": 0,   # 50256
        "unk": 3,
        "maxSrcLen":config['n_audio_ctx'],
        "maxTgtLen":config['n_text_ctx'],
        "maxRelativeLength": -1, 
        "fbank":config['n_mels'],
        "encEmbDim":config['n_audio_state'],
        "encLayerNum":config['n_audio_layer'],
        "encSelfAttHeadNum":config['n_audio_head'],
        "encFFNHiddenDim": ffnHiddenDim[scale],
        "decEmbDim":config['n_text_state'],
        "decLayerNum":config['n_text_layer'],
        "decSelfAttHeadNum":config['n_text_head'],
        "encDecAttHeadNum":config['n_text_head'],
        "decFFNHiddenDim": config['decFFNHiddenDim'],
        "fnnActFunType": 1,
        "idim": config["n_pre_state"],
        "adapterLayerNum": config['adapter_layer'],
    })
    # 3 floats
    config_dict_float = OrderedDict({
        "dropout": float(0.0),
        "ffnDropout": float(0.0),
        "attDropout": float(0.0),
        "threshold": float(config["threshold"]),
        "tail_threshold": float(config["tail_threshold"]),
    })
    
    whisper_config = {"bool":config_dict_bool, "int":config_dict_int, "float":config_dict_float}

    return whisper_config

def fusion_attention_params(names, params):
    assert len(names) == len(params)
    new_names = []
    new_params = []
    
    cache_names = []
    cache_params = []
    for (name, param) in zip(names, params):
        
        if "attn." in name and "out." not in name:
            
            cache_names.append(name)
            cache_params.append(param)
            
            if "query.bias" in name:
                assert len(cache_names) == 5 and len(cache_params) == 5
                for (cache_name, cache_param) in zip(cache_names, cache_params):
                    print(cache_name, cache_param.shape)
                    
                print("--------------------------------------")
                k_weight, v_weight, v_bias, q_weight, q_bias = cache_params
                k_bias = torch.zeros_like(q_bias)
                
                new_name_bias = name.replace("query", "qkv")
                new_name_weight = new_name_bias.replace("bias", "weight")
                
                new_bias = torch.stack((q_bias, k_bias, v_bias), dim=0)
                new_weight = torch.stack((q_weight, k_weight, v_weight), dim=0)
                
                print("+++++++++++++++++++++++++++++++++++++++")
                new_names.append(new_name_weight)
                new_params.append(new_weight)
                new_names.append(new_name_bias)
                new_params.append(new_bias)
                
                cache_names = []
                cache_params = []   
            
        else:
            new_names.append(name)
            new_params.append(param)

    return new_params

def fusion_attention_params_new(names, params):
    assert len(names) == len(params)
    new_names = []
    new_params = []
    
    cache_names = []
    cache_params = []
    for (name, param) in zip(names, params):
        
        if "attn." in name and "out." not in name:
            
            cache_names.append(name)
            cache_params.append(param)
            
            if "query.bias" in name:
                #assert len(cache_names) == 5 and len(cache_params) == 5
                assert len(cache_names) == len(cache_params)
                for (cache_name, cache_param) in zip(cache_names, cache_params):
                    print(cache_name, cache_param.shape)
                    
                if len(cache_names) == 5:
                    k_weight, v_weight, v_bias, q_weight, q_bias = cache_params
                    k_bias = torch.zeros_like(q_bias)
                elif len(cache_names) == 6:
                    k_weight, k_bias, v_weight, v_bias, q_weight, q_bias = cache_params
                else:
                    print("length error")
                    exit(-1)

                print("--------------------------------------")
                if "cross_attn" in name:

                    new_name_bias = name.replace("query", "fusion")
                    new_name_weight = new_name_bias.replace("bias", "weight")
                    
                    new_bias = torch.cat((k_bias, v_bias), dim=0)
                    new_weight = torch.cat((k_weight, v_weight), dim=1)

                    new_names.append(name.replace("bias", "weight"))
                    new_params.append(q_weight)
                    new_names.append(name)
                    new_params.append(q_bias)

                    new_names.append(new_name_weight)
                    new_params.append(new_weight)
                    new_names.append(new_name_bias)
                    new_params.append(new_bias)
                    print(new_name_weight, new_weight.shape)
                    print(new_name_bias, new_bias.shape)

                else:
                    new_name_bias = name.replace("query", "qkv")
                    new_name_weight = new_name_bias.replace("bias", "weight")
                    
                    new_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                    new_weight = torch.cat((q_weight, k_weight, v_weight), dim=1)

                    new_names.append(new_name_weight)
                    new_params.append(new_weight)
                    new_names.append(new_name_bias)
                    new_params.append(new_bias)
                
                print("+++++++++++++++++++++++++++++++++++++++")
                
                
                cache_names = []
                cache_params = []   
                
                
            
        else:
            new_names.append(name)
            new_params.append(param)
    
    return new_params, new_names

def main(args):

    with torch.no_grad():
        
        mode = "fp32"
        model_scale= args.scale
        model_file = args.i
        taget_file = args.o

        fusion_attn = True
       
        assert os.path.exists(model_file), "File not exist. {}".format(model_file)
        assert model_scale in model_file, "scale not match. {}".format(model_file)

        print("source model: \'{}\'\ntarget model: \'{}\'".format(model_file, taget_file))

        model = torch.load(model_file, map_location='cpu')

        print("+ checkpoint info: {}".format(model.keys()))

        print("+ Whisper Config")
        if model['dims'] is not None:
            config = model['dims']
            for k in config.keys():
                if isinstance(config[k], str):
                    config[k] = int(config[k])
                print("\t- {} : {}".format(k, config[k]))
        
        if model['model_state_dict'] is not None:
            params = model['model_state_dict']

        config = get_whisper_config(config, model_scale)
        print(config , len(config['int'].keys()) + len(config['bool'].keys()) + len(config['float'].keys()))
        

        origin_params = copy.deepcopy(params)

        # del params['decoder.positional_embedding']
        del params['encoder.positional_embedding']
        del params['ctc.ctc_lo.weight']
        del params['ctc.ctc_lo.bias']
        del params['embed_norm.weight']
        del params['embed_norm.bias']

        # print(len(params))
        # exit()

        params, params_name = get_whisper_parameters(params, config, model_scale)
        print("num of params: ", len(params))
        
        if fusion_attn:
            params, params_name = fusion_attention_params_new(params_name, params)
            print("num of params: ", len(params))
        
        # for param in params:
        #     print(param.shape)

        print(params_name)
        print(len(params_name))
        with open("/mnt/maxiangnan/NiuTrans.Speech-main/tools/Paras.txt", 'w') as fout:
            for idx,key in enumerate(params_name):
                fout.writelines(key+"  "+str(params[idx].size())+"\nstates\n"+str(params[idx])+"\n")

        
        print("----- Convert Mode -----")

        bool_config_list = list(config['bool'].values())
        int_config_list = list(config['int'].values())
        float_config_list = list(config['float'].values())
        print(bool_config_list, int_config_list, float_config_list)

        bool_configs = pack('?' * len(bool_config_list), *bool_config_list)
        int_configs = pack('i' * len(int_config_list), *int_config_list)
        float_configs = pack('f' * len(float_config_list), *float_config_list)

        # print("Didn't save file!")
        
        # package
        # assert False
        with open(taget_file, 'wb') as f:
            # part 1: model configurations
            f.write(bool_configs)
            f.write(int_configs)
            f.write(float_configs)

            # part 2: values of parameters (in FP32 or FP16)
            for p in params:
                if mode == 'fp32':
                    values = pack("f" * p.numel(), *
                                (p.contiguous().view(-1).cpu().numpy()))
                    f.write(values)
                elif mode == 'fp16':
                    values = pack(
                        "e" * p.numel(), *(p.contiguous().view(-1).cpu().numpy().astype(np.float16)))
                    f.write(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tool to convert fairseq checkpoint to NiuTrans.SMT model',
    )
    parser.add_argument('-i', required=True, type=str,
                        help='Input checkpoint path.')
    parser.add_argument('-o', required=True, type=str,
                        help='Output model path.')
    parser.add_argument('-scale', required=True, type=str, default='large-v3', help='scale of model.')
    parser.add_argument('-type', type=str,
                        help='Data type of the output model, FP32 (Default) or FP16',
                        default='fp32', choices=["fp32", "fp16"])
    args = parser.parse_args()
    main(args)
