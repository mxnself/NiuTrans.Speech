'''
Convert a whisper vocab to a NiuTrans.Speech vocab
Usage: python3 WhisperVocabConverter.py -o $niutransVocabPath -scale $model_scale
* `o` - Path to save the converted vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.
* `scale` - Scale of the model, must match to the converted model. Default: "large-v3"
'''
import yaml
import argparse
from whisper.tokenizer import get_tokenizer

parser = argparse.ArgumentParser(
    description='Tool to convert fairseq checkpoint to NiuTrans.SMT vocab',
    )
parser.add_argument('-i', required=True, type=str,
    help='token list file.')
parser.add_argument('-o', required=True, type=str,
    help='Output vocab path.')
args = parser.parse_args()

def get_config_from_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        token_list = data["token_list"]
    return  token_list

vocab_utf = get_config_from_file(args.i)
num = len(vocab_utf)
print(vocab_utf)

with open(args.o, "w") as f:  
    f.writelines("{}\t{}\n".format(len(vocab_utf), 0))
    for i in range(num):
        print("{}\t{}".format(vocab_utf[i], i))
        f.writelines("{}\t{}\n".format(vocab_utf[i], i))
