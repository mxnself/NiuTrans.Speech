'''
Convert a whisper vocab to a NiuTrans.Speech vocab
Usage: python3 WhisperVocabConverter.py -o $niutransVocabPath -scale $model_scale
* `o` - Path to save the converted vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.
* `scale` - Scale of the model, must match to the converted model. Default: "large-v3"
'''

import argparse
from whisper.tokenizer import get_tokenizer
parser = argparse.ArgumentParser(
    description='Tool to convert fairseq checkpoint to NiuTrans.SMT vocab',
    )
parser.add_argument('-scale', required=True, type=str, default="large-v3",
    help='Scale of model.')
parser.add_argument('-o', required=True, type=str,
    help='Output vocab path.')
args = parser.parse_args()

if args.scale == "large-v3":
    multilingual_tokenizer = get_tokenizer(multilingual=True, num_languages=100)     # v3 100, v2 99
    num = 51866     # v3 51866, v2 51865
else:
    multilingual_tokenizer = get_tokenizer(multilingual=True, num_languages=99)     # v3 100, v2 99
    num = 51865     # v3 51866, v2 51865

ids = [i for i in range(num)]
vocab_utf = [str(multilingual_tokenizer.encoding._core_bpe.decode_bytes([i]))[2:-1] for i in ids]
print(vocab_utf)

with open(args.o, "w") as f:  
    f.writelines("{}\t{}\n".format(len(vocab_utf), 0))
    for i in range(num):
        print("{}\t{}".format(vocab_utf[i], ids[i]))
        f.writelines("{}\t{}\n".format(vocab_utf[i], ids[i]))
