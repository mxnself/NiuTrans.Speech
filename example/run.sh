#convert the model
python3 ../tools/WhisperModelConverter.py -i "../model/tiny.pt" -o "../model/whisper_tiny.bin" -scale "tiny" -type "fp32"

# convert the vocab
python3 ../tools/WhisperVocabConverter.py -o "../model/tiny_vocab" -scale "tiny"

# generate mel filters. n_mels: v3 is 128 and v2 is 80
python ../tools/GenFilter.py -n_mels 80 -o "../model/tiny_filters.csv"

# run Niutrans.Speech
../bin/NiuTrans.Speech -config ../example/config.txt
