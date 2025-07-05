NiuTrans.Speech
- [NiuTrans.Speech](#niutransspeech)
  - [Features](#features)
  - [Recent Updates](#recent-updates)
  - [Installation](#installation)
    - [Files](#files)
    - [Requirements](#requirements)
    - [Build from Source](#build-from-source)
      - [Configure with cmake](#configure-with-cmake)
      - [Compile on Linux](#compile-on-linux)
      - [Compile on Windows](#compile-on-windows)
    - [Converting Models](#converting-models)
    - [Generate Filter Files](#generate-filter-files)
    - [Running](#running)
      - [Modify your configs.](#modify-your-configs)
      - [Runing NiuTrans.Speech in bash.](#runing-niutransspeech-in-bash)
    - [Testing](#testing)
  - [A Model Zoo](#a-model-zoo)
  - [Papers](#papers)
  - [Team Members](#team-members)
  - [Other Open Resources](#other-open-resources)

# NiuTrans.Speech

This project aim to build a toolkit for speech-to-text task, e.g. Speech Recognition. It is powered by the [NiuTensor](https://github.com/NiuTrans/NiuTensor).

## Features

NiuTrans.Speech now is a lightweight and efficient speech recognition system. Its main features are:

It can make inference with NAR whisper models or AR whisper models.
Few dependencies. It is implemented with pure C++ and CUDA, and all dependencies are optional.
High efficiency. It is heavily optimized for fast decoding.
Flexible running modes. The system can run with various systems (Linux vs. Windows, etc.).


## Recent Updates

We support the multilingual speech Recognition now! Try it now (https://asr-demo.niutrans.com)!

## Installation

### Files

_Files structure-----------------------------------------------------------------------------------_

_|---- config (cofig settings of running)_

_|---- data (raw audio files)_

_|---- example (several examples of the project)_

_|---- model (converted model & vocab & filter)_

_|---- source (source file written in C++)_

_|---- output (decoding results)_

_|---- tools (tools of pre/post-process script)_
_--------------------------------------------------------------------------------------------_

### Requirements

You need to prepare C++(for running NiuTrans.Speech) & Python(for converting models) environments for this project first.

**Requirements for compile**
- OS: Linux or Windows
- [GCC=8.X.X](https://gcc.gnu.org/) (>=5.4.0 should be ok)
- [CUDAToolkit](https://developer.nvidia.com/cuda-toolkit-archive) with [cuDNN](https://developer.nvidia.com/cudnn-downloads) (>= 11.3)
- [Cmake=3.2X.X](https://cmake.org/download/)

**Other Requirements (python)**
- torch
- numpy
- librosa
- whisper

### Build from Source

#### Configure with cmake

The default configuration only enables compiling for the **GPU** version.

1. Download the code

```bash
git clone https://github.com/NiuTrans/NiuTensor.Speech.git
cd ./NiuTensor.Speech
```

1. Run cmake (with CUDA)
   
```bash
mkdir build && cd build
cmake ../ -DUSE_CUDA=ON -DUSE_CUDNN=ON -DGPU_ARCH=P -DCUDA_TOOLKIT_ROOT='$YOUR_CUDA_PATH'
```

At present, the project is exclusively tested on Pascal architecture. If your GPU employs other architectures, it may encounter compatibility issues.

You can use `nvidia-smi -q | grep "Architecture"` to search your 'DGPU_ARCH'. More details please refer to [NiuTensor](https://github.com/NiuTrans/NiuTensor#linux%E5%92%8Cmacos)


#### Compile on Linux

```bash
make -j 32
chmod +x ../bin/NiuTrans.Speech
```

#### Compile on Windows

Add ``-A 64`` to the cmake command and it will generate a visual studio project on windows, i.e., ``NiuTrans.Speech.sln`` so you can open & build it with Visual Studio (>= Visual Studio 2019).

If it succeeds, you will get an executable file **`NiuTrans.Speech`** in the 'bin' directory.


### Converting Models

Before running, you need to convert the model and the vocab as [Converts](#converting-models-from-whisper) and generate filter files as [GenFilter](#generate-filter-files) with the following steps.

If you want to use Transformer models, please run this step; otherwise, you can skip it.
```python
python3 tools/TransformerConvert2Whisper.py -hf_path $whisperHFCheckpoint -save_path $whisperCheckpoint
```

1. Convert parameters of a single whisper model.

```python
python3 tools/WhisperModelConverter.py -i $whisperCheckpoint -o $niutransModel -scale $modelScale -type $dataType
```

Description:

* `i` - Path of the whisper checkpoint.
* `o` - Path to save the converted model parameters. All parameters are stored in a binary format. 
* `scale` - Scale of the whisper model, must match the whisper checkpoint scale. like: "large-v3".
* `type (optional)` - Save the parameters with 32-bit data type of 16-bit data type. Default: "fp32".


2. Convert the vocabulary. Default setting is for "large-v3".

```python
python3 tools/WhisperVocabConverter.py -o $niutransVocabPath -scale $model_scale
```


Description:

* `o` - Path to save the converted vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.
* `scale` - Scale of the model, must match to the converted model. like: "large-v3"

*the source language vocabulary is whisper.tokenizer.*


For NAR-whisper models, you should run the following files to convert models:

```python
python3 tools/NARwhisperModelFormat.py -i $NARmodelpath -o $NARwhisper.pt -c $config_file
python3 tools/NARwhisperModelConverter.py -i $NARwhisper.pt -o $NARwhisper.bin -scale $model_scale
python3 tools/NARwhisperVocabConverter.py -i $config_file -o $VocabPath
```

### Generate Filter Files

```python
python3 tools/GenFilter.py -n_mels $num_mels -o $output_path
```

Description:

* `n_mels` - Num of mel filters.  large-v3 is 128 and others are 80. Default: 128.
* `o` - Path to save the mel filters.


### Running

#### Modify your configs.

Before running, please modify configs in `./config/config.txt`. If you get some bugs that cannot be fixed, consider using an absolute path.

Details about the config.text:

* `dev` - Device id (>= 0 for GPUs). Default: 0.
* `model` - Path of the model.
* `tgtvocab` - Path of the target language vocabulary.
* `customFilter` - Path of the Mel-filter file.
* `inputAudio` - Path of the input audio/features.
* `input` - Path of the input file, .
* `output` - Path of the output file to be saved.
* `maxlen` - The max length for generation. Default: 224.
* `lenalpha` - The alpha parameter controls the length preference. Default: 1.0.
* `lang` - Language of the target output sequence.
* `beam` - The beam size for generation. Default: 1.
* `sbatch` - The batch size of sentences. Default: 1.
* `wbatch` - The batch size of words. Default: 300000.
* `numBins` - Dim of the Feature. large-v3 is 128 and others are 80.
* `numLanguage` - Num of Languages. large-v3 is 100 and others are 99.
* `NAR` - Set NAR to 1 to run NAR models. Otherwise, omit this optio to run AR models.

```Note: Only one of inputAudio or input should be specified.```

#### Runing NiuTrans.Speech in bash.

```bash
cd /PROJECT_DIR/bin/
./NiuTrans.ST -config ../config/config.txt
```

### Testing

We provide  an script `./example/run.sh` for testing. The default config is for "large-v3" model.

```bash
    bash ./example/run.sh $path_for_large_v3
```

**NOTICE!** Before testing, you should download the [large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt) to `path_for_large_v3` and run the convert files.

##  A Model Zoo

We provide several whisper links to run the project. For most choices, please go to [whisper](https://github.com/openai/whisper/blob/main/whisper/__init__.py).
| Model                                                                                                                                           | Type | Config File | Tested |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ----------- | ------ |
| [tiny](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)         | AR   | –           |        |
| [base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)         | AR   | –           |        |
| [small](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt)       | AR   | –           |        |
| [medium](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/medium.pt)     | AR   | –           | ✓      |
| [large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt) | AR   | –           | ✓      |
| [large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt) | AR   | –           | ✓      |
| [NARwhisper-zh](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt) | NAR   | [config.yaml]()          | ✓      |
| [NARwhisper-en](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt) | NAR   | [config.yaml]()           | ✓      |
| [NARwhisper-mix](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt) | NAR   | [config.yaml]()           | ✓      |


## Papers
- Coming Soon.

## Team Members

This project is maintained by a joint team from NiuTrans Research and NEU NLP Lab. Current maintenance members are *Yuhao Zhang, Xiangnan Ma, Kaiqi Kou, Erfeng He, Siming Wu, Yi Zhang, Chenghao Gao, Qing Yang.*

We'd like thank *Prof. XIAO, Tong* and *Prof. ZHU, Jingbo* for their support.

Feel free to contact niutrans@mail.neu.edu.cn if you have any questions.

## Other Open Resources
+ NiuTrans OpenSource(https://opensource.niutrans.com/home/index.html)(https://github.com/NiuTrans)
+ NiuTensor (https://github.com/NiuTrans/NiuTensor)
+ NiuTrans.NMT (https://github.com/NiuTrans/NiuTrans.NMT)
