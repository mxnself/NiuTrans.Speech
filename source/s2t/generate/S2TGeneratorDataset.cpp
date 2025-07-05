/* NiuTrans.S2T - an open-source speech to text system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
  * $Created by: Yuhao Zhang (yoohao.zhang) 2023-10-14
  */

#include <iostream>
#include <algorithm>
#include "S2TGeneratorDataset.h"
#include "../../niutensor/tensor/XTensor.h"
#include <unordered_map>
#include "../../niutensor/tensor/core/CHeader.h"

using namespace nts;

/* the S2T namespace */
namespace s2t {

/* transfrom a speech to a sequence */
TripleSample* S2TGeneratorDataset::LoadSample(XTensor* s)
{
    TripleSample* sample = new TripleSample(s);
    return sample;
}

/* transfrom a speech to a sequence */
TripleSample* S2TGeneratorDataset::LoadSample(string s)
{
    XTensor batchEnc;
    INT32 audio_length;
    strcpy(config->extractor.inputAudio, s.c_str());;
    struct FbankOptions fOpts(*config);
    class FbankComputer computer(fOpts);
    oft = new OfflineFeatureTpl<FbankComputer>(fOpts);
    oft->Read();
    oft->ComputeFeatures(oft->Data().Data(), oft->Data().SampFreq(), 1.0, &batchEnc, audio_length);
    batchEnc.FlushToDevice(config->common.devID);
    TripleSample* sample = new TripleSample(batchEnc);
    return sample;
}

/* transfrom a speech and a line to the sequence separately */
TripleSample* S2TGeneratorDataset::LoadSample(XTensor* s, string line)
{
    const string delimiter = " ";

    /* load tokens and transform them to ids */
    vector<string> srcTokens = SplitString(line, delimiter,
        config->model.maxSrcLen - 1);

    IntList* srcSeq = new IntList(int(srcTokens.size()));
    TripleSample* sample = new TripleSample(s);

    return sample;
}

/* this is a place-holder function to avoid errors */
TripleSample* S2TGeneratorDataset::LoadSample()
{
    return nullptr;
}

/* read data from a file to the buffer */
bool S2TGeneratorDataset::LoadBatchToBuf()
{
    int id = 0;
    ClearBuf();
    emptyLines.Clear();

    string line;
    while (getline(*ifp, line) && id < config->common.bufSize) {
        /* handle empty lines */

        if (line.size() > 0) {
            TripleSample* sequence = LoadSample(line);
            sequence->index = id;
            buf->Add(sequence);
        }
        else {
            emptyLines.Add(id);
        }
        id++;
    }

    SortByAudioLengthDescending();
    XPRINT1(0, stderr, "[INFO] loaded %d sentences\n", appendEmptyLine ? id - 1 : id);

    return true;
}

/* constructor */
S2TGeneratorDataset::S2TGeneratorDataset()
{
    ifp = NULL;
    appendEmptyLine = false;
}

/*
load a batch of sequences from the buffer to the host for translating
>> inputs - a list of input tensors (batchEnc and paddingEnc)
   batchEnc - a tensor to store the batch of input
   paddingEnc - a tensor to store the batch of paddings
>> info - the total length and indices of sequences
*/
bool S2TGeneratorDataset::GetBatchSimple(XList* inputs, XList* info)
{
    int realBatchSize = 1;

    /* get the maximum sequence length in a mini-batch */
    TripleSample* longestSample = (TripleSample*)(buf->Get(bufIdx));
    int maxLen = longestSample->fLen;

    /* we choose the max-token strategy to maximize the throughput */
    while (realBatchSize * (maxLen/30) * config->inference.beamSize < config->common.wBatchSize
        && realBatchSize < config->common.sBatchSize) {
        realBatchSize++;
    }
    realBatchSize = MIN(realBatchSize, config->common.sBatchSize);

    /* make sure the batch size is valid */
    realBatchSize = MIN(int(buf->Size()) - bufIdx, realBatchSize);
    realBatchSize = MAX(2 * (realBatchSize / 2), realBatchSize % 2);

    CheckNTErrors(maxLen != 0, "Invalid length");

    int* totalLength = (int*)(info->Get(0));
    IntList* indices = (IntList*)(info->Get(1));
    *totalLength = 0;
    indices->Clear();

    /* right padding */
    /* TODO!!! Check the length of audio */
    XTensor* batchEnc = (XTensor*)(inputs->Get(0));
    XTensor* paddingEnc = (XTensor*)(inputs->Get(1));
    InitTensor3D(batchEnc, realBatchSize, maxLen, config->s2tmodel.fbank, X_FLOAT, config->common.devID);

    for (int i = 0; i < realBatchSize; ++i) {
        TripleSample* sample = (TripleSample*)(buf->Get(bufIdx + i));
        _SetDataIndexed(batchEnc, &sample->audioSeq, 0, i);
        indices->Add(sample->index);
        *totalLength += sample->fLen;
    }

    bufIdx += realBatchSize;

    return true;
}

/*
constructor
>> myConfig - configuration of the NMT system
>> notUsed - as it is
*/
void S2TGeneratorDataset::Init(S2TConfig& myConfig, bool notUsed)
{
    config = &myConfig;
    
    /* load the source and target vocabulary */
    tgtVocab.Load(config->common.tgtVocabFN);

    /* share the source and target vocabulary */
    if (strcmp(config->common.srcVocabFN, "") != 0)
    {
        if (strcmp(config->common.srcVocabFN, config->common.tgtVocabFN) == 0)
            srcVocab.CopyFrom(tgtVocab);
        else
            srcVocab.Load(config->common.srcVocabFN);
        srcVocab.SetSpecialID(config->model.sos, config->model.eos,
            config->model.pad, config->model.unk);
    }
    
    tgtVocab.SetSpecialID(config->model.sos, config->model.eos,
        config->model.pad, config->model.unk, config->whisperdec.numLanguage);

    /* transcripe the content in a file */
    if (strcmp(config->inference.inputFN, "") != 0) {
        ifp = new ifstream(config->inference.inputFN);
        CheckNTErrors(ifp, "Failed to open the input file");
    }
    /* transcripe the content in stdin */
    else
        ifp = &cin;

    LoadBatchToBuf();
}

/* check if the buffer is empty */
bool S2TGeneratorDataset::IsEmpty() {
    if (bufIdx < buf->Size())
        return false;
    return true;
}

/* de-constructor */
S2TGeneratorDataset::~S2TGeneratorDataset()
{
    if (ifp != NULL && strcmp(config->inference.inputFN, "") != 0) {
        ((ifstream*)(ifp))->close();
        delete ifp;
    }
    if (oft != NULL) {
        delete oft;
        oft = NULL;
    }
}

} /* end of the s2t namespace */