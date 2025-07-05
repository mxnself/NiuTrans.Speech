/* NiuTrans.S2T - an open-source speech-to-text system.
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
 * $Created by: Yuhao Zhang (yoohao.zhang@gmail.com) 2023-09-19
 */

#include "Generator.h"
#include <iostream>

using namespace nts;
using namespace std;

namespace s2t
{
    /* constructor */
    Generator::Generator()
    {
        oft = NULL;
        config = NULL;
        model = NULL;
        searcher = NULL;
        vocab = NULL;
        batchLoader=NULL;
    }

    /* de-constructor */
    Generator::~Generator()
    {
        if (config->inference.beamSize > 1)
            delete (S2TBeamSearch*)searcher;
        else
            delete (S2TGreedySearch*)searcher;
        searcher = nullptr;
        if (batchLoader)
            delete batchLoader;
        if (oft)
            delete oft;
        if (model)
            delete model;
        if (config)
            delete config;
        if (vocab)
            delete vocab;	
    }

    /* initialize the model */
    void Generator::Init(S2TConfig* myConfig, S2TModel* myModel, bool offline)
    {
        cout << "----- Generator Init -----" << endl;
        model = myModel;
        config = myConfig;
        vocab = new S2TVocab;
        
        vocab->Load(config->common.tgtVocabFN);
        vocab->SetSpecialID(config->model.sos, 
                            config->model.eos, 
                            config->model.pad, 
                            config->model.unk,
                            config->whisperdec.numLanguage);
        if (config->model.pad==-1){            
            vocab->InitMask(config->common.devID);
        }
        // vocab->ShowVocab();


        // disable the following code when using for service
        struct FbankOptions fOpts(*myConfig);
        class FbankComputer computer(fOpts);
        if (offline)
            oft = NULL;
        else
            oft = new OfflineFeatureTpl<FbankComputer>(fOpts);
        

        if (config->inference.beamSize > 1) {
            LOG("Inferencing with beam search (beam=%d, batchSize= %d sents | %d tokens, lenAlpha=%.2f, maxLenAlpha=%.2f) ",
                config->inference.beamSize, config->common.sBatchSize, config->common.wBatchSize,
                config->inference.lenAlpha, config->inference.maxLenAlpha);
            searcher = new S2TBeamSearch();
            ((S2TBeamSearch*)searcher)->Init(*myConfig, vocab);
        }
        else if (config->inference.beamSize == 1) {
            LOG("Inferencing with greedy search (batchSize= %d sents | %d tokens, maxLenAlpha=%.2f)",
                config->common.sBatchSize, config->common.wBatchSize, config->inference.maxLenAlpha);
            searcher = new S2TGreedySearch();
            ((S2TGreedySearch*)searcher)->Init(*myConfig, vocab);
            
        }
        else {
            CheckNTErrors(false, "Invalid beam size\n");
        }
        cout << "--- Generator Init End ---" << endl;
    }

    XTensor Generator::DecodingBatch(XTensor& batchEnc, XTensor& paddingEnc, IntList& indices, int audio_length)
    {
        // change single to batch
        bool isSingle = 0;
        if (batchEnc.order == 2) {
            isSingle = 1;
            batchEnc = Unsqueeze(batchEnc, 0, 1);
            paddingEnc = Unsqueeze(paddingEnc, 0, 1);
        }

        // clear cache
        if (model->decoder->selfAttCache) {
            delete[] model->decoder->selfAttCache;
            model->decoder->selfAttCache = new Cache[model->decoder->nlayer];
        }
        if (model->decoder->enDeAttCache) {
            delete[] model->decoder->enDeAttCache;
            model->decoder->enDeAttCache = new Cache[model->decoder->nlayer];
        }

        // begin decoding task
        int batchSize = batchEnc.GetDim(0);
        for (int i = 0; i < model->decoder->nlayer; ++i) {
            model->decoder->selfAttCache[i].miss = true;
            model->decoder->enDeAttCache[i].miss = true;
        }

        IntList** outputs = new IntList * [batchSize];
        for (int i = 0; i < batchSize; i++)
            outputs[i] = new IntList();

        /* greedy search */
        if (config->inference.beamSize == 1) {
            if (!config->model.NAR) {
                ((S2TGreedySearch*)searcher)->Search(model, batchEnc, paddingEnc, outputs);
            }
            // isNar == true
            else { 
                ((S2TGreedySearch*)searcher)->Search(model, batchEnc, paddingEnc, outputs, audio_length);
            }
        }
        else {
            XTensor score;
            ((S2TBeamSearch*)searcher)->Search(model, batchEnc, paddingEnc, outputs, score);
        }

        string tokens = "";
        vector<int> tokensId;
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputs[i]->count; j++) {
                tokensId.push_back(outputs[i]->GetItem(j));
            }
            tokens += vocab->DecodingWord(&tokensId);
            tokensId.clear();
            if (i < batchSize)
                tokens += "\n";
        }

        for (int i = 0; i < batchSize; i++)
            delete outputs[i];
        delete[] outputs;

        ofstream file(config->inference.outputFN, std::ios::out | std::ios::binary | std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
        }
        else {
            file << tokens;
            file.close();
        }
        

        if (isSingle) {
            /*TODO*/
            batchEnc = Squeeze(batchEnc);
        }
            
        return batchEnc;
            
    }

    bool Generator::Generate()
    {
        /* inputs */
        XTensor batchEnc;
        INT32 audio_length;

        if (oft)    // online inference support .wav file
        {
            oft->Read();
            /*TODO !!!*/
            oft->ComputeFeatures(oft->Data().Data(), oft->Data().SampFreq(), 1.0, &batchEnc, audio_length);
            batchEnc.FlushToDevice(config->common.devID);

            LOG("Filter-bank features processing complete");
            // Batch 1
            IntList indices;
            indices.Add(0);
            XTensor paddingEncForAudio;
            if (config->common.useFP16) {
                XTensor* p = &batchEnc;
                InitTensor(p, p->order, p->dimSize, X_FLOAT16, p->devID, p->enableGrad && X_ENABLE_GRAD);
                InitTensor1D(&paddingEncForAudio, int(batchEnc.GetDim(0) / 2), X_FLOAT16, config->common.devID, false);
            }
            else {
                InitTensor1D(&paddingEncForAudio, int(batchEnc.GetDim(0)/ 2), X_FLOAT, config->common.devID, false);
            }

            paddingEncForAudio = paddingEncForAudio * 0 + 1;
            
            DecodingBatch(batchEnc, paddingEncForAudio, indices, audio_length);
        }
        else    // offline
        {
            batchLoader = new S2TGeneratorDataset;
            batchLoader->Init(*config, false);

            XTensor paddingEnc;
            /* sentence information */
            XList info;
            XList inputs;
            int wordCount;
            IntList indices;
            inputs.Add(&batchEnc);
            inputs.Add(&paddingEnc);
            info.Add(&wordCount);
            info.Add(&indices);
            while (!batchLoader->IsEmpty()) {
                batchLoader->GetBatchSimple(&inputs, &info);
            XTensor paddingEncForAudio;
            if (batchEnc.order == 3)
            {
                if (config->common.useFP16)
                    InitTensor2D(&paddingEncForAudio, batchEnc.GetDim(0), int(batchEnc.GetDim(1) / 2), X_FLOAT16, config->common.devID);
                else
                    InitTensor2D(&paddingEncForAudio, batchEnc.GetDim(0), int(batchEnc.GetDim(1) / 2), X_FLOAT, config->common.devID);
            }
            else if (batchEnc.order == 2){
                if (config->common.useFP16)
                    InitTensor1D(&paddingEncForAudio, int(batchEnc.GetDim(0) / 2), X_FLOAT16, config->common.devID);
                else
                    InitTensor1D(&paddingEncForAudio, int(batchEnc.GetDim(0) / 2), X_FLOAT, config->common.devID);
            }
            else
                CheckNTErrors(false, "Invalid batchEnc size\n");
            paddingEncForAudio = paddingEncForAudio * 0 + 1;
            
            DecodingBatch(batchEnc, paddingEncForAudio, indices, batchEnc.GetDim(1));
            }
        }

        return true;
    }
}
