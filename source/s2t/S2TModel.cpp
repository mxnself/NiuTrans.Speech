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
#include "S2TModel.h"
#include "../niutensor/tensor/core/arithmetic/Conv1D.h"
#include <iostream>
namespace s2t
{
    S2TModel::S2TModel()
    {
        devID = -1;
        config = NULL;
        encoder = new S2TAttEncoder();
        decoder = new AttDecoder();
        adapter = new Adapter();
        outputLayer = new OutputLayer();
        predictor = new CifPredictor();
        decoderOutputLayer = new LinearLayer();
    }
    S2TModel::~S2TModel()
    {
        delete encoder;
        delete decoder;
        delete outputLayer;
        delete adapter;
        delete predictor;
        delete decoderOutputLayer;
    }

    /* return a list to keep the configurations (interger) */
    vector<int*> S2TModel::GetIntConfigs()
    {

        /* 19 integers */
        vector<int*> intConfig = {
            &(config->model.srcVocabSize),
            &(config->model.tgtVocabSize),
            &(config->model.sos),
            &(config->model.eos),
            &(config->model.pad),
            &(config->model.unk),
            &(config->model.maxSrcLen),
            &(config->model.maxTgtLen),
            &(config->model.maxRelativeLength),
            &(config->s2tmodel.fbank),
            &(config->model.encEmbDim),
            &(config->model.encLayerNum),
            &(config->model.encSelfAttHeadNum),
            &(config->model.encFFNHiddenDim),
            &(config->model.decEmbDim),
            &(config->model.decLayerNum),
            &(config->model.decSelfAttHeadNum),
            &(config->model.encDecAttHeadNum),
            &(config->model.decFFNHiddenDim),
            &(config->model.fnnActFunType),
            //&(config->predictor.idim),
        };
        /*
        NAR config
        */
        if (config->model.NAR){
            intConfig.push_back(&(config->predictor.idim));
            intConfig.push_back(&(config->model.adapterLayerNum));
        }

        return intConfig;
    }

    /* return a list to keep the configurations (bool) */
    vector<bool*> S2TModel::GetBoolConfigs()
    {
        /* 12 bool */
        vector<bool*> boolConfig = {
            &(config->model.encoderL1Norm),
            &(config->model.decoderL1Norm),
            &(config->model.useBigAtt),
            &(config->model.decoderOnly),
            &(config->model.encFinalNorm),
            &(config->model.decFinalNorm),
            &(config->model.encPreLN),
            &(config->model.decPreLN),
            &(config->model.useEncHistory),
            &(config->model.useDecHistory),
            &(config->model.shareEncDecEmb),
            &(config->model.shareDecInputOutputEmb),
        };

        return boolConfig;
    }

    /* return a list to keep the configurations (float) */
    vector<float*> S2TModel::GetFloatConfigs()
    {
        /* 3 float */
        vector<float*> floatConfig = {
            &(config->model.dropout),
            &(config->model.ffnDropout),
            &(config->model.attDropout),
            /*
            NAR config
            */
        };
        if (config->model.NAR){
            floatConfig.push_back(&(config->predictor.threshold));
            floatConfig.push_back( &(config->predictor.tail_threshold));
        }

        return floatConfig;
    }

    /*
    initialize the model
    >> myConfig - configuration of the model
    */
    void S2TModel::InitModel(S2TConfig* myConfig)
    {
        std::cout << "----- S2TModel Init -----" << std::endl;
        config = myConfig;
        devID = config->common.devID;

        /* configurations for the model */
        vector<int*> intConfig = GetIntConfigs();
        vector<bool*> boolConfig = GetBoolConfigs();
        vector<float*> floatConfig = GetFloatConfigs();

        FILE* modelFile = NULL;
        modelFile = fopen(config->common.modelFN, "rb");
        cout << "+ modelFile: " << config->common.modelFN << "\t" << (modelFile == NULL) << endl;


        /* read model configurations */
        if (modelFile) {
        
            // CheckNTErrors(modelFile, "Failed to open the model file");

            LOG("loading configurations from the model file...");

            /* 12 booleans */
            for (auto c : boolConfig) {
                fread(c, sizeof(bool), 1, modelFile);
            }
            int maxSrcLen = config->model.maxSrcLen;
            /* 19 intergers */
            for (auto c : intConfig) {
                fread(c, sizeof(int), 1, modelFile);
            }
            /* reset the maximum source sentence length */
            // config->model.maxSrcLen = MIN(maxSrcLen, config->model.maxSrcLen);
            /* 3 float */
            for (auto c : floatConfig) {
                fread(c, sizeof(float), 1, modelFile);
            }
        }

        config->showConfig();

        if (config->training.isTraining) {
            
            /* currently we do not support training */

        }

        if (config->training.isTraining)
        {
            ShowNTErrors("TODO!!");
        }
        adapter->InitModel(*config);
        encoder->InitModel(*config);
        decoder->InitModel(*config);
        outputLayer->InitModel(*config);
        if (config->model.NAR){
            predictor->InitModel(*config);
        }
        decoderOutputLayer->InitModel(*config, config->model.decEmbDim, config->model.tgtVocabSize);

        /* share encoder&decoder embeddings */
        if (config->model.shareEncDecEmb) {
            // decoder->embedder = &(encoder->embedder);
            LOG("share encoder decoder embeddings");
        }

        /* share embeddings with output weights */
        if (config->model.shareDecInputOutputEmb) {
            outputLayer->w = decoder->embedder->w;
            LOG("share decoder embeddings with output weights");
        }

        /* load parameters for translation or incremental training */
        if (config->training.incremental || (!config->training.isTraining))
            LoadFromFile(modelFile);
        #ifdef USE_CUDNN
            WarmUpCudnn();
        #endif
        if (modelFile)
            fclose(modelFile);

        std::cout << "--- S2TModel Init End ---" << std::endl;
    }

    void S2TModel::GetParams(TensorList& list)
    {
        list.Clear();
        if (!config->model.decoderOnly) {
            if (config->model.NAR) {
                 for (int i = 0; i < adapter->nlayer; i++) {
                    if (! config->model.fusionAttn) {
                        list.Add(&adapter->selfAtts[i].weightK);
                        list.Add(&adapter->selfAtts[i].biasK);
                        list.Add(&adapter->selfAtts[i].weightV);
                        list.Add(&adapter->selfAtts[i].biasV);
                        list.Add(&adapter->selfAtts[i].weightQ);
                        list.Add(&adapter->selfAtts[i].biasQ);
                    }
                    else {
                        list.Add(&adapter->selfAtts[i].weightFusion);
                        list.Add(&adapter->selfAtts[i].biasFusion);
                    }
                    list.Add(&adapter->selfAtts[i].weightO);
                    list.Add(&adapter->selfAtts[i].biasO);
                    list.Add(&adapter->attLayerNorms[i].weight);
                    list.Add(&adapter->attLayerNorms[i].bias);
                    list.Add(&adapter->ffns[i].w1);
                    list.Add(&adapter->ffns[i].b1);
                    list.Add(&adapter->ffns[i].w2);
                    list.Add(&adapter->ffns[i].b2);
                    list.Add(&adapter->fnnLayerNorms[i].weight);
                    list.Add(&adapter->fnnLayerNorms[i].bias);
                }
                if (adapter->finalNorm) {
                    list.Add(&adapter->encoderLayerNorm->weight);
                    list.Add(&adapter->encoderLayerNorm->bias);
                }
            }
        }

        if (!config->model.decoderOnly) {
            /* extractor parameters */
            for (int i = 0; i < encoder->extractor->nConv; i++) {
                list.Add(&encoder->extractor->kernels[i]);
                list.Add(&encoder->extractor->biases[i]);
            }
        }
        
        if (config->model.useBigAtt) {
            /* encoder parameters */
            if (!config->model.decoderOnly) {

                if (encoder->useHistory) {
                    for (int i = 0; i < encoder->nlayer + 1; i++)
                        list.Add(&encoder->history->weights[i]);
                    for (int i = 0; i < encoder->nlayer; i++) {
                        list.Add(&encoder->history->layerNorms[i].weight);
                        list.Add(&encoder->history->layerNorms[i].bias);
                    }
                }
                for (int i = 0; i < encoder->nlayer; i++) {
                    list.Add(&encoder->selfAtts[i].weightQ);
                    list.Add(&encoder->selfAtts[i].weightK);
                    list.Add(&encoder->selfAtts[i].weightV);
                    list.Add(&encoder->selfAtts[i].biasQ);
                    //list.Add(&encoder->selfAtts[i].biasK);
                    list.Add(&encoder->selfAtts[i].biasV);
                    if (encoder->selfAtts[i].useRPR)
                        list.Add(&encoder->selfAtts[i].RPEmbK);
                    list.Add(&encoder->selfAtts[i].weightO);
                    list.Add(&encoder->selfAtts[i].biasO);
                    list.Add(&encoder->ffns[i].w1);
                    list.Add(&encoder->ffns[i].b1);
                    list.Add(&encoder->ffns[i].w2);
                    list.Add(&encoder->ffns[i].b2);
                    list.Add(&encoder->attLayerNorms[i].weight);
                    list.Add(&encoder->attLayerNorms[i].bias);
                    list.Add(&encoder->fnnLayerNorms[i].weight);
                    list.Add(&encoder->fnnLayerNorms[i].bias);
                }
                if (encoder->finalNorm) {
                    list.Add(&encoder->encoderLayerNorm->weight);
                    list.Add(&encoder->encoderLayerNorm->bias);
                }
            }

            /* decoder parameters */
            if (decoder->useHistory) {
                for (int i = 0; i < decoder->nlayer + 1; i++)
                    list.Add(&decoder->history->weights[i]);
                for (int i = 0; i < decoder->nlayer; i++) {
                    list.Add(&decoder->history->layerNorms[i].weight);
                    list.Add(&decoder->history->layerNorms[i].bias);
                }
            }

            for (int i = 0; i < decoder->nlayer; i++) {
                list.Add(&decoder->selfAtts[i].weightQ);
                list.Add(&decoder->selfAtts[i].weightK);
                list.Add(&decoder->selfAtts[i].weightV);
                list.Add(&decoder->selfAtts[i].biasQ);
                //list.Add(&decoder->selfAtts[i].biasK);
                list.Add(&decoder->selfAtts[i].biasV);
                if (decoder->selfAtts[i].useRPR)
                    list.Add(&decoder->selfAtts[i].RPEmbK);
                list.Add(&decoder->selfAtts[i].weightO);
                list.Add(&decoder->selfAtts[i].biasO);
                list.Add(&decoder->selfAttLayerNorms[i].weight);
                list.Add(&decoder->selfAttLayerNorms[i].bias);
                if (!config->model.decoderOnly) {
                    list.Add(&decoder->enDeAtts[i].weightQ);
                    list.Add(&decoder->enDeAtts[i].weightK);
                    list.Add(&decoder->enDeAtts[i].weightV);
                    list.Add(&decoder->enDeAtts[i].biasQ);
                    //list.Add(&decoder->enDeAtts[i].biasK);
                    list.Add(&decoder->enDeAtts[i].biasV);
                    list.Add(&decoder->enDeAtts[i].weightO);
                    list.Add(&decoder->enDeAtts[i].biasO);
                    list.Add(&decoder->enDeAttLayerNorms[i].weight);
                    list.Add(&decoder->enDeAttLayerNorms[i].bias);
                }
                if (decoder->ffns != NULL) {
                    list.Add(&decoder->ffns[i].w1);
                    list.Add(&decoder->ffns[i].b1);
                    list.Add(&decoder->ffns[i].w2);
                    list.Add(&decoder->ffns[i].b2);
                }
                list.Add(&decoder->ffnLayerNorms[i].weight);
                list.Add(&decoder->ffnLayerNorms[i].bias);
            }
        }
        else {
            /* adapter parameters */

            /* encoder parameters */
            if (!config->model.decoderOnly) {

                if (encoder->useHistory) {
                    for (int i = 0; i < encoder->nlayer + 1; i++)
                        list.Add(&encoder->history->weights[i]);
                    for (int i = 0; i < encoder->nlayer; i++) {
                        list.Add(&encoder->history->layerNorms[i].weight);
                        list.Add(&encoder->history->layerNorms[i].bias);
                    }
                }
                for (int i = 0; i < encoder->nlayer; i++) {
                    if (encoder->selfAtts[i].useRPR)
                        list.Add(&encoder->selfAtts[i].RPEmbK);
                    if (! config->model.fusionAttn) {
                        list.Add(&encoder->selfAtts[i].weightK);
                        //list.Add(&encoder->selfAtts[i].biasK);
                        list.Add(&encoder->selfAtts[i].weightV);
                        list.Add(&encoder->selfAtts[i].biasV);
                        list.Add(&encoder->selfAtts[i].weightQ);
                        list.Add(&encoder->selfAtts[i].biasQ);
                    }
                    else {
                        list.Add(&encoder->selfAtts[i].weightFusion);
                        list.Add(&encoder->selfAtts[i].biasFusion);
                    }
                    list.Add(&encoder->selfAtts[i].weightO);
                    list.Add(&encoder->selfAtts[i].biasO);
                    list.Add(&encoder->attLayerNorms[i].weight);
                    list.Add(&encoder->attLayerNorms[i].bias);
                    list.Add(&encoder->ffns[i].w1);
                    list.Add(&encoder->ffns[i].b1);
                    list.Add(&encoder->ffns[i].w2);
                    list.Add(&encoder->ffns[i].b2);
                    list.Add(&encoder->fnnLayerNorms[i].weight);
                    list.Add(&encoder->fnnLayerNorms[i].bias);
                }
                if (encoder->finalNorm) {
                    list.Add(&encoder->encoderLayerNorm->weight);
                    list.Add(&encoder->encoderLayerNorm->bias);
                }
            }

            /* decoder parameters */
            if (decoder->useHistory) {
                for (int i = 0; i < decoder->nlayer + 1; i++)
                    list.Add(&decoder->history->weights[i]);
                for (int i = 0; i < decoder->nlayer; i++) {
                    list.Add(&decoder->history->layerNorms[i].weight);
                    list.Add(&decoder->history->layerNorms[i].bias);
                }
            }

            for (int i = 0; i < decoder->nlayer; i++) {
                if (decoder->selfAtts[i].useRPR)
                    list.Add(&decoder->selfAtts[i].RPEmbK);
                if (! config->model.fusionAttn) {
                    list.Add(&decoder->selfAtts[i].weightK);
                    list.Add(&decoder->selfAtts[i].biasK);
                    list.Add(&decoder->selfAtts[i].weightV);
                    list.Add(&decoder->selfAtts[i].biasV);
                    list.Add(&decoder->selfAtts[i].weightQ);
                    list.Add(&decoder->selfAtts[i].biasQ);
                }
                else {
                    list.Add(&decoder->selfAtts[i].weightFusion);
                    list.Add(&decoder->selfAtts[i].biasFusion);
                }
                list.Add(&decoder->selfAtts[i].weightO);
                list.Add(&decoder->selfAtts[i].biasO);
                list.Add(&decoder->selfAttLayerNorms[i].weight);
                list.Add(&decoder->selfAttLayerNorms[i].bias);
                if (!config->model.decoderOnly) {
                    if (! config->model.fusionAttn) {
                        list.Add(&decoder->enDeAtts[i].weightK);
                        //list.Add(&decoder->enDeAtts[i].biasK);
                        list.Add(&decoder->enDeAtts[i].weightV);
                        list.Add(&decoder->enDeAtts[i].biasV);
                        list.Add(&decoder->enDeAtts[i].weightQ);
                        list.Add(&decoder->enDeAtts[i].biasQ);
                    }
                    else {
                        list.Add(&decoder->enDeAtts[i].weightQ);
                        list.Add(&decoder->enDeAtts[i].biasQ);
                        list.Add(&decoder->enDeAtts[i].weightFusion);
                        list.Add(&decoder->enDeAtts[i].biasFusion);
                    }
                    list.Add(&decoder->enDeAtts[i].weightO);
                    list.Add(&decoder->enDeAtts[i].biasO);
                    list.Add(&decoder->enDeAttLayerNorms[i].weight);
                    list.Add(&decoder->enDeAttLayerNorms[i].bias);
                }
                if (decoder->ffns != NULL) {
                    list.Add(&decoder->ffns[i].w1);
                    list.Add(&decoder->ffns[i].b1);
                    list.Add(&decoder->ffns[i].w2);
                    list.Add(&decoder->ffns[i].b2);
                }
                list.Add(&decoder->ffnLayerNorms[i].weight);
                list.Add(&decoder->ffnLayerNorms[i].bias);
            }
        }

        if (decoder->finalNorm) {
            list.Add(&decoder->decoderLayerNorm->weight);
            list.Add(&decoder->decoderLayerNorm->bias);
        }

        if (!config->model.decoderOnly) {
            // list.Add(encoder->embedder.w);
        }

        if (!config->model.shareEncDecEmb) {
            list.Add(decoder->embedder->w);
        }

        if (!config->model.shareDecInputOutputEmb) {
             list.Add(outputLayer->w);
        }
        /*
        RAW whisper
        */
        if (! config->model.NAR){
            list.Add(&decoder->embedder->posEmbeddingBase);

        }
        else{
            /* 
            NAR generation
            */
            list.Add(&decoderOutputLayer->w1);
            list.Add(&decoderOutputLayer->b1);
            
            for (int i = 0; i < predictor->nConvNAR; i++) {
                list.Add(&predictor->kernels[0]);
                list.Add(&predictor->biases[0]);
            }

            list.Add(&predictor->prelinear->w1);
            list.Add(&predictor->prelinear->b1);
        }


    }

    void S2TModel::LoadFromFile(FILE* file)
    {
        double startT = GetClockSec();

        LOG("loading parameters from the model file...");

        TensorList params;
        GetParams(params);

        cout << "+ Num of S2T Model Params: " << params.Size() << endl;

        int size = 0;
        for (int i = 0; i < params.Size(); i++) {
            size += params[i]->unitNum;
        }

        if (config->common.useFP16) {
            LOG("running with fp16");
        }
        else {
            LOG("running with fp32");
        }

        /* convert parameters to FP16 before reading files */
        if (config->common.useFP16) {
            for (int i = 0; i < params.Size(); i++) {
                XTensor* p = params[i];
                InitTensor(p, p->order, p->dimSize, X_FLOAT16, p->devID, p->enableGrad && X_ENABLE_GRAD);
            }

            // XTensor& encEmb = encoder->embedder.posEmbeddingBase;
            XTensor& encEmb = encoder->posEmbeddingBase;
            encEmb = ConvertDataType(encEmb, X_FLOAT16);
            if (!config->model.shareEncDecEmb) {
                XTensor& decEmb = decoder->embedder->posEmbeddingBase;
                decEmb = ConvertDataType(decEmb, X_FLOAT16);
            }
        }

        for (int i = 0; i < params.Size(); i++) {
            params[i]->BinaryRead(file);
            // LOG("--- 1 ---");
            // params[i]->Dump(sterr, NULL,10);
        }
            
        double elapsed = GetClockSec() - startT;
        LOG("model loaded (took %.1fs)", elapsed);

    }

    void S2TModel::TestDumpParams(XTensor* params)
    {
        params->Dump(stderr, NULL, 5);
    }

    /*
    make the mask of the encoder
    >> paddingEnc - padding of the encoder input, (batchSize, srcLen)
    >> maskEnc - mask of the encoder self-attention, (headNum, batchSize, srcLen, srcLen)
    */
    void S2TModel::MakeS2TMaskEnc(XTensor& paddingEnc, XTensor& maskEnc)
    {
        XTensor padding2;
        /* mask of the padding */
        Unsqueeze(paddingEnc, padding2, paddingEnc.order - 1, paddingEnc.GetDim(-1));
        Unsqueeze(padding2, maskEnc, 0, config->model.encSelfAttHeadNum);
        ScaleAndShiftMe(maskEnc, 1e9F, -1e9F);
    }

    void S2TModel::MakeEncDecMaskEnc(XTensor& paddingEnc, XTensor& maskEnc, int num)
    {
        XTensor padding2;
        /* mask of the padding */
        Unsqueeze(paddingEnc, padding2, paddingEnc.order-1, num);
        Unsqueeze(padding2, maskEnc, 0, config->model.encDecAttHeadNum);
        ScaleAndShiftMe(maskEnc, 1e9F, -1e9F);
    }

    XTensor S2TModel::MakeS2TTriMaskDecInference(int batchSize, int length)
    {
        /* encoder-decoder mask that prevents the attention to paded words */
        XTensor maskEncDec;

        InitTensor2D(&maskEncDec, length, length, X_FLOAT, devID);
        _SetDataLowTri(&maskEncDec, 1.0, 0);
        maskEncDec = Unsqueeze(maskEncDec, 0, batchSize);

        if (config->model.encDecAttHeadNum > 1) {
            maskEncDec = Unsqueeze(maskEncDec, 0, config->model.encDecAttHeadNum);
            ScaleAndShiftMe(maskEncDec, 1e9F, -1e9F);
            return maskEncDec;
        }
        else {
            ScaleAndShiftMe(maskEncDec, 1e9F, -1e9F);
            return maskEncDec;
        }
    }


} /* end of the s2t namespace */
