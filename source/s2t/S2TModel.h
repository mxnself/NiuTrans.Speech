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

#ifndef __MODEL_S2T__
#define __MODEL_S2T__

#include "S2TConfig.h"
#include "../nmt/Decoder.h"
#include "S2TEncoder.h"
#include "./submodel/CifPredictor.h"
#include "./submodel/Adapter.h"
#include "../nmt/submodel/Output.h"
#include "../niutensor/train/XModel.h"
using namespace nts;

  /* the s2t namespace */
namespace s2t
{

    /* an nmt model that keeps parameters of the encoder,
       the decoder and the output layer (softmax). */
    class S2TModel : public XModel
    {
    public:
        /* device id */
        int devID;

        ///* configurations */
        S2TConfig* config;

        /* the encoder */
        Adapter* adapter;

        /* the encoder */
        S2TAttEncoder* encoder;

        /* the decoder */
        AttDecoder* decoder;

        /* output layer */
        OutputLayer* outputLayer;

        /* predictor layer */
        CifPredictor* predictor;

        LinearLayer* decoderOutputLayer;

    public:
        /* constructor */
        S2TModel();

        /* de-constructor */
        ~S2TModel();

    //    /* get configurations */
        vector<int*> GetIntConfigs();
        vector<bool*> GetBoolConfigs();
        vector<float*> GetFloatConfigs();

        /* initialize the model */
        void InitModel(S2TConfig* config);

    //    /* make the mask of the encoder */
        void MakeS2TMaskEnc(XTensor& paddingEnc, XTensor& maskEnc);

        void MakeEncDecMaskEnc(XTensor& paddingEnc, XTensor& maskEnc, int num);

    //    /* make the lower triangle mask of the decoder for inference */
        XTensor MakeS2TTriMaskDecInference(int batchSize = 1, int length = 1);

        /* get parameter matrices */
        void GetParams(TensorList& list);


        /* read the parameters */
        void LoadFromFile(FILE* file);

        void TestDumpParams(XTensor* params);
    
    };

} /* end of the s2t namespace */

#endif /* __MODEL_S2T__ */
