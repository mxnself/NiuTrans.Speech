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
  * $Created by: yuhao zhang(yoohao.zhang@gmail.com) 2023-09-22
  */

#include "Adapter.h"
#include <iostream>
using namespace nmt;
namespace s2t
{
    Adapter::Adapter()
    {
        devID = -1;
        selfAtts = NULL;
        ffns = NULL;
        attLayerNorms = NULL;
        fnnLayerNorms = NULL;
        encoderLayerNorm = NULL;
        useHistory = false;
        history = NULL;
        dropoutP = 0.0;
        embDim = -1;
        finalNorm = false;
        ignored = -1;
        nlayer = -1;
        preLN = false;
        vSize = -1;
        isTraining = false;
    }

    
    Adapter::~Adapter()
    {
    }

    void Adapter::InitModel(S2TConfig& config)
    {        

        SetTrainingFlag(config.training.isTraining);
        devID = config.common.devID;
        preLN = config.model.encPreLN;
        dropoutP = config.model.dropout;
        embDim = config.model.encEmbDim;
        nlayer = config.model.adapterLayerNum;
        vSize = config.model.srcVocabSize;
        finalNorm = config.model.encFinalNorm;
        useHistory = config.model.useEncHistory;

        
        //CheckNTErrors(vSize > 1, "Set vocabulary size by \"-vsize\"");
        if (nlayer > 1){
            CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");

            ffns = new FFN[nlayer];
            selfAtts = new Attention[nlayer];
            attLayerNorms = new LayerNorm[nlayer];
            fnnLayerNorms = new LayerNorm[nlayer];

            
            if (useHistory) {
                history = new LayerHistory;
                history->InitModel(config, true);
            }
            

            if (finalNorm) {
                encoderLayerNorm = new LayerNorm;
                encoderLayerNorm->InitModel(config, devID, embDim, config.model.encoderL1Norm);
            }

            /* initialize the stacked layers */

            for (int i = 0; i < nlayer; i++) {
                ffns[i].InitModel(config, true);
                selfAtts[i].InitModel(config, true, true, config.model.fusionAttn);
                attLayerNorms[i].InitModel(config, devID, embDim, config.model.encoderL1Norm);
                fnnLayerNorms[i].InitModel(config, devID, embDim, config.model.encoderL1Norm);
            }
        }
    }

    XTensor Adapter::applyLfr(XTensor& input)
    {
        INT32 batchSize, lenT, hiddenSize;
        int* index = new int[1];

        batchSize = input.GetDim(0);
        lenT = input.GetDim(1);
        hiddenSize = input.GetDim(2);
        
        int lfrM=4, lfrN=3;
        int startIndex;
        int lfrNum = (lenT + lfrN - 1) / lfrN;
        int rightPad = (lfrNum - 1) * lfrN + lfrM - lenT;
        XTensor lfrEmb, meanEnb, rightEmb, sumEmb, tmpEmb;
        InitTensor3D(&tmpEmb, batchSize, 1, hiddenSize, input.dataType, input.devID);
        InitTensor3D(&lfrEmb, batchSize, lfrNum, hiddenSize, input.dataType, input.devID);
        InitTensor3D(&meanEnb, batchSize, 4, hiddenSize, input.dataType, input.devID);
        InitTensor3D(&rightEmb, batchSize, 1, hiddenSize, input.dataType, input.devID);
        InitTensor3D(&sumEmb, batchSize, 1, hiddenSize, input.dataType, input.devID);
        index[0] = lenT-1;
        _Select(&input, &rightEmb, index, 1);
        rightEmb=Squeeze(rightEmb, 1);
        

        for (int i = 0; i < lfrNum; i++) {
            startIndex = i * lfrN;
            meanEnb.SetZeroAll();
            if (lfrM > (lenT-startIndex))
            {
                for (int j=0; j<lfrM-rightPad; j++){
                    index[0] = startIndex+j;
                    _Select(&input, &sumEmb, index, 1);
                    tmpEmb = Squeeze(sumEmb, 1);
                    _SetDataIndexed(&meanEnb, &tmpEmb, 1, j);
                }
                for (int j=0; j<rightPad; j++){
                    _SetDataIndexed(&meanEnb, &rightEmb, 1, j+lfrM-rightPad);
                }
                tmpEmb=ReduceMean(meanEnb, 1);
                _SetDataIndexed(&lfrEmb, &tmpEmb, 1, i);
            }
            else{
                for (int j=0; j< 4; j++){
                    index[0] = startIndex+j;
                    _Select(&input, &sumEmb, index, 1);
                    tmpEmb = Squeeze(sumEmb, 1);
                    _SetDataIndexed(&meanEnb, &tmpEmb, 1, j);
                }
                tmpEmb=ReduceMean(meanEnb, 1);
                _SetDataIndexed(&lfrEmb, &tmpEmb, 1, i);
            }
        }
        delete[] index;
        return lfrEmb;
    }

    XTensor Adapter::RunFastPreNorm(XTensor& input, XTensor* mask)
    {
        XTensor x = input;  

        for (int i = 0; i < nlayer; i++) {

            XTensor xn;
            
            /* layer normalization with pre-norm for self-attn */
            xn = attLayerNorms[i].Run(x);
            /* self attention */
            // LOG("--- selfAtts[i] ---");

            xn = selfAtts[i].Make(xn, xn, xn, mask, NULL);
            
            /* residual connection */
            SumMe(xn, x);
            

            /* layer normalization with pre-norm for ffn */
            x = fnnLayerNorms[i].Run(xn);

            /* ffn */
            x = ffns[i].Make(x);

            /* residual connection */
            SumMe(x, xn);

        }

        if (finalNorm) {
            return encoderLayerNorm->Run(x); 
        }

        return x;
    }
}
