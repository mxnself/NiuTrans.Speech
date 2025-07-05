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
#include "CifPredictor.h"
#include "../../niutensor/tensor/function/GELU.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/core/shape/Transpose.h"
#include "../../niutensor/tensor/core/getandset/Select.h"
#include <cmath>
#define CLOCKS_PER_SEC ((clock_t)1000)

namespace s2t{
    CifPredictor::CifPredictor()
    {
        isTraining = false;
        devID = -1;
        inSize = -1;
        hSize = -1;
        nConvNAR = -1;
        kernels= NULL;
        dropoutP = 0.0F;
        smooth_factor = 1.0F;
        noise_threshold = 0.0F;
    }
    CifPredictor::~CifPredictor()
    {
        if (kernels != NULL)
            delete[] kernels;
        if (biases != NULL)
            delete[] biases;
        delete prelinear;
        prelinear = NULL;
    }

    /* 
        set the training flag
        indicates whether train the model 
    */
    void CifPredictor::SetTrainingFlag(bool myIsTraining)
    {
        isTraining = myIsTraining;
    }

    /*
        Initialize the model
    */
    void CifPredictor::InitModel(S2TConfig& config)
    {
        SetTrainingFlag(config.training.isTraining);
        devID = config.common.devID;
        nConvNAR = config.predictor.nConvNAR;
        convKernelsNAR.assign(config.predictor.convKernelsNAR.begin(), config.predictor.convKernelsNAR.end());
        convStridesNAR.assign(config.predictor.convStridesNAR.begin(), config.predictor.convStridesNAR.end());

        threshold = config.predictor.threshold;
        tail_threshold = config.predictor.tail_threshold;
        inSize = config.model.encEmbDim;
        hSize = config.model.encEmbDim;
        kernels = new XTensor[nConvNAR];
        biases = new XTensor[nConvNAR];
        prelinear = new LinearLayer();

        for (int i = 0; i < nConvNAR; i++)
        {
            if (i == 0)
            {
                InitTensor3D(&kernels[i], hSize, inSize, convKernelsNAR[i], X_FLOAT, devID);
            }
            else
            {
                InitTensor3D(&kernels[i], hSize, hSize, convKernelsNAR[i], X_FLOAT, devID);
            }
            InitTensor1D(&biases[i], hSize, X_FLOAT, devID);
        }
  
        prelinear->InitModel(config, hSize, 1);
    }

    /* 
        tail process function
        >> input - encoder output
        >> alphas - acoustic embedding weight
        << token_num - predicted text token number
        >> mask - mask matrix
    */
    void CifPredictor::_tailProcess(XTensor& input, XTensor& alphas, XTensor& token_num, XTensor mask)
    {
        // read file
        INT32 b, t, d, num;
        XTensor zeros, ones, zeroT, tailT;
        XTensor Mask1, Mask2, mask3;
        b = input.GetDim(0);
        t = input.GetDim(1);
        d = input.GetDim(2);
        InitTensor2D(&zeros, b, 1, input.dataType, input.devID);
        zeros.SetZeroAll();
        ones = zeros + 1.0;
        Mask1 = Concatenate(mask,zeros,zeros.order-1);
        Mask2 = Concatenate(ones,mask,ones.order-1);
        mask3 = Mask2 - Mask1;
        tailT = mask3 * tail_threshold;
        alphas = Concatenate(alphas, zeros, 1);
        alphas = Sum(alphas, tailT);
        InitTensor3D(&zeroT, b, 1, d, input.dataType, input.devID);
        zeroT.SetZeroAll();
        input = Concatenate(input, zeroT, 1);
        token_num = ReduceSum(alphas, alphas.order-1, 1, false);
        num = token_num.Get1D(0);
        token_num = Clip(token_num, num, num);
    }

    /*
        continuous integrate-and-fire function to estimate the target number and generate the hidden variables
        >> input - encoder output 
        >> alphas - acoustic embedding weight
        << output - tensor with feature vector calculated from the input audio 
    */
    XTensor CifPredictor::cif(XTensor& input, XTensor& alphas){
        INT32 batchSize, lenT, hiddenSize;
        XTensor integrate, frame, alpha, distributionCompletion, alphaSq, hidden, encodingTMP;
        XTensor output, fires, frames, cur, remainds, remaind, curTmp, hiddenTmp, framesTmp, swap;
        int* index = new int[1];

        batchSize = input.GetDim(0);
        lenT = input.GetDim(1);
        hiddenSize = input.GetDim(2);

        InitTensor2D(&fires, batchSize, lenT, input.dataType, input.devID);
        InitTensor3D(&frames, batchSize, lenT, hiddenSize, input.dataType, input.devID);
        InitTensor1D(&integrate, batchSize, input.dataType, input.devID);
        InitTensor1D(&cur, batchSize, input.dataType, input.devID);
        InitTensor2D(&alpha, batchSize, 1, input.dataType, input.devID);
        InitTensor3D(&hidden, batchSize, 1, hiddenSize, input.dataType, input.devID);
        integrate.SetZeroAll();
        InitTensor2D(&frame, batchSize, hiddenSize, input.dataType, input.devID);
        InitTensor3D(&framesTmp, batchSize, 1, hiddenSize, input.dataType, input.devID);
        InitTensor3D(&output, batchSize, lenT, hiddenSize, input.dataType, input.devID);
        InitTensor1D(&distributionCompletion, batchSize, input.dataType, input.devID);
        frame.SetZeroAll();
        for(int i=0; i< lenT; i++){
            index[0] = i;
            _Select(&alphas, &alpha, index, 1);
            alphaSq = Squeeze(alpha, alpha.order-1);
            distributionCompletion.SetZeroAll();
            distributionCompletion = distributionCompletion + 1.0;
            distributionCompletion = Sub(distributionCompletion, integrate);
            integrate = Sum(integrate, alphaSq);
            XTensor lables(integrate);
            _SetDataIndexed(&fires, &integrate, 1, i);
            for(int gt=0; gt < lables.GetDim(0); gt++){
                if (lables.Get1D(gt) >= threshold)
                {
                    integrate.Set1D(integrate.Get1D(gt)-1.0, gt);
                    cur.Set1D(distributionCompletion.Get1D(gt), gt);
                }
                else
                    cur.Set1D(alphaSq.Get1D(gt), gt);
            }
            
            remainds = alphaSq - cur;
            curTmp = Unsqueeze(cur, 1, input.GetDim(-1));
            _Select(&input, &hidden, index, 1);
            hiddenTmp = Squeeze(hidden, 1);
            frame = frame + Multiply(curTmp, hiddenTmp);
            _SetDataIndexed(&frames, &frame, 1, i);
            for(int gt=0; gt < lables.GetDim(0); gt++){
                if (lables.Get1D(gt) >= threshold)
                {
                    remaind = Unsqueeze(remainds, 1, input.GetDim(-1));
                    frame = Multiply(remaind, hiddenTmp);
                }
            }
        }
        

        INT32 out_idx = 0;
        INT32 output_length = 0;
        for(int i=0; i<fires.GetDim(1); i++)
        {
            if(fires.Get2D(0,i) >= threshold){
                index[0] = i;
                _Select(&frames, &framesTmp, index, 1);
                swap = Squeeze(framesTmp, 0);
                output_length += 1;
                _SetDataIndexed(&output, &swap, 1, out_idx);
                out_idx = out_idx + 1;
            }
        }
        output = SelectRange(output, 1, 0, output_length);
        
        delete[] index;
        return output;
    }
    
    /*
        initialize the network
        Perform forward propagation and calculate the output results
        >> input - encoder output 
        >> mask - mask matrix
        << output - the result of forward propagation
    */
    XTensor CifPredictor::Make(XTensor input, XTensor& mask)
    {
        CheckNTErrors(tail_threshold>0.0, "tail_threshold must > 0");
        XTensor outFeature;
        XTensor context, queries, memory;
        XTensor alphas;
        XTensor token_num;
        XTensor maskEnc;
        XTensor output;
        XTensor encodingTMP;
        int order;

        maskEnc = mask;
        order = mask.order;
        context = Transpose(input, 1, 2);
        memory = Conv1DBias(context, kernels[0], biases[0], convStridesNAR[0], 1);
        outFeature = memory + context;
        
        outFeature = Transpose(outFeature, 1, 2);
        outFeature = Rectify(outFeature);
        outFeature = prelinear->Make(outFeature, false);
        alphas = Sigmoid(outFeature);
        alphas = Rectify(alphas*smooth_factor - noise_threshold);
        mask = Transpose(mask, order-1, order-2);
        alphas = alphas * mask;
        SqueezeMe(alphas, order-1);
        SqueezeMe(mask, order-1);
        token_num = ReduceSum(alphas, alphas.order-1, 1, false);
        _tailProcess(input, alphas, token_num, mask);
        output = cif(input, alphas);
        return output;

    }
}
