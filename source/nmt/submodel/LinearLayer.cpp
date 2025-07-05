/* NiuTrans.NMT - an open-source neural machine translation system.
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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "LinearLayer.h"
#include "Embedding.h"
#include "../Config.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/function/FHeader.h"

/* the nmt namespace */
namespace nmt
{

/* set the training flag */
void LinearLayer::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
LinearLayer::LinearLayer()
{
    inSize = -1;
    outSize = -1;
    devID = -1;
    isTraining = false;
}

/* de-constructor */
LinearLayer::~LinearLayer()
{
}

/*
initialize the model
>> config - configurations of the model
>> isEnc - indicates wether it is a encoder module
*/
void LinearLayer::InitModel(NMTConfig& config, bool isEnc)
{
    SetTrainingFlag(config.training.isTraining);
    devID = config.common.devID;
    inSize = isEnc ? config.model.encEmbDim : config.model.decEmbDim;
    outSize = isEnc ? config.model.encEmbDim : config.model.decEmbDim;

    InitTensor2D(&w1, outSize, inSize, X_FLOAT, devID);
    InitTensor1D(&b1, outSize, X_FLOAT, devID);


    if (isTraining) {
        _SetDataFanInOut(&w1);

        b1.SetZeroAll();
    }
}

void LinearLayer::InitModel(NMTConfig& config, int indim, int outdim)
{
    SetTrainingFlag(config.training.isTraining);
    devID = config.common.devID;
    inSize = indim;
    outSize = outdim;

    InitTensor2D(&w1, inSize, outSize, X_FLOAT, devID);
    InitTensor1D(&b1, outSize, X_FLOAT, devID);


    if (isTraining) {
        _SetDataFanInOut(&w1);
        b1.SetZeroAll();
    }
}

/*
make the network
y = max(0, x * w1 + b1) * w2 + b2
>> input - the input tensor
>> return - the output tensor
*/
XTensor LinearLayer::Make(XTensor& input, bool normalized)
{
    /* t1 = max(0, x * w1 + b1) */
    XTensor output;
    output = MulAndShift(input, w1, b1);
    // output.Dump(stderr, "--- output ----", 5);
    if (w1.enableGrad)
        return Softmax(output, -1);
    /* normalize the output for beam search */
    if (normalized) {
        TENSOR_DATA_TYPE dataType = output.dataType;
        if (dataType == X_FLOAT16)
            output = ConvertDataType(output, X_FLOAT);

        output = LogSoftmax(output, -1);

        if (output.dataType != dataType)
            output = ConvertDataType(output, dataType);
    }
    return output;
}


XTensor LinearLayer::Make(XTensor& input)
{
    /* t1 = x * w1 + b1)*/
    XTensor output;
    output = MulAndShift(input, w1, b1);
    return output;
}

} /* end of the nmt namespace */
