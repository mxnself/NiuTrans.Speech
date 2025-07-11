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
  * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
  */

#include "Attention.h"
#include "Embedding.h"
#include "Softmax.cuh"
#include <iostream>

/* the nmt namespace */
namespace nmt
{

/* set the training flag */
void Attention::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
Attention::Attention()
{
    devID = -1;
    nhead = -1;
    kDim = -1;
    vDim = -1;
    embDim = -1;
    dropoutP = 0.0;
    maxRP = -1;
    useRPR = false;
    isTraining = false;
}

/* de-constructor */
Attention::~Attention()
{
}

/*
initialize the model
>> config - the configurations of the network
>> isEnc - indicates whether it is a encoder module
>> isSeleAtt - indicates whether it is a self-attention module
*/
void Attention::InitModel(NMTConfig& config, bool isEnc, bool isSelfAtt, bool isFusion)
{
    SetTrainingFlag(config.training.isTraining);
    devID = config.common.devID;
    useRPR = (config.model.maxRelativeLength > 0);

    if (isEnc) {
        nhead = config.model.encSelfAttHeadNum;
        embDim = config.model.encEmbDim;
        kDim = config.model.encEmbDim;
        vDim = config.model.encEmbDim;
        attType =ATTN_TYPE::SELF_ATT;
    }
    else {
        nhead = isSelfAtt ? config.model.decSelfAttHeadNum : config.model.encDecAttHeadNum;
        embDim = config.model.decEmbDim;
        kDim = config.model.decEmbDim;
        vDim = config.model.decEmbDim;
        attType = isSelfAtt ? ATTN_TYPE::SELF_ATT : ATTN_TYPE::EN_DE_ATT;
    }

    dropoutP = config.model.attDropout;
    maxRP = config.model.maxRelativeLength;

    /* initialize the parameters */
    if (! isFusion) {
        InitTensor2D(&weightQ, embDim, embDim, X_FLOAT, devID);
        InitTensor1D(&biasQ, embDim, X_FLOAT, devID);
        InitTensor2D(&weightK, kDim, embDim, X_FLOAT, devID);
        InitTensor1D(&biasK, embDim, X_FLOAT, devID);
        InitTensor2D(&weightV, vDim, embDim, X_FLOAT, devID);
        InitTensor1D(&biasV, embDim, X_FLOAT, devID);
    }
    else {
        // LOG("Fusion Attention");
        if (isSelfAtt) {
        InitTensor2D(&weightFusion, embDim, embDim * 3, X_FLOAT, devID);
        InitTensor1D(&biasFusion, embDim * 3, X_FLOAT, devID);
        }
        else {
            InitTensor2D(&weightQ, embDim, embDim, X_FLOAT, devID);
            InitTensor1D(&biasQ, embDim, X_FLOAT, devID);
            InitTensor2D(&weightFusion, embDim, embDim * 2, X_FLOAT, devID);
            InitTensor1D(&biasFusion, embDim * 2, X_FLOAT, devID);
        }
    }
    
    InitTensor2D(&weightO, embDim, embDim, X_FLOAT, devID);
    InitTensor1D(&biasO, embDim, X_FLOAT, devID);
    
    /* currently, we only support k-only mode, i.e., we do not set RPR for values */
    if (useRPR)
        InitTensor2D(&RPEmbK, maxRP * 2 + 1, embDim / nhead, X_FLOAT, devID);

    if (isTraining) {
        const float scale = 1.0F / sqrtf(2.0F);
        _SetDataFanInOut(&weightK, scale);
        _SetDataFanInOut(&weightQ, scale);
        _SetDataFanInOut(&weightV, scale);
        _SetDataFanInOut(&weightO, 1.0F);

        if (useRPR)
            _SetDataFanInOut(&RPEmbK, scale);

        biasQ.SetZeroAll();
        biasO.SetZeroAll();

        _SetDataXavierNormal(&biasK);
        _SetDataXavierNormal(&biasV);
    }
}

void fused_proj(const XTensor& activation,
                const XTensor& weight,
                const XTensor& bias,
                std::vector<XTensor*> results) {
  XTensor fused_result = MulAndShift(activation, weight, bias);
  int dimSize[MAX_TENSOR_DIM_NUM]{};
  std::copy(std::begin(fused_result.dimSize), std::end(fused_result.dimSize), std::begin(dimSize));
  dimSize[fused_result.order-1] /= results.size();

  for(auto r : results) {
    *r = XTensor{fused_result.order, dimSize, fused_result.dataType, fused_result.denseRatio,
                 fused_result.devID, fused_result.mem};
  }

  TensorList list;
  for(auto r : results) {
    list.Add(r);
  }
  Split(fused_result, list, fused_result.order - 1, results.size());
}

/*
make the network
>> k - keys, B * L * H or N * B * L * H (when N > 1 and not using rpr attn).
       where N = num head, B = batch size, L = sequence length,
       and H = vector size of each position
>> q - queries, B * L * H or N * B * L * H for encoders, B * 1 * H or N * B * 1 * H for decoders during inference
>> v - values, B * L * H or N * B * L * H
>> mask - as it is
>> isTraining - indicates whether the model is used for training
>> cache - decoder cache
>> cacheType - type of cache, e.g., self-attention
<< return - attention result
*/
XTensor Attention::Make(XTensor& k, XTensor& q, XTensor& v, 
                        XTensor* mask, Cache* cache)
{
    const bool isEnc = (!cache) ? true : false;
    // TODO: support new kv cache layout in rpr attn.
    const bool split_in_kv_cache = nhead > 1 && !useRPR;

    /* linear transformation before self-attention */
    XTensor q2, k2, v2;

    if(attType == ATTN_TYPE::SELF_ATT) {
        fused_proj(q, weightFusion, biasFusion, {&q2,&k2,&v2});
    } else {
        q2 = MulAndShift(q, weightQ, biasQ);
    }
    if (!cache || isTraining || !(cache->enable)) {
        /* self attention for encoder layers */
        if (split_in_kv_cache) {
            q2 = Split(q2, q2.order - 1, nhead);
            k2 = Split(k2, k2.order - 1, nhead);
            v2 = Split(v2, v2.order - 1, nhead);
        }

        if (useRPR && attType == ATTN_TYPE::SELF_ATT)
            return MakeRPRAttention(k2, q2, v2, mask, isEnc);
        return MakeAttention(k2, q2, v2, mask, isEnc);
    }

    else {
        if (split_in_kv_cache) {
            q2 = Split(q2, q2.order - 1, nhead);
        }
        if (attType == ATTN_TYPE::SELF_ATT) {
            const int concat_dim = split_in_kv_cache ? 2 : 1;
            if (split_in_kv_cache) {
                k2 = Split(k2, k2.order - 1, nhead);
                v2 = Split(v2, v2.order - 1, nhead);
            }

            /* if hit, we only concat the cache with the new token */
            if (!cache->miss) {
                k2 = Concatenate(cache->key, k2, concat_dim);
                v2 = Concatenate(cache->value, v2, concat_dim);
            }
            cache->key = std::move(k2);
            cache->value = std::move(v2);
            cache->miss = false;
            // q2.Dump(stderr, "--- Q ---", 5);
            // k2.Dump(stderr, "--- K ---", 5);
            // v2.Dump(stderr, "--- V ---", 5);
            if (useRPR)
                return MakeRPRAttention(cache->key, q2, cache->value, mask, isEnc);
            return MakeAttention(cache->key, q2, cache->value, mask, isEnc);
        }
        else if (attType == ATTN_TYPE::EN_DE_ATT) {
            if (cache->miss) {
                fused_proj(k, weightFusion, biasFusion, {&cache->key,&cache->value});
                cache->miss = false;

                if (split_in_kv_cache) {
                    cache->key = Split(cache->key, cache->key.order - 1, nhead);
                    cache->value = Split(cache->value, cache->value.order - 1, nhead);
                }
            }

            return MakeAttention(cache->key, q2, cache->value, mask, isEnc);
        }
        CheckNTErrors(0, "invalid cache type");
    }
}

/*
make the attention network given keys, queries and values (after linear transformation)
>> k - keys, B * L * H
>> q - queries, B * L * H
>> v - values, B * L * H
>> mask - as it is
>> isEnc - indicates whether it is a encoder module
*/
XTensor Attention::MakeAttention(XTensor& k, XTensor& q, XTensor& v, 
                                 XTensor* mask, bool isEnc)
{
    const auto dataType = k.dataType;

    XTensor att;

    if (isTraining)
        q = Scale(q, 1.0F / (float)sqrt((float)kDim / nhead));
    else
        ScaleMe(q, 1.0F / (float)sqrt((float)kDim / nhead));

    /* scalar = softmax(Q * K^T / sqrt(dk)) * V */
    const int seqlen_q = q.dimSize[q.order-2];
    const int seqlen_kv = k.dimSize[k.order-2];
    int q_original_dim_size[MAX_TENSOR_DIM_NUM];
    int kv_original_dim_size[MAX_TENSOR_DIM_NUM];
    const int q_original_order = q.order;
    const int kv_original_order = k.order;
    if(attType == ATTN_TYPE::EN_DE_ATT) {
      std::copy(std::begin(q.dimSize), std::end(q.dimSize), std::begin(q_original_dim_size));
      std::copy(std::begin(k.dimSize), std::end(k.dimSize), std::begin(kv_original_dim_size));
      q.ReshapeMerged(q.order-3);
      k.ReshapeMerged(k.order-3);
      v.ReshapeMerged(v.order-3);
    }
    att = BMMul(q, X_NOTRANS, k, X_TRANS);

    if (att.dataType == X_FLOAT16) {
        att = ConvertDataType(att, X_FLOAT);
    }
    if (mask) {
        if (isTraining)
            att = Sum(att, *mask, /*inplace=*/true);
        else
            SumMe(att, *mask);
    }
    #ifdef USE_CUDNN
        att = nmt::softmax(att);
    #else 
        att=Softmax(att, -1);
    #endif

    if (isTraining && dropoutP > 0)
        att = Dropout(att, dropoutP);

    if (dataType != att.dataType)
        att = ConvertDataType(att, dataType);
    
    att = BMMul(att, v);

    if(attType == ATTN_TYPE::EN_DE_ATT) {
      q.Reshape(q_original_order, q_original_dim_size);
      att.Reshape(q_original_order, q_original_dim_size);
      k.Reshape(kv_original_order, kv_original_dim_size);
      v.Reshape(kv_original_order, kv_original_dim_size);
    }
    /* concatenate the heads */
    if (nhead > 1)
        return MulAndShift(Merge(att, att.order - 1), weightO, biasO);
    else
        return MulAndShift(att, weightO, biasO);
}
    
/*
make the attention network by incorporating the relative position representation
with the given keys, queries and values (after linear transformation)
>> k - keys, B * L * H
>> q - queries, B * L * H
>> v - values, B * L * H
>> mask - as it is
>> isEnc - indicates whether it is encoder
*/
XTensor Attention::MakeRPRAttention(XTensor& k, XTensor& q, XTensor& v,
                                    XTensor* mask, bool isEnc)
{
    XTensor kheads;
    XTensor qheads;
    XTensor vheads;

    const int batchSize = q.GetDim(0);
    const int lenQ = q.GetDim(1);
    const int lenKV = k.GetDim(1);

    const auto dataType = k.dataType;

    /* multi head */
    kheads = Split(k, k.order - 1, nhead);
    qheads = Split(q, q.order - 1, nhead);
    vheads = Split(v, v.order - 1, nhead);

    XTensor att;
    XTensor dot;
    XTensor scalar;

    XTensor embMatrix, relativeKey;

    /* generate the relative emb index (L_q, L_kv) */
    embMatrix = GetRPEmbedding(lenQ, lenKV, isEnc || isTraining);

    /* generate the relative key from the RPEmbK (L_q, L_kv, H/K) */
    relativeKey = Gather(RPEmbK, embMatrix);

    if (qheads.dataType == X_FLOAT16) {
        qheads = ConvertDataType(qheads, X_FLOAT);
        kheads = ConvertDataType(kheads, X_FLOAT);
        relativeKey = ConvertDataType(relativeKey, X_FLOAT);
    }

    float scaling = (float)sqrt(embDim / nhead);
    qheads = ScaleAndShift(qheads, 1.0F / scaling);

    dot = RPDotProduct(qheads, kheads, relativeKey, true);

    if (mask)
        dot = Sum(dot, *mask, /*inplace=*/true);

    /* softmax */
    // scalar = Softmax(dot, -1);
    #ifdef USE_CUDNN
        scalar = nmt::softmax(att);
    #else 
        scalar=Softmax(att, -1);
    #endif

    if (isTraining && dropoutP > 0)
        scalar = Dropout(scalar, dropoutP);

    if (vheads.dataType != scalar.dataType)
        vheads = ConvertDataType(vheads, scalar.dataType);

    /* generate the relative attention output (K, B, L_q, H/K) */
    att = BMMul(scalar, vheads);

    if (dataType != att.dataType)
        att = ConvertDataType(att, dataType);

    /* concatenate the heads */
    return MulAndShift(Merge(att, att.order - 1), weightO, biasO);
}

/*
generate relative position embeddings
>> lenQ - the length of query
>> lenKV - the length of key and value
>> isEnc - indicates whether it is in the encoder
*/
XTensor Attention::GetRPEmbedding(int lenQ, int lenKV, bool isEnc)
{
    XTensor range;
    XTensor embMatrix;

    InitTensor1D(&range, lenKV, X_INT, devID);
    int* index = new int[lenKV];

    if (isEnc) {
        for (int i = 0; i < lenKV; i++)
            index[i] = i;
        range.SetData(index, lenKV);
        XTensor range2D;
        XTensor range2DTrans;
        range2D = Unsqueeze(range, 0, lenQ);
        range2DTrans = Transpose(range2D, 0, 1);

        embMatrix = Sum(range2D, range2DTrans, false, -1);
    }
    else {
        for (int i = 0; i < lenKV; i++)
            index[i] = -lenKV + i + 1;
        range.SetData(index, lenKV);
        embMatrix = Unsqueeze(range, 0, lenQ);
    }

    ClipMe(embMatrix, -float(maxRP), float(maxRP));
    ScaleAndShiftMe(embMatrix, 1.0F, float(maxRP));

    delete[] index;

    /* disable gradient flow */
    if (isTraining) {
        XTensor copyEmbMatrix;
        InitTensor(&copyEmbMatrix, &embMatrix);
        _CopyValues(&embMatrix, &copyEmbMatrix);
        return copyEmbMatrix;
    }

    return embMatrix;
}

/*
relative position-aware dot-product attention inner calculation.
>> x - Tensor with shape [batch_size*heads, length, length or depth].
>> y - Tensor with shape [batch_size*heads, length, depth].
>> z - Tensor with shape [length, length, depth].
>> isKey - Whether y is key.
<< return - A Tensor with shape [batch_size*heads, length, length or depth].
*/
XTensor Attention::RPDotProduct(XTensor& x, XTensor& y, XTensor& z, const bool isKey)
{
    const int headNum = nhead;
    const int batchSize = x.GetDim(1);
    const int lenQ = x.GetDim(2);
    const int lenKV = y.GetDim(2);
    const int depth = y.GetDim(3);

    const int lastDim = isKey ? lenKV : depth;
    auto transposeFlag = isKey ? X_TRANS : X_NOTRANS;

    int mergeDimsX[] = { headNum * batchSize, lenQ, x.GetDim(3) };
    int mergeDimsY[] = { headNum * batchSize, lenKV, y.GetDim(3) };
    
    x = Reshape(x, 3, mergeDimsX);
    y = Reshape(y, 3, mergeDimsY);

    if (isKey) {
        y = Transpose(y, 1, 2);
    }

    XTensor context;
    context = BMMul(x, y);

    int newDims[]{ headNum, batchSize, context.GetDim(1), context.GetDim(2) };

    context = Reshape(context, 4, newDims);

    XTensor xTrans;
    xTrans = Transpose(x, 0, 1);

    XTensor relative;
    relative = MatrixMulBatched(xTrans, X_NOTRANS, z, transposeFlag);

    XTensor relativeTrans;
    relativeTrans = Transpose(relative, 0, 1);

    int splitDims[] = { headNum, batchSize, lenQ, lastDim };

    relativeTrans = Reshape(relativeTrans, 4, splitDims);

    return Sum(context, relativeTrans);
}

/* constructor */
Cache::Cache()
{
    miss = true;
    enable = true;
}

/* update the states cache */
void Cache::Update(XTensor&& k, XTensor&& v)
{
    key = k;
    value = v;
    miss = false;
}

/* keep alive states */
void Cache::KeepAlive(XTensor& aliveIdx)
{
    if (!miss) {
        key = AutoGather(key, aliveIdx);
        value = AutoGather(value, aliveIdx);
    }
}

/* reorder alive states */
void Cache::Reorder(XTensor& reorder)
{
    if (!miss) {
        key = AutoGather(key, reorder);
        value = AutoGather(value, reorder);
    }
}

} /* end of the nmt namespace */
