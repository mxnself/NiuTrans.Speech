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

#include "S2TSearcher.h"
#include <iostream>
#include <fstream>
using namespace std;

namespace s2t {

	S2TGreedySearch::S2TGreedySearch()
	{
		maxLen = 0;
		batchSize = 0;
		endSymbolNum = 0;
		endSymbols = new int[32];
		startSymbolNum = 0;
		startSymbols = new int[64];
		suppressSymbolNum = 0;
		suppressSymbols = new int[100];
		scalarMaxLength = -1;
		vocab = NULL;
		withoutTimeStamps = true;
	}

	S2TGreedySearch::~S2TGreedySearch()
	{
		if (endSymbols != NULL)
			delete[] endSymbols;
		if (startSymbols != NULL)
			delete[] startSymbols;
		if (suppressSymbols != NULL)
			delete[] suppressSymbols;
		vocab = nullptr;

	}

	void S2TGreedySearch::Init(S2TConfig& config, S2TVocab* tgtVocab)
	{
		vocab = tgtVocab;
		int PadID;
		PadID = vocab->getpadID();
		withoutTimeStamps = config.whisperdec.withoutTimeStamps;
		maxLen = config.inference.maxLen;
		batchSize = config.common.sBatchSize;
		endSymbols[0] = config.model.eos;
		startSymbols[0] = config.model.sos;
		scalarMaxLength = config.inference.maxLenAlpha;

		if (endSymbols[0] >= 0)
			endSymbolNum = 1;
		if (startSymbols[0] >= 0)
			startSymbolNum = 1;
		if (PadID==-1){
			InitStartSymbols(config);

			const int tokenNum = 88;
			// blank 220
			// <eot> 50257
			int suppressTokens[tokenNum] = { 1, 2, 7, 8, 9, 10, 14, 25, 26, 27,
									28, 29, 31, 58, 59, 60, 61, 62, 63, 90,
									91, 92, 93, 359, 503, 522, 542, 873, 893,
									902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246,
									3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273,
									9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157,
									14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520,
									26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863,
									47425, 49870, 50254, 50258, vocab->taskIDs[0], vocab->taskIDs[1], vocab->startLM, vocab->startPrev, vocab->noSpeech };
			for (int i=0; i<tokenNum; i++)
				cout << suppressTokens[i] << " ";
			cout << endl;
			InitSuppressSymbols(config, suppressTokens, tokenNum);
		}
	}

	void S2TGreedySearch::InitPromptSymbols()
	{
		const int tokenNum = 27;
		int promptTokens[tokenNum] = { vocab->startPrev,   220, 35748,  9830,   241,  7781,  5975, 17174, 11249,   222,
									   29485,   171,   120,   234,  3509,   114, 20334,  9830,   241,  7781,
										 162,  3921, 12579,  5437,    99, 26987,  1543 };

		// int sos = startSymbols[0];

		for (startSymbolNum = 0; startSymbolNum < tokenNum; startSymbolNum++) {
			startSymbols[startSymbolNum] = promptTokens[startSymbolNum];
		}
		// startSymbols[startSymbolNum++] = sos;
	}

	void S2TGreedySearch::InitPromptSymbols(int* textPrompt, const int textPromptNum, int* decodingPrompt, const int decodingPromptNum) {
		CheckNTErrors((textPromptNum + decodingPromptNum) <= 200, "");
		startSymbolNum = 0;
		for (int i = 0; i < textPromptNum; i++, startSymbolNum++) 
			startSymbols[startSymbolNum] = textPrompt[i];
		for (int i = 0; i < decodingPromptNum; i++, startSymbolNum++) 
			startSymbols[startSymbolNum] = decodingPrompt[i];
	}

	void S2TGreedySearch::InitStartSymbols(S2TConfig& config)
	{
		startSymbolNum = 0;
		if (config.whisperdec.language.languageToken == 50260)	// Chinese prompt
			InitPromptSymbols();
		startSymbols[startSymbolNum++] = config.model.sos;
		startSymbols[startSymbolNum++] = config.whisperdec.language.languageToken; // en 50259
		startSymbols[startSymbolNum++] = vocab->taskIDs[1];			// 50359
		if (withoutTimeStamps)
			startSymbols[startSymbolNum++] = vocab->noTimeStamps; // notimestamps
	}

	bool S2TGreedySearch::IsEnd(int token)
	{
		CheckNTErrors(endSymbolNum > 0, "No end symbol?");

		for (int i = 0; i < endSymbolNum; i++) {
			if (endSymbols[i] == token)
				return true;
		}

		return false;
	}

	void S2TGreedySearch::InitSuppressSymbols(S2TConfig& config, int* tokens, const int num)
	{
		if ( num > 0 )
		{
			/*init suppress symbol from tokens*/ 
			CheckNTErrors(num <= 100, "Invalid suppress token length ( should less than 100 )");
			suppressSymbolNum = num;
			// blank 220
			// <eot> 50257
			for (int i = 0; i < suppressSymbolNum; i++) {
				suppressSymbols[i] = tokens[i];
			}
		}
		else {
			/*init suppress symbol from config*/
			/*TODO*/

		}
		
	}

	XTensor S2TGreedySearch::Suppress(IntList** tokens, XTensor& input, bool isBegin)
	{
		// input.Dump(stderr, NULL, 1);
		XTensor modify;
		InitTensor3D(&modify, input.GetDim(0), 1, 1, X_FLOAT, input.devID);
		modify = ScaleAndShift(modify, 0.0, -1e9);

		if (suppressSymbolNum <= 0)
			return input;

		for (int i = 0; i < suppressSymbolNum; i++) {
			_SetDataIndexed(&input, &modify, input.order - 1, suppressSymbols[i]);
		}
		if (isBegin) {
			// LOG("Doing Begin Suppress");
			_SetDataIndexed(&input, &modify, input.order - 1, 220);
			_SetDataIndexed(&input, &modify, input.order - 1, 50257);
		}

		// cout << "withoutTimeStamps: " << withoutTimeStamps << endl;
		if (!withoutTimeStamps) {
			// cout << "Doing Timestamp Rules Here" << endl;
			_SetDataIndexed(&input, &modify, input.order - 1, vocab->noTimeStamps);

			for (int k = 0; k < batchSize; k++) {
				bool lastWasTimestamp = false;
				bool penultimateWasTimestamp = false;
				if ((tokens[k]->count >= 1) && (tokens[k]->GetItem(tokens[k]->count -1) >= vocab->tStampIDs[0]))
					lastWasTimestamp = true;
				if ((tokens[k]->count < 2) || (tokens[k]->GetItem(tokens[k]->count -2) >= vocab->tStampIDs[0]))
					penultimateWasTimestamp = true;
				
				if (lastWasTimestamp) {
					if (penultimateWasTimestamp) {
						
						XTensor prob = SelectRange(input, 0, k, k+1);
						_SetDataDim(&prob, vocab->tStampIDs[0], (vocab->vocabSize - vocab->tStampIDs[0]), -1, -1e9);
						SqueezeMe(prob, 1);
						_SetDataIndexed(&input, &prob, 0, k);
					}
					else {
						XTensor prob = SelectRange(input, 0, k, k+1);
						_SetDataDim(&prob, 0, vocab->eosID, -1, -1e9);
						SqueezeMe(prob, 1);
						_SetDataIndexed(&input, &prob, 0, k);
					}
				}
				int numTimestamp = 0;
				int lastTimestamp = -1;
				for (int i=0; i < tokens[k]->count; i++)
					if (tokens[k]->GetItem(i) >= vocab->tStampIDs[0]) {
						numTimestamp++;
						lastTimestamp = tokens[k]->GetItem(i);
					}
						
				if (numTimestamp > 0) {
					if (!(lastWasTimestamp && !penultimateWasTimestamp))
						if (lastTimestamp + 1 < vocab->vocabSize)
							lastTimestamp++;
				}
				if (lastTimestamp != -1){
					XTensor prob = SelectRange(input, 0, k, k+1);
					_SetDataDim(&prob, vocab->tStampIDs[0], (lastTimestamp - vocab->tStampIDs[0]), -1, -1e9);
					SqueezeMe(prob, 1);
					_SetDataIndexed(&input, &prob, 0, k);
				}
			}

			if (isBegin) {
				for (int v = 0; v < vocab->tStampIDs[0]; v++)
					_SetDataIndexed(&input, &modify, input.order - 1, v);
				if (true) {
					for (int v = vocab->tStampIDs[0] + 51; v < vocab->vocabSize; v++)
						_SetDataIndexed(&input, &modify, input.order - 1, v);
				}
			}

			XTensor logits = Softmax(input, input.order-1);
			XTensor timestampLogprob = Log(ReduceSum(SelectRange(logits, logits.order-1, vocab->tStampIDs[0], vocab->vocabSize), logits.order-1));	//2D
			XTensor maxTextLogprob = ReduceMax(Log(SelectRange(logits, logits.order-1, 0, vocab->tStampIDs[0])), logits.order-1);
			for (int k = 0; k < batchSize; k++) {
				if (timestampLogprob.Get2D(k, 0) > maxTextLogprob.Get2D(k, 0)) {

					XTensor prob = SelectRange(input, 0, k, k+1);
					_SetDataDim(&prob, 0, vocab->tStampIDs[0], -1, -1e9);
					SqueezeMe(prob, 1);
					_SetDataIndexed(&input, &prob, 0, k);
				}
			}
		}

		return input;
	}

	XTensor S2TGreedySearch::Predict(XTensor& tokens, XTensor& logits, XTensor* sumLogprobs)
	{	
		// logits b * l * v
		/*TODO Temperature sets to 0.0*/
		XTensor nextToken, bestScore, logProbs;
		logits.Reshape(logits.dimSize[0], logits.dimSize[logits.order - 1]);	// b * v
		InitTensor2D(&nextToken, tokens.dimSize[0], 1, X_INT, tokens.devID);
		InitTensor2D(&bestScore, logits.dimSize[0], 1, X_FLOAT, logits.devID);
		TopK(logits, bestScore, nextToken, -1, 1);

		/*TODO calculate sumLogprobs*/
		// logProbs = Log(Softmax(logits, -1));

		/*modify tokens to <eot> if it appeared before*/
		XTensor lastToken = SelectRange(tokens, tokens.order - 1, tokens.GetDim(tokens.order - 1) - 1, tokens.GetDim(tokens.order - 1));
		// lastToken.Dump(stderr, "Last Tokens: ", -1);
		for (int i = 0; i < lastToken.GetDim(0); i++) {
			if (IsEnd(lastToken.GetInt(i)))
				nextToken.Set2DInt(endSymbols[0], i, 0);
		}
		return nextToken;
	}

	void S2TGreedySearch::Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs)
	{
		// cout << "----- S2T Greedy Search -----" << endl;

		XTensor maskEnc;
		XTensor encoding;
		batchSize = input.GetDim(0);

		/* encoder mask */
		model->MakeS2TMaskEnc(padding, maskEnc);

		/* make the encoding network */
		// cout << "----- Encoding -----" << endl;
		if (model->config->model.encPreLN)
			encoding = model->encoder->RunFastPreNorm(input, &maskEnc);
		/* max output-length = scalar * source-length */
		int lengthLimit = MIN(int(float(input.GetDim(-2)) * scalarMaxLength), maxLen);
		CheckNTErrors(lengthLimit > 0, "Invalid maximum output length");
		// cout << "lengthLimit: " << lengthLimit << endl;

		/* the first token */
		XTensor inputDec;
		InitTensor1D(&inputDec, startSymbolNum, X_INT, input.devID);
		inputDec.SetData(startSymbols, startSymbolNum);
		inputDec = Unsqueeze(inputDec, 0, batchSize);


		/* initialize the finished flags */
		int* finishedFlags = new int[batchSize];
		for (int i = 0; i < batchSize; i++)
			finishedFlags[i] = 0;

		XTensor prob, nextToken;
		XTensor maskDec;
		XTensor decoding;
		XTensor indexCPU;
		XTensor bestScore;

		InitTensor2D(&indexCPU, batchSize, 1, inputDec.dataType, -1);
		InitTensor2D(&bestScore, batchSize, 1, encoding.dataType, encoding.devID);

		int initTokenLen = inputDec.GetDim(-1);

		/* decoder mask */
		maskDec = model->MakeS2TTriMaskDecInference(batchSize, inputDec.GetDim(-1));

		model->decoder->embedder->scale = false;
		
		for (int l = 0; l < lengthLimit; l++) {

			// cout << "----- Decoding -----" << l << endl;

			int nstep = l;
			if (l > 0)
				nstep += (initTokenLen - 1);
			/* make the decoding network */
			if (model->config->model.decPreLN)
				if ( l == 0 ) {
					decoding = model->decoder->RunFastPreNorm(inputDec, encoding, &maskDec, NULL, nstep);

				}
				else {
					decoding = model->decoder->RunFastPreNorm(inputDec, encoding, NULL, NULL, nstep);

				}

			/* generate the output probabilities */
			/*TODO*/
			bool outputSoftmax = false;
			if (outputSoftmax)
				prob = model->outputLayer->Make(decoding, false);
			else
				prob = MMul(decoding, X_NOTRANS, *model->outputLayer->w, X_TRANS);

			// FILE* probOutput = fopen("../tools/data/probOutput.bin", "wb");
			// logits.BinaryDump(probOutput);
			// logits.Dump(stderr, "probOutput: ", 10);

			/*calculate prob of no_speech (whisper) */
			if (l == 0) {
				// no speech token 50362 TODO

			}

			/*only consider the last token*/
			prob = SelectRange(prob, 1, prob.GetDim(1) - 1, prob.GetDim(1));
			// logits.Dump(stderr, "logits: ", 10);

			/*apply the logit filters*/
			XTensor probFilted;
			probFilted = Suppress(outputs, prob, l==0);

			/*calculate next token*/
			nextToken = Predict(inputDec, probFilted);
			// nextToken.Dump(stderr, "New inputDec: ", -1);

			/* save the predictions */
			CopyValues(nextToken, indexCPU);
			
			for (int i = 0; i < batchSize; i++) {
				if (IsEnd(indexCPU.GetInt(i)))
					finishedFlags[i] = 1;
				else if (finishedFlags[i] != 1)
					(outputs[i])->Add(indexCPU.GetInt(i));
			}
			
			/*next loop*/
			inputDec = nextToken;

			// cout << "--- Decoding End ---" << l << endl;

			int finishedSentNum = 0;
			for (int i = 0; i < batchSize; i++)
				finishedSentNum += finishedFlags[i];
			if (finishedSentNum == batchSize) {
				l = lengthLimit;
				break;
			}
		}

		delete[] finishedFlags;

		// cout << "--- S2T Greedy Search End ---" << endl;
	}


	/*
		S2T Greedy Search with Non-Autoregressive Decoding
	*/
	void S2TGreedySearch::Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs, int audioLength)
	{
		cout << "----- S2T Greedy Search with **Non-Autoregressive Decoding** -----" << endl;
		XTensor sub_input, sub_padding;
		XTensor maskEnc, paddingMask;
		XTensor encoding, encodingTMP;
		XTensor encodingLength;
		XTensor encodingMask, encodingMaskDec;
		XTensor acousticEmbeds, lfracousticEmbeds;
		XTensor acousticEmbedsMask;
		XTensor predictLength;
		
		INT32 maskLength, preMaskLength;
		batchSize = input.GetDim(0);

		audioLength -= 1;
		sub_input = SelectRange(input, 1, 0, audioLength);
		sub_padding = SelectRange(padding, 1, 0, int(audioLength+1)/2);

		/* encoder mask */	
		model->MakeS2TMaskEnc(sub_padding, maskEnc);
		/* make the encoding network */
		// cout << "----- Encoding -----" << endl;
		if (model->config->model.encPreLN)
			encoding = model->encoder->RunFastPreNorm(sub_input, &maskEnc);
		// cout << "----- Encoding End-----" << endl;
		
		/*   apply lfr */
		lfracousticEmbeds = model->adapter->applyLfr(encoding);


		maskLength = lfracousticEmbeds.GetDim(1);
		InitTensor3D(&encodingMask, 1, 1, maskLength, input.dataType, 0.0);
		encodingMask.SetZeroAll();
		for (int i=0; i<maskLength; i++)
			encodingMask.Set3D(1.0, 0, 0, i);


		encoding = model->adapter->RunFastPreNorm(lfracousticEmbeds, &encodingMask);

		/*
		TODO: Pretictor
		explanation: Using Encoding vector to calculate semantic embedding vector and predict sentence length
		input:
		output: (return / pointer)
		*/

		acousticEmbeds = model->predictor->Make(encoding, encodingMask);
		preMaskLength = acousticEmbeds.GetDim(1);

		encodingTMP = lfracousticEmbeds; //SelectRange(encoding, 1, 0, maskLength);
		InitTensor1D(&paddingMask, lfracousticEmbeds.GetDim(1), input.dataType, 0.0);
		paddingMask.SetZeroAll();
		for (int i=0; i<maskLength; i++)
			paddingMask.Set1D(1.0, i);
		model->MakeEncDecMaskEnc(paddingMask, encodingMaskDec, preMaskLength);
		
		acousticEmbedsMask = model->MakeS2TTriMaskDecInference(acousticEmbeds.GetDim(0), acousticEmbeds.GetDim(1));
		acousticEmbedsMask.SetZeroAll();


		InitTensor1D(&predictLength, 1, X_INT, 0); // batch size
		predictLength.Set1DInt(preMaskLength, 0);

		
		/*
		TODO: NarDecoding
		explanation: Using Encoding vector and semantic embedding vector to calculate decoding results
		input: inputDec, encoding, &maskDec, NULL, nstep
		output: decoding
		*/

		XTensor decoding;

		/* max output-length = scalar * source-length */
		int lengthLimit = MIN(int(float(input.GetDim(-2)) * scalarMaxLength), maxLen);
		CheckNTErrors(lengthLimit > 0, "Invalid maximum output length");
		// cout << "lengthLimit: " << lengthLimit << endl;

		
		for (int i = 0; i < batchSize; i++) {
			int length = predictLength.Get1DInt(i);
			if(length < 1) {
				LOG("Predictor Exception: Length less than 1");
				// exit(1) // ??
			}
			else if (length > lengthLimit) {
				LOG("Predictor Exception: Length() more than limit()");
				predictLength.Set1DInt(lengthLimit, 0);
			}
		}
		
		if (model->config->model.decPreLN) {
			decoding = model->decoder->RunFastNarPreNorm(encodingTMP, &encodingMaskDec, acousticEmbeds, &acousticEmbedsMask);
		}

		/*
			XTensor after decoder layers  ->  output layer
			then softmax and calculate probablity and tokenID
		*/
		XTensor prob;
		
		// InitTensor2D(&bestScore, batchSize, 1, encoding.dataType, encoding.devID);
		
		
		bool outputSoftmax = true;
		if (outputSoftmax){
			prob = model->decoderOutputLayer->Make(decoding, false);
		}
		else
			prob = MMul(decoding, X_NOTRANS, *model->outputLayer->w, X_TRANS);
		
		/*calculate next token*/
		XTensor tokens, bestScore, indexCPU;
		InitTensor3D(&tokens, batchSize, prob.dimSize[1], 1, X_INT, prob.devID);
		InitTensor3D(&bestScore, batchSize, prob.dimSize[1], 1, input.dataType, prob.devID);
		InitTensor3D(&indexCPU, batchSize, prob.dimSize[1], 1, X_INT, -1);
		TopK(prob, bestScore, tokens, -1, 1); 
		
		/* save the predictions */
		CopyValues(tokens, indexCPU);
		
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < predictLength.Get1DInt(i); j++) // sentence_length
				(outputs[i])->Add(indexCPU.Get3DInt(i, j, 0));
		}

		cout << "--- S2TNonAutoregressiveSearch Search End ---" << endl;
	}

	S2TBeamSearch::S2TBeamSearch()
	{
		alpha = 0;
		maxLen = 0;
		beamSize = 0;
		batchSize = 0;
		endSymbolNum = 0;
		startSymbolNum = 0;
		suppressSymbolNum = 0;
		fullHypos = NULL;
		endSymbols = new int[32];
		startSymbols = new int[250];
		suppressSymbols = new int[100];
		isEarlyStop = false;
		needReorder = false;
		scalarMaxLength = 0.0F;
		vocab = NULL;
		withoutTimeStamps = false;
	}

	/* de-constructor */
	S2TBeamSearch::~S2TBeamSearch()
	{
		if (fullHypos != NULL)
			delete[] fullHypos;
		if (endSymbols != NULL)
			delete[] endSymbols;
		if (startSymbols != NULL)
			delete[] startSymbols;
		if (suppressSymbols != NULL)
			delete[] suppressSymbols;
		vocab = NULL;
	}

	void S2TBeamSearch::Init(S2TConfig& config, S2TVocab* tgtVocab)
	{
		vocab = tgtVocab;
		withoutTimeStamps = config.whisperdec.withoutTimeStamps;
		maxLen = config.inference.maxLen;
		beamSize = config.inference.beamSize;
		batchSize = config.common.sBatchSize;
		alpha = config.inference.lenAlpha;
		endSymbols[0] = config.model.eos;
		startSymbols[0] = config.model.sos;
		scalarMaxLength = config.inference.maxLenAlpha;

		if (endSymbols[0] >= 0)
			endSymbolNum = 1;
		if (startSymbols[0] >= 0)
			startSymbolNum = 1;

		InitStartSymbols(config);

		const int tokenNum = 88;
		// blank 220
		// <eot> 50257
		int suppressTokens[tokenNum] = { 1, 2, 7, 8, 9, 10, 14, 25, 26, 27,
								28, 29, 31, 58, 59, 60, 61, 62, 63, 90,
								91, 92, 93, 359, 503, 522, 542, 873, 893,
								902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246,
								3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273,
								9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157,
								14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520,
								26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863,
								47425, 49870, 50254, 50258, vocab->taskIDs[0], vocab->taskIDs[1], vocab->startLM, vocab->startPrev, vocab->noSpeech };
		for (int i=0; i<tokenNum; i++)
			cout << suppressTokens[i] << " ";
		cout << endl;

		InitSuppressSymbols(config, suppressTokens, tokenNum);
		BeamSearch::Init(endSymbols, endSymbolNum, startSymbols[0], maxLen, beamSize, batchSize, alpha, scalarMaxLength);
	}

	void S2TBeamSearch::InitPromptSymbols()
	{
		const int tokenNum = 27;
		int promptTokens[tokenNum] = { vocab->startPrev,   220, 35748,  9830,   241,  7781,  5975, 17174, 11249,   222,
									   29485,   171,   120,   234,  3509,   114, 20334,  9830,   241,  7781,
										 162,  3921, 12579,  5437,    99, 26987,  1543 };

		//int sos = startSymbols[0];

		for (startSymbolNum = 0; startSymbolNum < tokenNum; startSymbolNum++) {
			startSymbols[startSymbolNum] = promptTokens[startSymbolNum];
		}
		//startSymbols[startSymbolNum++] = sos;
	}

	void S2TBeamSearch::InitStartSymbols(S2TConfig& config)
	{
		startSymbolNum = 0;
		if (config.whisperdec.language.languageToken == 50260)	// Chinese prompt
			InitPromptSymbols();
		startSymbols[startSymbolNum++] = config.model.sos;
		startSymbols[startSymbolNum++] = config.whisperdec.language.languageToken; // en 50259
		startSymbols[startSymbolNum++] = vocab->taskIDs[1];			// 50359
		if (withoutTimeStamps)
			startSymbols[startSymbolNum++] = vocab->noTimeStamps; // notimestamps
	}

	void S2TBeamSearch::InitSuppressSymbols(S2TConfig& config, int* tokens, const int num)
	{
		if (num > 0)
		{
			/*init suppress symbol from tokens*/
			CheckNTErrors(num <= 100, "Invalid suppress token length ( should less than 100 )");
			suppressSymbolNum = num;
			// blank 220
			// <eot> 50257
			for (int i = 0; i < suppressSymbolNum; i++) {
				suppressSymbols[i] = tokens[i];
			}
		}
		else {
			/*init suppress symbol from config*/
			/*TODO*/

		}

	}

	/*
	prepare for search
	>> batchSize - size of the batch
	>> beamSize - size of the beam
	*/
	void S2TBeamSearch::Prepare(int myBatchSize, int myBeamSize)
	{
		batchSize = myBatchSize;
		beamSize = myBeamSize;
		needReorder = false;

		/* prepare for the heap of hypotheses */
		if (fullHypos != NULL)
			delete[] fullHypos;

		fullHypos = new XHeap<MIN_HEAP, float>[batchSize];

		for (int i = 0; i < batchSize; i++)
			fullHypos[i].Init(beamSize);

		/* prepare for the indices of alive states */
		aliveStatePids.Clear();
		aliveSentList.Clear();
		for (int i = 0; i < batchSize; i++) {
			aliveStatePids.Add(i);
			aliveSentList.Add(i);
		}
	}

	/*
	compute the model score for each hypotheses
	>> prev - the beam of the previous state
	>> beam - the beam that keeps a number of states
	*/
	void S2TBeamSearch::Score(StateBundle* prev, StateBundle* beam)
	{
		XTensor& score = beam->modelScore;
		XTensor& prob = beam->prob;
		XTensor& probPath = beam->probPath;
		XTensor& probPathPrev = prev->probPath;
		XTensor mask;

		int order = prob.order;
		int outputSize = prob.dimSize[prob.order - 1];
		int dims[MAX_TENSOR_DIM_NUM];
		for (int i = 0; i < order; i++)
			dims[i] = prob.dimSize[i];

		if (prob.dataType == X_FLOAT16)
			prob = ConvertDataType(prob, X_FLOAT);

		InitTensor(&score, &prob);
		InitTensor(&probPath, &prob);

		prob.Reshape(prob.unitNum / outputSize, outputSize);
		score.Reshape(score.unitNum / outputSize, outputSize);
		probPath.Reshape(score.unitNum / outputSize, outputSize);
		probPathPrev.Reshape(probPathPrev.unitNum);

		/* the log-scale probability of the entire sequence */
		SumDim(prob, probPathPrev, probPath, 0);

		// beam->nstep = prev->nstep + 1.0F;
		/* the GNMT-like length penalty */
		// float lp = LengthPenalizer::GNMT(beam->nstep, alpha);

		/* score = log-prob/lp */
		// score = probPath / lp;
		score = probPath / 1.00;

		if (prev->isStart) {
			XTensor firstMask = MakeFirstMask(beam);
			firstMask.Reshape(firstMask.unitNum);

			/* mask the hypotheses in the beam except the first one */
			SumDim(score, firstMask, score, 0);
		}

		InitTensor(&mask,
			prev->endMark.order, prev->endMark.dimSize, X_FLOAT,
			prev->endMark.devID);
		mask.SetZeroAll();
		_SetDataFixedCond(&mask, &prev->endMark, -1e9F);

		mask.Reshape(mask.unitNum);

		/* mask the completed hypotheses so that they cannot
		be involved in further sorting and beam search. */
		SumDim(score, mask, score, 0);

		prob.Reshape(order, dims);
		score.Reshape(order, dims);
		probPath.Reshape(order, dims);
	}

	/*
	expand the search graph
	>> prev - the last beam
	>> beam - the beam that keeps a number of states
	>> reorderState - the new order of states
	*/
	void S2TBeamSearch::Expand(StateBundle* prev, StateBundle* beam, XTensor& reorderState)
	{
		CheckNTErrors(beam->prediction.unitNum == beam->preID.unitNum, 
					"A problem occurs in the beam!");

		beam->MakeStates(beam->prediction.unitNum);

		State* states = beam->states;
		XTensor& idRef = beam->preID;
		XTensor& modelScoreRef = beam->modelScore;
		XTensor& probRef = beam->prob;
		XTensor& probPathRef = beam->probPath;
		XTensor& predictionRef = beam->prediction;
		XTensor& endMark = beam->endMark;
		XTensor id;
		XTensor modelScore;
		XTensor prob;
		XTensor probPath;
		XTensor prediction;
		XTensor endMarkCPU;
		XTensor reorderStateCPU;

		InitTensorOnCPU(&id, &idRef);
		InitTensorOnCPU(&modelScore, &modelScoreRef);
		InitTensorOnCPU(&prob, &probRef);
		InitTensorOnCPU(&probPath, &probPathRef);
		InitTensorOnCPU(&prediction, &predictionRef);
		InitTensorOnCPU(&endMarkCPU, &predictionRef);
		InitTensor(&endMark, &predictionRef);
		InitTensorOnCPU(&reorderStateCPU, &reorderState);

		/* we copy the data to CPU because the frequent access to GPU is slow
		and we can speed-up the process by doing the job on CPU. */
		CopyValues(idRef, id);
		CopyValues(modelScoreRef, modelScore);
		CopyValues(probRef, prob);
		CopyValues(probPathRef, probPath);
		CopyValues(predictionRef, prediction);

		CheckNTErrors(beam->stateNum == id.unitNum, "Errors occur in counting!");

		/* Related variables are kept on the states of the graph. All these are
		maintained on CPUs to ease the implementation of frequent access and
		modification of the states. An alternative is to do this on GPUs but
		it needs much more coding work and the speed-up is not obvious. */

		for (int i = 0; i < beam->stateNum; i += beamSize) {
			for (int j = 0; j < beamSize; j++) {
				int k = i + j;
				State& state = states[k];

				int offset = id.GetInt(k);
				int pid = i / beamSize;
				reorderStateCPU.SetInt(i + offset, i + j);
				if (offset != j)
					needReorder = true;

				State* last = prev->states + pid * beamSize + offset;

				CheckNTErrors(offset >= 0, "Wrong state index!");

				/* pointer to the previous state */
				if (prev->isStart) {
					state.last = NULL;
					state.pid = pid;
					state.nstep = 0;
					state.isCompleted = false;
				}
				else {
					state.last = last;
					state.pid = state.last->pid;
					if (state.last->isCompleted) 
						state.nstep = last->nstep;
					else
						state.nstep = last->nstep + 1;
					state.isCompleted = last->isCompleted;
					CheckNTErrors(offset < prev->stateNum, "Wrong state index!");
				}
				/*if(aliveStatePids.size() < batchSize)
					state.pid = aliveStatePids[i/beamSize];*/

				/* scores */
				state.modelScore = modelScore.Get(k);
				state.prob = prob.Get(k);
				state.probPath = probPath.Get(k);

				/* prediction */
				state.prediction = prediction.GetInt(k);

				CheckNTErrors(state.prediction >= 0, "Illegal prediction!");

				/* check if it is the end of the sequence */
				state.isEnd = IsEnd(state.prediction);
				state.isCompleted = (state.isCompleted || state.isEnd);

				/* set the ending mark */
				endMarkCPU.SetInt(state.isEnd, k);
			}
		}

		/* copy the ending mark from CPU to the target device */
		CopyValues(endMarkCPU, endMark);
		CopyValues(reorderStateCPU, reorderState);
	}

	/*
	collect hypotheses with ending symbols. Given a beam of hypotheses,
	we remove the finished hypotheses and keep them in a heap.
	>> beam  - the beam that keeps a number of states
	*/
	void S2TBeamSearch::Collect(StateBundle* beam)
	{
		State* states = beam->states;

		for (int i = 0; i < beam->stateNum; i++) {
			State& state = states[i];

			CheckNTErrors(state.pid >= 0 && state.pid < batchSize,
				"Invalid sample id!");

			/* check if this is the first end symbol. It is false
			   if there have been end symbols in previously generated words. */
			bool isCompleted = state.isCompleted &&
				(state.last == NULL || !state.last->isCompleted);

			/* we push the hypothesis into the heap when it is completed */
			if ((state.isEnd && isCompleted)) {

				int length = state.nstep;
				float lp = LengthPenalizer::GNMT(length, alpha);
				state.modelScore = state.modelScore / lp;

				fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
			}
		}
	}

	/*
	fill the hypothesis heap with incomplete hypotheses
	>> beam  - the beam that keeps a number of states (final)
	*/
	void S2TBeamSearch::FillHeap(StateBundle* beam)
	{
		State* states = beam->states;

		for (int i = 0; i < beam->stateNum / beamSize; i++) {
			for (int j = 0; j < beamSize; j++) {
				State& state = states[i * beamSize + j];

				/* we push the incomplete hypothesis into the heap */
				if (fullHypos[state.pid].Count() == 0) {
					fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
				}
				else {
					auto node = fullHypos[state.pid].Top();
					float score = node.value;
					if (score < state.modelScore)
						fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
				}
			}
		}
	}

	/*
	save the output sequences in a tensor
	>> output - output sequences (for return)
	>> score - score of thes sequences
	*/
	void S2TBeamSearch::Dump(IntList** output, XTensor* score)
	{
		LOG("Dump Results");
		int dims[3] = { batchSize, 1 };

		InitTensor(score, 2, dims, X_FLOAT);
		score->SetZeroAll();

		/* heap for an input sentence in the batch */
		for (int h = 0; h < batchSize; h++) {
			IntList* tgt = output[h];
			XHeap<MIN_HEAP, float>& heap = fullHypos[h];
			int c = heap.Count();

			float bestScore = -1e9F;
			State* state = NULL;
			for (int i = 0; i < c; i++) {
				auto node = heap.Pop();
				State* s = (State*)node.index;
				if (i == 0 || bestScore < node.value) {
					state = s;
					bestScore = node.value;
				}
			}

			int count = 0;
			bool isCompleted = true;

			/* we track the state from the end to the beginning */
			while (state != NULL) {
				if (!state->isCompleted)
					isCompleted = false;
				if (!isCompleted) {
					tgt->Add(state->prediction);
				}
				state = state->last;
			}
			tgt->Reverse();

			score->Set2D(bestScore, h, 0);
		}
	}

	/*
	check if the token is an end symbol
	>> token - token to be checked
	*/
	bool S2TBeamSearch::IsEnd(int token)
	{
		CheckNTErrors(endSymbolNum > 0, "No end symbol?");

		for (int i = 0; i < endSymbolNum; i++) {
			if (endSymbols[i] == token)
				return true;
		}

		return false;
	}

	/*
	search for the most promising states
	>> model - the transformer model
	>> input - input of the model
	>> padding - padding of the input
	>> outputs - outputs that represent the sequences as rows
	>> score - score of the sequences
	*/
	void S2TBeamSearch::Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs, XTensor& score)
	{
		cout << "----- S2TBeamSearch Search -----" << endl;
		
		S2TPredictor predictor;
		XTensor maskEnc;
		XTensor encoding;
		XTensor inputBeam;
		XTensor paddingBeam;

		CheckNTErrors(endSymbolNum > 0, "The search class is not initialized!");
		CheckNTErrors(startSymbolNum > 0, "The search class is not initialized!");

		Prepare(input.GetDim(0), beamSize);

		/* encoder mask */
		model->MakeS2TMaskEnc(padding, maskEnc);

		/* make the encoding network */
		// cout << "----- Encoding -----" << endl;
		if (model->config->model.encPreLN)
			encoding = model->encoder->RunFastPreNorm(input, &maskEnc);
		// cout << "--- Encoding End ---" << endl;

		inputBeam = Unsqueeze(input, input.order - 2, beamSize);
		paddingBeam = Unsqueeze(padding, padding.order - 1, beamSize);

		inputBeam.ReshapeMerged(inputBeam.order - 4);
		paddingBeam.ReshapeMerged(paddingBeam.order - 3);

		/* max output-length = scalar * source-length */
		int lengthLimit = MIN(int(float(input.GetDim(-2)) * scalarMaxLength), maxLen);
		CheckNTErrors(lengthLimit > 0, "Invalid maximum output length");
		// cout << "lengthLimit: " << lengthLimit << endl;

		StateBundle* states = new StateBundle[lengthLimit + 1];
		StateBundle* first = states;
		StateBundle* cur = NULL;
		StateBundle* next = NULL;

		/* create the first state */
		predictor.Init(batchSize, beamSize, endSymbols, endSymbolNum, startSymbols, startSymbolNum, suppressSymbols, suppressSymbolNum, vocab, model->config->whisperdec.withoutTimeStamps);
		predictor.Create(&input, beamSize, first);
		
		first->isStart = true;

		XTensor aliveState;
		InitTensor1D(&aliveState, batchSize * beamSize, X_INT, input.devID);
		SetAscendingOrder(aliveState, 0);

		XTensor reorderState;
		InitTensor1D(&reorderState, batchSize * beamSize, X_INT, input.devID);
		SetAscendingOrder(reorderState, 0);

		model->decoder->embedder->scale = false;

		/* generate the sequence from left to right */
		// lengthLimit = 1;
		for (int l = 0; l < lengthLimit; l++) {

			int nstep = l;
			if (l > 0)
				nstep += (startSymbolNum - 1);
			
			if (beamSize > 1) {
				inputBeam = AutoGather(inputBeam, reorderState);
				paddingBeam = AutoGather(paddingBeam, reorderState);
			}

			cur = states + l;
			next = states + l + 1;

			/* read the current state */
			predictor.Read(model, cur);

			/* predict the next state */
			predictor.Predict(next, aliveState, encoding, inputBeam,
				paddingBeam, batchSize * beamSize, l == 0, reorderState, needReorder, nstep, outputs);

			/* compute the model score (given the prediction probability) */
			Score(cur, next);

			/* beam pruning */
			Generate(cur, next);

			/* expand the search graph */
			Expand(cur, next, reorderState);

			/* push complete hypotheses into the heap */
			Collect(next);

			/* stop searching when all hypotheses are completed */
			if (IsAllCompleted(next)) {
				break;
			}

			/* remove finished sentences */
			//RemoveFinishedStates(next, encodingBeam, inputBeam, paddingBeam, aliveState);
		}

		/* fill the heap with incomplete hypotheses if necessary */
		// FillHeap(next);

		Dump(outputs, &score);

		delete[] states;

		cout << "--- S2TBeamSearch Search End ---" << endl;
	}

	S2TPredictor::S2TPredictor()
	{
		m = NULL;
		s = NULL;
		endSymbols = NULL;
		endSymbolNum = 0;
		startSymbols = NULL;
		startSymbolNum = 0;
		suppressSymbols = NULL;
		suppressSymbolNum = 0;
		vocab = NULL;
		withoutTimeStamps = true;
	}

	S2TPredictor::~S2TPredictor()
	{
		m = NULL;
		s = NULL;
		endSymbols = NULL;
		startSymbols = NULL;
		suppressSymbols = NULL;
		vocab = NULL;
	}

	void S2TPredictor::Init(int batchN, int beamN, int* endS, int endN, int* startS, int startN, int* suppS, int suppN, S2TVocab* tgtVocab, bool noTimeStamps)
	{
		batchSize = batchN;
		beamSize = beamN;
		vocab = tgtVocab;
		endSymbols = endS;
		endSymbolNum = endN;
		startSymbols = startS;
		startSymbolNum = startN;
		suppressSymbols = suppS;
		suppressSymbolNum = suppN;
		withoutTimeStamps = noTimeStamps;
	}

	/*
	create an initial state
	>> model - the  model
	>> top - the top-most layer of the network
	>> input - input of the network
	>> beamSize - beam size
	>> state - the state to be initialized
	*/
	void S2TPredictor::Create(const XTensor* input, int beamSize, StateBundle* state)
	{
		/*TODO*/
		int dims[MAX_TENSOR_DIM_NUM];
		dims[0] = input->dimSize[0];
		dims[1] = beamSize;

		InitTensor(&state->probPath, input->order-1, dims, X_FLOAT, input->devID);
		InitTensor(&state->endMark, input->order-1, dims, X_INT, input->devID);

		state->probPath.SetZeroAll();
		state->nstep = 0.0F;
		state->endMark.SetZeroAll();

		state->stateNum = 0;
	}

	/*
	read a state
	>> model - the  model that keeps the network created so far
	>> state - a set of states. It keeps
	1) hypotheses (states)
	2) probabilities of hypotheses
	3) parts of the network for expanding toward the next state
	*/
	void S2TPredictor::Read(S2TModel* model, StateBundle* state)
	{
		m = model;
		s = state;
	}

	XTensor S2TPredictor::Suppress(StateBundle* cur, XTensor& input, bool isBegin)
	{
		XTensor modify;
		InitTensor3D(&modify, input.GetDim(0), 1, 1, X_FLOAT, input.devID);
		modify = ScaleAndShift(modify, 0.0, -1e9);

		if (suppressSymbolNum <= 0)
			return input;

		for (int i = 0; i < suppressSymbolNum; i++) {
			_SetDataIndexed(&input, &modify, input.order - 1, suppressSymbols[i]);
		}
		if (isBegin) {
			// LOG("Doing Begin Suppress");
			_SetDataIndexed(&input, &modify, input.order - 1, 220);
			_SetDataIndexed(&input, &modify, input.order - 1, 50257);
		}

		// cout << "withoutTimeStamps: " << withoutTimeStamps << endl;
		if (!withoutTimeStamps) {
			
			_SetDataIndexed(&input, &modify, input.order - 1, vocab->noTimeStamps);
			// _SetDataDim(&input, vocab->noTimeStamps, 1, -1, -1e9);
			if (isBegin) {
				int v = 0;
				// for (v = 0; v < vocab->tStampIDs[0]; v++)
				// 	_SetDataIndexed(&input, &modify, input.order - 1, v);
				_SetDataDim(&input, 0, vocab->tStampIDs[0], -1, -1e9);
				if (true) {
					// for (v = vocab->tStampIDs[0] + 51; v < vocab->vocabSize; v++)
					// 	_SetDataIndexed(&input, &modify, input.order - 1, v);
					_SetDataDim(&input, vocab->tStampIDs[0] + 51, (vocab->vocabSize -  vocab->tStampIDs[0] - 51), -1, -1e9);
				}

				// SelectRange(input, 2, vocab->tStampIDs[0] + 51, vocab->tStampIDs[0] + 100).Dump(stderr, NULL, -1);
			}
			
			State* states = cur->states;
			// cout << "-------------"<< endl;
			for (int k = 0; k < batchSize * beamSize; k++) {
				
				int lastToken = -1;
				int penultimateToken = -1;
				int lastTimestamp = -1;
				bool lastWasTimestamp = false;
				bool penultimateWasTimestamp = false;
				int length = 0;

				if (cur->stateNum != 0) {
					CheckNTErrors(cur->stateNum == (batchSize * beamSize), "Unmatched size!");
					State* state = states + k;

					// cout << k << "--"; 
					bool isCompleted = true;
					while (state != NULL) {
						if (!state->isCompleted)
							isCompleted = false;
						if (!isCompleted) {
							length++;
							// cout << state->prediction << " ";
							if (length == 1)
								lastToken = state->prediction;
							else if (length == 2)
								penultimateToken = state->prediction;
							if (lastTimestamp == -1 && state->prediction >= vocab->tStampIDs[0])
								lastTimestamp = state->prediction;
							if (lastToken != -1 && penultimateToken != -1 && lastTimestamp != -1)
								break;
						}
						state = state->last;
					}
				}

				if ((length >= 1) && (lastToken >= vocab->tStampIDs[0]))
					lastWasTimestamp = true;
				if ((length < 2) || (penultimateToken >= vocab->tStampIDs[0]))
					penultimateWasTimestamp = true;
	
				if (lastWasTimestamp) {
					if (penultimateWasTimestamp)
					{
						// for (int v = vocab->tStampIDs[0]; v < vocab->vocabSize; v++)
						// 	input.Set3D(-1e9, k, 0, v);
						XTensor prob = SelectRange(input, 0, k, k+1);
						_SetDataDim(&prob, vocab->tStampIDs[0], (vocab->vocabSize - vocab->tStampIDs[0]), -1, -1e9);
						SqueezeMe(prob, 1);
						_SetDataIndexed(&input, &prob, 0, k);
					}
					else {
						// for (int v = 0; v < vocab->eosID; v++)	// should be v < vocab->eosID
						// 	input.Set3D(-1e9, k, 0, v);
						XTensor prob = SelectRange(input, 0, k, k+1);
						_SetDataDim(&prob, 0, vocab->eosID, -1, -1e9);
						SqueezeMe(prob, 1);
						_SetDataIndexed(&input, &prob, 0, k);
					}
				}

				if (lastTimestamp != -1) {
					if (!(lastWasTimestamp && !penultimateWasTimestamp))
						if (lastTimestamp + 1 < vocab->vocabSize)
							lastTimestamp++;
				
					// for (int v = vocab->tStampIDs[0]; v < lastTimestamp; v++)
					// 	input.Set3D(-1e9, k, 0, v);

					XTensor prob = SelectRange(input, 0, k, k+1);
					_SetDataDim(&prob, vocab->tStampIDs[0], (lastTimestamp - vocab->tStampIDs[0]), -1, -1e9);
					SqueezeMe(prob, 1);
					_SetDataIndexed(&input, &prob, 0, k);

					// CheckNTErrors(false, "");
				}
			}

			XTensor logits = Softmax(input, input.order-1);
			XTensor timestampLogprob = Log(ReduceSum(SelectRange(logits, logits.order-1, vocab->tStampIDs[0], vocab->vocabSize), logits.order-1));	//2D
			XTensor maxTextLogprob = ReduceMax(Log(SelectRange(logits, logits.order-1, 0, vocab->tStampIDs[0])), logits.order-1);
			for (int k = 0; k < batchSize * beamSize; k++) {
				if (timestampLogprob.Get2D(k, 0) > maxTextLogprob.Get2D(k, 0)) {
					// for (int v = 0; v < vocab->tStampIDs[0]; v++)
					// 	input.Set3D(-1e9, k, 0, v);

					XTensor prob = SelectRange(input, 0, k, k+1);
					_SetDataDim(&prob, 0, vocab->tStampIDs[0], -1, -1e9);
					SqueezeMe(prob, 1);
					_SetDataIndexed(&input, &prob, 0, k);
				}
			}
		}

		return input;
	}

	void S2TPredictor::Predict(StateBundle* next, XTensor& aliveState, XTensor& encoding,
		XTensor& inputEnc, XTensor& paddingEnc, int batchSize, bool isStart,
		XTensor& reorderState, bool needReorder, int nstep, IntList** outputs)
	{
		int dims[MAX_TENSOR_DIM_NUM];
		/* word indices of positions up to next state */
		XTensor inputDec;

		/* the first token */
		XTensor first;
		InitTensor1D(&first, startSymbolNum, X_INT, inputEnc.devID);
		first.SetData(startSymbols, startSymbolNum);
		first = Unsqueeze(first, 0, batchSize);

		/* add a new word into the input sequence of the decoder side */
		if (isStart) {
			inputDec = Identity(first);
		}
		else {
			/* only pass one step to the decoder */
			inputDec = GetLastPrediction(s, inputEnc.devID);
		}
		// inputDec.Dump(stderr, "inputDec: ", -1);

		/* keep alive states for the decoder */
		if (aliveState.dimSize[0] < batchSize) {
			/* alive inputs */
			inputDec = AutoGather(inputDec, aliveState);

			/* alive cache */
			for (int i = 0; i < m->decoder->nlayer; i++) {
				m->decoder->selfAttCache[i].KeepAlive(aliveState);
				m->decoder->enDeAttCache[i].KeepAlive(aliveState);
			}
		}

		if (needReorder) {
			for (int i = 0; i < m->decoder->nlayer; i++) {
				m->decoder->selfAttCache[i].Reorder(reorderState);
			}
		}

		/* prediction probabilities */
		XTensor& output = next->prob;
		XTensor decoding;

		for (int i = 0; i < inputDec.order - 1; i++)
			dims[i] = inputDec.dimSize[i];
		dims[inputDec.order - 1] = inputDec.dimSize[inputDec.order - 1];

		XTensor maskDec;

		/* decoder mask */
		maskDec = m->MakeS2TTriMaskDecInference(batchSize, inputDec.GetDim(-1));

		// inputDec.Dump(stderr, "inputDec: ", 1);
		/* make the decoding network */
		if (m->config->model.decPreLN)
			if (isStart)
				decoding = m->decoder->RunFastPreNorm(inputDec, encoding, &maskDec, NULL, nstep);
			else
				decoding = m->decoder->RunFastPreNorm(inputDec, encoding, NULL, NULL, nstep);
		
		/* TODO
		else
			decoding = m->decoder->RunFastPostNorm(inputDec, encoding, &maskEncDec, nstep);*/

		CheckNTErrors(decoding.order >= 2, "The tensor must be of order 2 or larger!");

		/* generate the output probabilities */
		bool outputSoftmax = false;
		if (outputSoftmax)

		
			output = m->outputLayer->Make(decoding, false);
		else
			output = MMul(decoding, X_NOTRANS, *m->outputLayer->w, X_TRANS);

		/*only consider the last token*/
		output = SelectRange(output, 1, output.GetDim(1) - 1, output.GetDim(1));

		output = Suppress(next-1, output, isStart);
		
		output = Log(Softmax(output, -1));
	}

}
