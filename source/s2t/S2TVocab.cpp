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

#include <fstream>
#include <iostream>
#include <sstream>
#include <locale>
#include <codecvt>
#include "S2TVocab.h"
#include "./S2TConfig.h"

using namespace std;

namespace s2t
{
    std::string decodeEscapedString(const std::string& escaped) {
        std::string result;
        std::istringstream stream(escaped);
        char ch;

        while (stream >> std::noskipws >> ch) { // std::noskipws 确保空格和标点不被跳过
            if (ch == '\\') {
                if (stream >> ch && ch == 'x') {
                    std::string code;
                    code.push_back(stream.get()); // 获取十六进制数的第一位
                    code.push_back(stream.get()); // 获取十六进制数的第二位
                    char byte = static_cast<char>(std::stoi(code, nullptr, 16)); // 将十六进制字符串转换为字符
                    result += byte;
                } else {
                    // 如果 \ 后面不是 x，就将 \ 和后面的字符直接添加到结果中
                    result += '\\';
                    result += ch;
                }
            } else {
                // 直接将非转义字符添加到结果字符串中
                result += ch;
            }
        }
        return result;
    }

/* set ids for special tokens */
/* TODO!!! update for multilingual tokens */
void S2TVocab::SetSpecialID(int sos, int eos, int pad, int unk, int numLanguage)
{
    sosID = sos;
    eosID = eos;
    padID = pad;
    unkID = unk;

    if (padID == -1){
        if (numLanguage != -1){
            cout << "Vocab size: " << vocabSize << " Language: " << numLanguage << endl;
            int index = sosID + 1;
            langIDs = new int[numLanguage];
            for (int i = 0; i < numLanguage; i++){
                langIDs[i] = index;
                index++;
            }
            taskIDs = new int[2];
            for (int i = 0; i < 2; i++){
                taskIDs[i] = index;
                index++;
            }
            startLM = index;
            index++;
            startPrev = index;
            index++;
            noSpeech = index;
            index++;
            noTimeStamps = index;
            index++;
            tStampIDs = new int[1501];
            for (int i = 0; i < 1501; i++){
                tStampIDs[i] = index;
                index++;
            }
        }
    }
}

int S2TVocab::getpadID(){
    return padID;
}

void S2TVocab::InitMask(int devID)
{
    InitTensor1D(&timestampMask, vocabSize, X_INT, devID, false);
    InitTensor1D(&nonTimestampMask, vocabSize, X_INT, devID, false);
    InitTensor1D(&contentMask, vocabSize, X_INT, devID, false);
    cout << "Vocab Check: " << vocabSize << " " << eosID << " " << tStampIDs[0] << " " << endl;
    for (int v=0; v < vocabSize; v++) {
        if (v < eosID) {
            timestampMask.Set1DInt(1, v);
            contentMask.Set1DInt(0, v);
            nonTimestampMask.Set1DInt(0, v);
        }
        else if (v < tStampIDs[0]) {
            timestampMask.Set1DInt(1, v);
            contentMask.Set1DInt(1, v);
            nonTimestampMask.Set1DInt(0, v);
        }
        else {
            timestampMask.Set1DInt(0, v);
            contentMask.Set1DInt(1, v);
            nonTimestampMask.Set1DInt(1, v);
        }
    }
    
}

/* constructor */
S2TVocab::S2TVocab()
{
    sosID = -1;
    eosID = -1;
    padID = -1;
    unkID = -1;
    vocabSize = -1;
    langIDs = NULL;
    tStampIDs = NULL;
    taskIDs = NULL;
}

/* de-constructor */
S2TVocab::~S2TVocab() 
{
    if (langIDs)
        delete langIDs;
    if (tStampIDs)
        delete tStampIDs;
    if (taskIDs)
        delete taskIDs;
}

void S2TVocab::Load(const string& vocabFN)
{
    string vsz, sid;
    ifstream f(vocabFN, ios::in);
    CheckNTErrors(f.is_open(), "Failed to open the vocabulary file");
        
    /* get the vocab size */
    std::getline(f, vsz);
    // sosID = (int)stol(sid);
    vocabSize = (int)stol(vsz);

    string word, id;
    for (int i = 0; i < vocabSize; i++) {
        // f >> word >> id;
        string line;
        std::getline(f, line);
        //cout << line << endl;
        size_t pos = line.find('\t');
        word = line.substr(0, pos);
        id = line.substr(pos+1, line.length()- (pos + 1));
        //cout << word << " " << id << endl;

        token2id[word] = (int)stol(id);
        id2token[(int)stol(id)] = word;

        //cout << word << " " << id << endl;
    }

    f.close();
}

/* save a vocabulary to a file */
void S2TVocab::Save(const string& vocabFN)
{
    ofstream f(vocabFN, ios::out);

    /* the first line: size of the vocab and the start id */
    f << vocabSize << "\t" << sosID;

    /* other lines: words and indices */
    for (const auto& p : token2id)
        f << p.first << "\t" << p.second;

    f.close();
}

/*
copy data from another vocabulary
>> v - the target vocabulary
*/
void S2TVocab::CopyFrom(const S2TVocab& v)
{
    for (const auto& w2i : v.token2id)
        token2id.insert(w2i);

    for (const auto& i2w : v.id2token)
        id2token.insert(i2w);
}

void S2TVocab::ShowVocab()
{
    for (int i = 0; i < vocabSize; i++) {
        cout << id2token[i] << "\t" << i << endl;
    }
    cout << "Vocab size: " << vocabSize << endl;
}

string S2TVocab::DecodingWord(vector<int>* tokensId)
{
    string result = "";
    for (int i = 0; i < tokensId->size(); i++) {
        string token = id2token[(*tokensId)[i]];
        // token = token.substr(2, token.length() - 2 - 1);
        result += token;
    }
    std::string utf8String(decodeEscapedString(result));
    // std::cout << "Encoded string: " << result << std::endl;
    std::cout << "Decoded string: " << utf8String << std::endl;

    return utf8String;
}

void S2TVocab::Test()
{
    // utf-8 decode test
    vector<int> tokensId = { 50364, 2664, 1530, 223, 17665, 4510, 104, 19488, 13545, 14812, 6866, 19563, 5157, 50536 };
    vector<string> tokensUtf8;
    cout << "ids" << "-->" << "tokens" << endl;
    for (int i = 0; i < tokensId.size(); i++) {
        string token = id2token[tokensId[i]];
        // token = token.substr(2, token.length() - 2 - 1);
        tokensUtf8.push_back(token);
        cout << tokensId[i] << "-->" << token << endl;
    }
    cout << endl;

}

} /* end of the s2t namespace */
