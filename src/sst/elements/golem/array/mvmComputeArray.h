// Copyright 2009-2025 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2025, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#ifndef _MVMCOMPUTEARRAY_H
#define _MVMCOMPUTEARRAY_H

#include <sst/elements/golem/array/computeArray.h>
#include <algorithm>
#include <type_traits>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace SST {
namespace Golem {

template<typename T>
class MVMComputeArray : public ComputeArray {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_DERIVED_API(
            MVMComputeArray<T>, 
            SST::Golem::ComputeArray,
            TimeConverter*,
            Event::HandlerBase*
    )

    MVMComputeArray(ComponentId_t id, Params& params,
                         TimeConverter* tc,
                         Event::HandlerBase* handler)
        : ComputeArray(id, params, tc, handler) {
        // Configure selfLink
        selfLink = configureSelfLink("Self", *tc, new Event::Handler2<MVMComputeArray,&MVMComputeArray::handleSelfEvent>(this));
        selfLink->setDefaultTimeBase(*clockTC);

        // Initialize vectors   numArrays表示有几个矩阵
        inputVectors.resize(numArrays);
        outputVectors.resize(numArrays);
        matrixData.resize(numArrays);
        outputModes.resize(numArrays, OutputMode::Overwrite);
        for (uint32_t i = 0; i < numArrays; i++) {
            inputVectors[i].resize(inputArraySize, T());
            outputVectors[i].resize(outputArraySize, T());
            matrixData[i].resize(inputArraySize * outputArraySize, T());
        }

        dumpEnabled = params.find<int>("mvm_dump_enable", 0) != 0;
        coreId = params.find<int>("core_id", -1);
        dumpRootDir = params.find<std::string>("mvm_dump_dir", "mvm_dumps");
        const std::string dumpMode = params.find<std::string>("mvm_dump_mode", "overwrite");
        dumpOverwrite = (dumpMode != "append");

        if (dumpEnabled) {
            std::ostringstream folder;
            folder << dumpRootDir << "/core_" << coreId;
            dumpCoreDir = folder.str();
            std::error_code ec;
            std::filesystem::create_directories(dumpCoreDir, ec);
            if (ec) {
                out.verbose(CALL_INFO, 1, 0,
                            "mvmComputeArray: failed to create dump dir %s, disable dump\n",
                            dumpCoreDir.c_str());
                dumpEnabled = false;
            }
        }
    }
    //启动计算 根据 arrayID 获取阵列的延迟，并通过 selfLink 发送一个计算事件
    virtual void beginComputation(uint32_t arrayID) override {
        SimTime_t latency = getArrayLatency(arrayID);   // 得到阵列延迟，比如返回1
        ArrayEvent* ev = new ArrayEvent(arrayID);       // 新建事件，带arrayID
        selfLink->send(latency, ev);                    // 延迟latency后将事件ev发送给自己
        //把事件通过selfLink（SST框架的本地自环连接）发送，延迟latency后会回到当前组件。
        //这就是模拟“异步/延迟计算”，即MVM计算不是立刻完成，而是等一段时间（比如硬件计算延迟）。
    }


    /*
    1.beginComputation：“我要让阵列2开始算，硬件要1个cycle，先发事件出去。”
    2.SST内部延迟1个单位
    3.handleSelfEvent： “1个cycle到了！事件回来了，阵列2现在可以开始真正计算了！”*/

    virtual void handleSelfEvent(Event* ev) override {
        ArrayEvent* aev = static_cast<ArrayEvent*>(ev);
        uint32_t arrayID = aev->getArrayID();

        compute(arrayID);   //对arrayID的阵列进行计算

        (*tileHandler)(ev);
    }
    //这个函数的作用是将 value 存入指定阵列 arrayID 中的矩阵 matrixData，位置是 index,
    //可以理解为arrayID是不同的二维矩阵的编号，index是二维矩阵里元素的索引
    virtual void setMatrixItem(int32_t arrayID, int32_t index, double value) override {
        matrixData[arrayID][index] = static_cast<T>(value);
    }
    //同理
    virtual void setVectorItem(int32_t arrayID, int32_t index, double value) override {
        inputVectors[arrayID][index] = static_cast<T>(value);
    }

    virtual void compute(uint32_t arrayID) override {
        auto& inputVector = inputVectors[arrayID];  //arrayID 是当前计算阵列的标识符
        auto& outputVector = outputVectors[arrayID];
        auto& matrix = matrixData[arrayID];

        // Ensure output vector is correctly sized
        outputVector.resize(outputArraySize);

        bool accumulateMode = (outputModes[arrayID] == OutputMode::Accumulate);
        if (!accumulateMode) {
            clearOutputVector(arrayID);
        }

        // Print input vector
        out.verbose(CALL_INFO, 2, 0, "MVM for array %u:\n\n", arrayID);
        for (uint32_t col = 0; col < inputArraySize; col++) {
            printValue(inputVector[col]);
        }
        out.verbose(CALL_INFO, 2, 0, "\n\n");

        // Perform matrix-vector multiplication
        for (uint32_t row = 0; row < outputArraySize; row++) {
            T dot = 0;
            for (uint32_t col = 0; col < inputArraySize; col++) {
                dot += matrix[row * inputArraySize + col] * inputVector[col];
                printValue(matrix[row * inputArraySize + col]);
            }
            if (accumulateMode) {
                outputVector[row] += dot;
            } else {
                outputVector[row] = dot;
            }
            out.verbose(CALL_INFO, 2, 0, "  ");
            printValue(outputVector[row]);
            out.verbose(CALL_INFO, 2, 0, "\n");
        }
        out.verbose(CALL_INFO, 2, 0, "\n\n");

        if (dumpEnabled) {
            dumpMvmSnapshot(arrayID, inputVector, matrix, outputVector);
        }
    }
    //这个函数是为了返回阵列的延迟时间，通常在模拟中用来表示阵列进行计算所需的时间。在这段代码中，延迟被硬编码为 1。
    virtual SimTime_t getArrayLatency(uint32_t arrayID) override {
        (void)arrayID;
        return modeledComputeCycles;
    }
    //这个虚拟函数用于将某个阵列的输出向量移动到另一个阵列的输入向量
    virtual void moveOutputToInput(uint32_t srcArrayID, uint32_t destArrayID) override {
        std::copy(outputVectors[srcArrayID].begin(), outputVectors[srcArrayID].end(), inputVectors[destArrayID].begin());
    }
    //该函数返回指定阵列的输入向量的指针
    virtual void* getInputVector(uint32_t arrayID) override {
        return static_cast<void*>(&inputVectors[arrayID]);
    }
    //该函数返回指定阵列的输出向量的指针
    virtual void* getOutputVector(uint32_t arrayID) override {
        return static_cast<void*>(&outputVectors[arrayID]);
    }

    virtual void configureOutputMode(uint32_t arrayID, uint64_t command) override {
        if (arrayID >= outputModes.size()) {
            out.verbose(CALL_INFO, 1, 0,
                        "mvmComputeArray: invalid arrayID %u for ocfg\n", arrayID);
            return;
        }
        switch (command) {
            case 0:
                outputModes[arrayID] = OutputMode::Overwrite;
                out.verbose(CALL_INFO, 3, 0,
                            "mvmComputeArray: array %u set to overwrite mode\n", arrayID);
                break;
            case 1:
                outputModes[arrayID] = OutputMode::Accumulate;
                out.verbose(CALL_INFO, 3, 0,
                            "mvmComputeArray: array %u set to accumulate mode\n", arrayID);
                break;
            case 2:
                clearOutputVector(arrayID);
                out.verbose(CALL_INFO, 3, 0,
                            "mvmComputeArray: array %u output buffer cleared\n", arrayID);
                break;
            default:
                out.verbose(CALL_INFO, 1, 0,
                            "mvmComputeArray: unknown ocfg command %" PRIu64 "\n", command);
                break;
        }
    }

protected:
    enum class OutputMode : uint8_t { Overwrite = 0, Accumulate = 1 };
    std::vector<std::vector<T>> inputVectors;
    std::vector<std::vector<T>> outputVectors;
    std::vector<std::vector<T>> matrixData;
    std::vector<OutputMode> outputModes;
    bool dumpEnabled = false;
    int coreId = -1;
    std::string dumpRootDir;
    std::string dumpCoreDir;
    uint64_t dumpSeq = 0;
    bool dumpOverwrite = true;
    std::vector<bool> dumpFileInitialized;

    void dumpMvmSnapshot(uint32_t arrayID,
                         const std::vector<T>& inputVector,
                         const std::vector<T>& matrix,
                         const std::vector<T>& outputVector) {
        if (dumpFileInitialized.size() < numArrays) {
            dumpFileInitialized.assign(numArrays, false);
        }

        std::ostringstream fpath;
        fpath << dumpCoreDir << "/mvm_array_" << arrayID << ".log";
        std::ios::openmode openMode = std::ios::out;
        if (!dumpOverwrite || dumpFileInitialized[arrayID]) {
            openMode |= std::ios::app;
        } else {
            openMode |= std::ios::trunc;
        }

        std::ofstream ofs(fpath.str(), openMode);
        if (!ofs) {
            out.verbose(CALL_INFO, 1, 0,
                        "mvmComputeArray: cannot open dump file %s\n", fpath.str().c_str());
            return;
        }
        dumpFileInitialized[arrayID] = true;

        ofs << "=== MVM Snapshot #" << dumpSeq++ << " core=" << coreId
            << " array=" << arrayID << " ===\n";
        ofs << "InputVector:";
        for (uint32_t col = 0; col < inputArraySize; col++) {
            if constexpr (std::is_same<T, int64_t>::value) {
                ofs << " " << static_cast<long long>(inputVector[col]);
            } else if constexpr (std::is_same<T, float>::value) {
                ofs << " " << inputVector[col];
            } else {
                ofs << " " << inputVector[col];
            }
        }
        ofs << "\n";

        ofs << "MatrixAndOutput:\n";
        for (uint32_t row = 0; row < outputArraySize; row++) {
            for (uint32_t col = 0; col < inputArraySize; col++) {
                const T value = matrix[row * inputArraySize + col];
                if constexpr (std::is_same<T, int64_t>::value) {
                    ofs << static_cast<long long>(value) << " ";
                } else if constexpr (std::is_same<T, float>::value) {
                    ofs << value << " ";
                } else {
                    ofs << value << " ";
                }
            }
            ofs << " | ";
            if constexpr (std::is_same<T, int64_t>::value) {
                ofs << static_cast<long long>(outputVector[row]);
            } else if constexpr (std::is_same<T, float>::value) {
                ofs << outputVector[row];
            } else {
                ofs << outputVector[row];
            }
            ofs << "\n";
        }
        ofs << "\n";
    }

    void clearOutputVector(uint32_t arrayID) {
        std::fill(outputVectors[arrayID].begin(), outputVectors[arrayID].end(), T());
    }

    void printValue(const T& value) {
        if constexpr (std::is_same<T, int64_t>::value) {
            out.verbose(CALL_INFO, 2, 0, "%" PRId64 " ", value);
        } else if constexpr (std::is_same<T, float>::value) {
            out.verbose(CALL_INFO, 2, 0, "%f ", value);
        }
    }
};

} // namespace Golem
} // namespace SST

#endif 
