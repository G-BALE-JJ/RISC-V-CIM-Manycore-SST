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

#ifndef _COMPUTEARRAY_H
#define _COMPUTEARRAY_H

#include <sst/core/component.h>
#include <sst/core/subcomponent.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/timeConverter.h>
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <deque>
#include <functional>
#include <unordered_map>
#include <vector>

namespace SST {
namespace Golem {

class ArrayEvent : public SST::Event {
public:
    ArrayEvent() {} // For serialization only
    ArrayEvent(uint32_t array) : SST::Event(), arrayID(array) {}

    uint32_t getArrayID() { return arrayID; };

protected:
    uint32_t arrayID;

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(arrayID);
    }
    ImplementSerializable(SST::Golem::ArrayEvent);
};

class ArrayBufferEvent : public SST::Event {
public:
    ArrayBufferEvent() = default;
    explicit ArrayBufferEvent(uint64_t requestId) : requestId_(requestId) {}

    uint64_t requestId() const { return requestId_; }

protected:
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(requestId_);
    }
    ImplementSerializable(SST::Golem::ArrayBufferEvent);

private:
    uint64_t requestId_ = 0;
};

class ComputeArray : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(
        SST::Golem::ComputeArray,
        TimeConverter*,
        Event::HandlerBase*
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"verbose", "Verbosity of outputs", "1"},
        {"core_id", "Owning core ID used in array-buffer statistics", "-1"},
        {"clock", "Array clock frequency", "1GHz"},
        {"arrayLatency", "Latency of array operation", "100ns"},
        {"modeledComputeCycles", "Modeled compute latency in array clock cycles", "1"},
        {"numArrays", "Number of arrays", "1"},
        {"arrayInputSize", "Input size of arrays", "1"},
        {"arrayOutputSize", "Output size of arrays", "1"},
        {"inputOperandSize", "Size of input operands", "1"},
        {"outputOperandSize", "Size of output operands", "1"},
        {"arrayBufferBaseLatencyCycles", "Base latency of an array-buffer transfer", "1"},
        {"arrayBufferBytesPerCycle", "Bytes transferred per array-buffer port per cycle", "64"},
        {"arrayBufferPorts", "Number of array-buffer transfer ports", "1"},
        {"arrayBufferQueueDepth", "Maximum queued plus active array-buffer transfers", "64"},
    )

    ComputeArray(ComponentId_t id, Params& params,
                 TimeConverter* tc,
                 Event::HandlerBase* handler)
        : SubComponent(id), out("", params.find<int>("verbose", 1), 0, Output::STDOUT),
          tileHandler(handler) {
        // Initialize parameters
        arrayClock = params.find<UnitAlgebra>("clock", "1GHz");
        arrayLatency = params.find<UnitAlgebra>("arrayLatency", "100ns");
        clockTC = getTimeConverter(arrayClock);
        latencyTC = getTimeConverter(arrayLatency);
        modeledComputeCycles = params.find<uint64_t>("modeledComputeCycles", 1);
        if (modeledComputeCycles == 0) {
            modeledComputeCycles = 1;
        }

        numArrays = params.find<uint64_t>("numArrays", 1);
        inputArraySize = params.find<uint64_t>("arrayInputSize", 1);
        outputArraySize = params.find<uint64_t>("arrayOutputSize", 1);
        inputOperandSize = params.find<uint64_t>("inputOperandSize", 1);
        outputOperandSize = params.find<uint64_t>("outputOperandSize", 1);
        arrayCoreId_ = params.find<int>("core_id", -1);
        arrayBufferBaseLatencyCycles_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayBufferBaseLatencyCycles", 1), 1);
        arrayBufferBytesPerCycle_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayBufferBytesPerCycle", 64), 1);
        arrayBufferPorts_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayBufferPorts", 1), 1);
        arrayBufferQueueDepth_ =
            std::max<uint64_t>(params.find<uint64_t>("arrayBufferQueueDepth", 64), 1);
        bufferLink_ = configureSelfLink(
            "BufferSelf", *tc,
            new Event::Handler2<ComputeArray, &ComputeArray::handleBufferEvent>(this));
        bufferLink_->setDefaultTimeBase(*clockTC);
    }

    virtual ~ComputeArray() {}

    virtual void init(unsigned int phase) override {}
    virtual void setup() override {}
    virtual void finish() override {
        out.output(
            "GOLEM_ARRAY_BUFFER_STATS core=%d requests=%" PRIu64
            " bytes=%" PRIu64 " rejected=%" PRIu64
            " high_water=%" PRIu64 " transfer_cycles=%" PRIu64 "\n",
            arrayCoreId_, arrayBufferRequests_, arrayBufferBytes_, arrayBufferRejected_,
            arrayBufferHighWater_, arrayBufferTransferCycles_);
    }
    virtual void emergencyShutdown() override {}

    virtual void beginComputation(uint32_t arrayID) = 0;
    virtual void handleSelfEvent(Event* ev) = 0;
    virtual SimTime_t getArrayLatency(uint32_t arrayID) = 0;
    virtual void setMatrixItem(int32_t arrayID, int32_t index, double value) = 0;
    virtual void setVectorItem(int32_t arrayID, int32_t index, double value) = 0;
    virtual void compute(uint32_t arrayID) = 0;
    virtual void moveOutputToInput(uint32_t srcArrayID, uint32_t destArrayID) = 0;
    virtual void* getInputVector(uint32_t arrayID) = 0;
    virtual void* getOutputVector(uint32_t arrayID) = 0;
    using BufferCallback = std::function<void(bool, uint64_t)>;
    using BufferReadCallback =
        std::function<void(bool, uint64_t, const std::vector<double>&)>;
    using BufferByteReadCallback =
        std::function<void(bool, uint64_t, const std::vector<uint8_t>&)>;
    virtual bool programMatrixAsync(uint32_t arrayID,
                                    const std::vector<double>& matrix,
                                    size_t elemBytes,
                                    uint64_t tag,
                                    BufferCallback callback) = 0;
    virtual bool programInputAsync(uint32_t arrayID,
                                   const std::vector<double>& input,
                                   size_t elemBytes,
                                   uint64_t tag,
                                   BufferCallback callback) = 0;
    virtual bool programOperandsAsync(uint32_t arrayID,
                                      const std::vector<double>& matrix,
                                      const std::vector<double>& input,
                                      size_t elemBytes,
                                      uint64_t tag,
                                      BufferCallback callback) = 0;
    virtual bool readOutputAsync(uint32_t arrayID, size_t elemBytes,
                                 uint64_t tag, BufferReadCallback callback) = 0;
    virtual bool readOutputBytesAsync(uint32_t arrayID, size_t elemBytes,
                                      uint64_t tag,
                                      BufferByteReadCallback callback) = 0;
    virtual bool writeOutputAsync(uint32_t arrayID,
                                  const std::vector<double>& output,
                                  size_t elemBytes,
                                  uint64_t tag,
                                  BufferCallback callback) = 0;
    // Optional override: configure output buffer behavior (e.g., accumulate vs overwrite)
    virtual void configureOutputMode(uint32_t, uint64_t) {}

protected:
    bool enqueueBufferTransfer(size_t bytes, uint64_t tag,
                               std::function<void()> completion) {
        if (!completion || bufferRequests_.size() >= arrayBufferQueueDepth_) {
            arrayBufferRejected_ += 1;
            return false;
        }
        const uint64_t requestId = nextBufferRequestId_++;
        bufferRequests_.emplace(
            requestId, BufferRequest{requestId, tag, bytes, std::move(completion)});
        bufferQueue_.push_back(requestId);
        arrayBufferRequests_ += 1;
        arrayBufferBytes_ += bytes;
        arrayBufferHighWater_ =
            std::max<uint64_t>(arrayBufferHighWater_, bufferRequests_.size());
        tryIssueBufferTransfers();
        return true;
    }

    SST::Output out;
    SST::Link* selfLink = nullptr;
    SST::Event::HandlerBase* tileHandler = nullptr;
    UnitAlgebra arrayClock;
    UnitAlgebra arrayLatency;
    TimeConverter* clockTC = nullptr;
    TimeConverter* latencyTC = nullptr;
    uint64_t modeledComputeCycles = 1;

    uint64_t numArrays;
    uint64_t inputArraySize;
    uint64_t outputArraySize;
    uint64_t inputOperandSize;
    uint64_t outputOperandSize;

private:
    struct BufferRequest {
        uint64_t requestId = 0;
        uint64_t tag = 0;
        size_t bytes = 0;
        std::function<void()> completion;
    };

    void tryIssueBufferTransfers() {
        while (arrayBufferInFlight_ < arrayBufferPorts_ && !bufferQueue_.empty()) {
            const uint64_t requestId = bufferQueue_.front();
            bufferQueue_.pop_front();
            const auto it = bufferRequests_.find(requestId);
            if (it == bufferRequests_.end()) {
                continue;
            }
            const uint64_t transferCycles = arrayBufferBaseLatencyCycles_ +
                (it->second.bytes + arrayBufferBytesPerCycle_ - 1) /
                    arrayBufferBytesPerCycle_;
            arrayBufferTransferCycles_ += transferCycles;
            arrayBufferInFlight_ += 1;
            bufferLink_->send(transferCycles, new ArrayBufferEvent(requestId));
        }
    }

    void handleBufferEvent(Event* event) {
        auto* bufferEvent = dynamic_cast<ArrayBufferEvent*>(event);
        if (bufferEvent == nullptr) {
            delete event;
            return;
        }
        const auto it = bufferRequests_.find(bufferEvent->requestId());
        if (it != bufferRequests_.end()) {
            auto completion = std::move(it->second.completion);
            bufferRequests_.erase(it);
            if (completion) {
                completion();
            }
        }
        if (arrayBufferInFlight_ > 0) {
            arrayBufferInFlight_ -= 1;
        }
        delete bufferEvent;
        tryIssueBufferTransfers();
    }

    SST::Link* bufferLink_ = nullptr;
    uint64_t arrayBufferBaseLatencyCycles_ = 1;
    uint64_t arrayBufferBytesPerCycle_ = 64;
    uint64_t arrayBufferPorts_ = 1;
    uint64_t arrayBufferQueueDepth_ = 64;
    uint64_t arrayBufferInFlight_ = 0;
    uint64_t nextBufferRequestId_ = 1;
    uint64_t arrayBufferRequests_ = 0;
    uint64_t arrayBufferBytes_ = 0;
    uint64_t arrayBufferRejected_ = 0;
    uint64_t arrayBufferHighWater_ = 0;
    uint64_t arrayBufferTransferCycles_ = 0;
    int arrayCoreId_ = -1;
    std::deque<uint64_t> bufferQueue_;
    std::unordered_map<uint64_t, BufferRequest> bufferRequests_;
};

} // namespace Golem
} // namespace SST

#endif /* _COMPUTEARRAY_H */
