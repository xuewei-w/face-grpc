#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include "engine.hpp"

std::unique_ptr<char[]> getTimeStr() {
  auto now = std::time(nullptr);
  char *cStr = std::ctime(&now);
  auto len = std::strlen(cStr);
  std::unique_ptr<char[]> str(new char[len]);
  std::memcpy(str.get(), cStr, len - 1);
  str[len - 1] = '\0';
  return str;
}

std::unique_ptr<char[]> getSizeStr(std::uint64_t size) {
  std::unique_ptr<char[]> str(new char[16]);
  if (size < 1UL << 10) {
    snprintf(str.get(), 16, "%lu Bytes", size);
  } else if (size < 1UL << 20) {
    snprintf(str.get(), 16, "%.2f KiB", (double)size / (1UL << 10));
  } else if (size < 1UL << 30) {
    snprintf(str.get(), 16, "%.2f MiB", (double)size / (1UL << 20));
  } else {
    snprintf(str.get(), 16, "%.2f GiB", (double)size / (1UL << 30));
  }
  return str;
}

class Logger : public nvinfer1::ILogger {
public:
  Logger() : logLevel(Severity::kWARNING) {}
  Logger(Severity severity) : logLevel(severity) {}
  void log(Severity severity, const char *msg) noexcept override {
    static std::string SEVERITY[] = {"Internal Error", "Error", "Warning",
                                     "Info", "Verbose"};
    static std::ostream *OUT = nullptr;
    if (severity <= logLevel) {
      if (severity <= Severity::kWARNING) {
        OUT = &std::cerr;
      } else {
        OUT = &std::cout;
      }
      *OUT << "[" << getTimeStr().get() << "] ["
           << SEVERITY[(std::int32_t)severity] << "] " << msg << std::endl;
    }
  }

private:
  Severity logLevel;
};

class GpuAllocator : public nvinfer1::IGpuAsyncAllocator {
public:
  GpuAllocator() = delete;
  GpuAllocator(nvinfer1::ILogger &logger) : allocatedSize(0), logger(logger) {}
  ~GpuAllocator() {
    while (!memoryManager.empty()) {
      deallocateAsync(memoryManager.begin()->first, nullptr);
    }
  }

  void *allocateAsync(const std::uint64_t size, const std::uint64_t alignment,
                      const nvinfer1::AllocatorFlags flags,
                      cudaStream_t stream) noexcept override {
    void *memory = nullptr;
    cudaMalloc(&memory, size);
    if (memory) {
      memoryManager[memory] = size;
      logger.log(nvinfer1::ILogger::Severity::kINFO,
                 getLogMsg(true, memory).get());
    }
    return memory;
  }

  bool deallocateAsync(void *const memory,
                       cudaStream_t stream) noexcept override {
    cudaError_t status;
    status = cudaFree(memory);
    if (memoryManager.find(memory) != memoryManager.end()) {
      logger.log(nvinfer1::ILogger::Severity::kINFO,
                 getLogMsg(false, memory).get());
      memoryManager.erase(memory);
    }
    return status == cudaSuccess;
  }

private:
  std::uint64_t allocatedSize;
  std::unordered_map<void *, std::uint64_t> memoryManager;

  nvinfer1::ILogger &logger;

  std::unique_ptr<char[]> getLogMsg(bool allocate, void *memory) {
    std::uint64_t size = memoryManager[memory];
    allocatedSize += allocate ? size : -size;
    std::unique_ptr<char[]> msg(new char[128]);
    snprintf(msg.get(), 128,
             "%s %s at %p, "
             "total %s in use",
             allocate ? "Allocated" : "Deallocated", getSizeStr(size).get(),
             memory, getSizeStr(allocatedSize).get());
    return msg;
  }
};

std::uint64_t sizeofDataType(nvinfer1::DataType dataType) {
  switch (dataType) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
    return 1;
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kBOOL:
    return 1;
  case nvinfer1::DataType::kUINT8:
    return 1;
  case nvinfer1::DataType::kFP8:
    return 1;
  case nvinfer1::DataType::kBF16:
    return 2;
  case nvinfer1::DataType::kINT64:
    return 8;
  case nvinfer1::DataType::kINT4:
    return 1;
  }
  return 4;
}

int getCvDepth(nvinfer1::DataType dataType) {
  switch (dataType) {
  case nvinfer1::DataType::kFLOAT:
    return CV_32F;
  case nvinfer1::DataType::kHALF:
    return CV_16F;
  case nvinfer1::DataType::kINT8:
    return CV_8S;
  case nvinfer1::DataType::kINT32:
    return CV_32S;
  case nvinfer1::DataType::kBOOL:
    return CV_8U;
  case nvinfer1::DataType::kUINT8:
    return CV_8U;
  case nvinfer1::DataType::kFP8:
    return CV_8U;
  case nvinfer1::DataType::kBF16:
    return CV_16U;
  case nvinfer1::DataType::kINT64:
    return CV_64F;
  case nvinfer1::DataType::kINT4:
    return CV_8U;
  }
  return CV_32F;
}

InferEngine::InferEngine(
    const void *engineData, std::size_t engineSize,
    const std::unordered_map<std::string, std::vector<int>> &inputInfo,
    const std::unordered_map<std::string, std::vector<int>> &outputInfo,
    int batchSize, nvinfer1::ILogger::Severity logLevel)
    : batchSize(batchSize), logger(new Logger(logLevel)) {
  allocator.reset(new GpuAllocator(*logger));
  runtime.reset(nvinfer1::createInferRuntime(*logger));
  runtime->setGpuAllocator(allocator.get());
  engine.reset(runtime->deserializeCudaEngine(engineData, engineSize));
  context.reset(engine->createExecutionContext(
      nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  for (auto &nameSizes : inputInfo) {
    auto &name = nameSizes.first;
    auto &sizes = nameSizes.second;
    auto size = sizeofDataType(engine->getTensorDataType(name.c_str()));
    nvinfer1::Dims dims;
    dims.nbDims = sizes.size() + 1;
    auto d = dims.d;
    *d++ = batchSize;
    for (int sizeItem : sizes) {
      size *= sizeItem;
      *d++ = sizeItem;
    }
    context->setInputShape(name.c_str(), dims);
    context->setDeviceMemory(allocator->allocateAsync(
        context->updateDeviceMemorySizeForShapes(), 0UL, 0U, nullptr));
    inputBuffer[name] = {
        allocator->allocateAsync(batchSize * size, 0UL, 0U, nullptr), size,
        sizes};
    context->setInputTensorAddress(name.c_str(), inputBuffer[name].addr);
  }
  for (auto &nameSizes : outputInfo) {
    auto &name = nameSizes.first;
    auto &sizes = nameSizes.second;
    auto size = sizeofDataType(engine->getTensorDataType(name.c_str()));
    for (int sizeItem : sizes) {
      size *= sizeItem;
    }
    outputBuffer[name] = {
        allocator->allocateAsync(batchSize * size, 0UL, 0U, nullptr), size,
        sizes};
    context->setOutputTensorAddress(name.c_str(), outputBuffer[name].addr);
  }
}

InferEngine::~InferEngine() {
  context.reset();
  engine.reset();
  runtime.reset();
  allocator.reset();
}

inline nvinfer1::DataType
InferEngine::getTensorDataType(const std::string &name) {
  return engine->getTensorDataType(name.c_str());
}

std::unordered_map<std::string, cv::Mat>
InferEngine::infer(const std::unordered_map<std::string, cv::Mat> &input) {
  std::unordered_map<std::string, cv::Mat> output;
  int totalBatchSize = input.begin()->second.size[0];
  int epochs = std::ceil((double)totalBatchSize / batchSize);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  for (int epoch = 0; epoch < epochs; epoch++) {
    int curBatchSize = std::min(batchSize, totalBatchSize - epoch * batchSize);
    for (auto &nameMat : input) {
      auto &name = nameMat.first;
      auto &buffer = inputBuffer[name];
      cudaMemcpy(buffer.addr,
                 input.at(name).data + epoch * batchSize * buffer.size,
                 curBatchSize * buffer.size, cudaMemcpyHostToDevice);
    }
    context->enqueueV3(stream);
    for (auto &nameBuffer : outputBuffer) {
      auto &name = nameBuffer.first;
      auto &buffer = nameBuffer.second;
      if (output.find(name) == output.end()) {
        std::vector<int> sizes(buffer.sizes.size() + 1);
        sizes[0] = totalBatchSize;
        auto sizesIt = sizes.begin() + 1;
        for (int sizeItem : buffer.sizes) {
          *sizesIt++ = sizeItem;
        }
        output[name] = cv::Mat(sizes, getCvDepth(getTensorDataType(name)));
      }
      cudaMemcpy(output[name].data + epoch * batchSize * buffer.size,
                 buffer.addr, curBatchSize * buffer.size,
                 cudaMemcpyDeviceToHost);
    }
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return output;
}
