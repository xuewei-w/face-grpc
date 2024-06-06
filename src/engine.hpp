#ifndef PROJECT_SRC_ENGINE_HPP_
#define PROJECT_SRC_ENGINE_HPP_

#include <cstdint>
#include <cstdlib>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>
#include <opencv2/core.hpp>

/**
 * @class InferEngine
 * @brief TensorRT inference engine
 */
class InferEngine {
public:
  InferEngine() = delete;
  /**
   * @brief Constructor
   * @param engineData
   * TensorRT engine file data
   * @param engineSize
   * TensorRT engine file size
   * @param inputInfo
   * Input name and size(without batch size),
   * e.g. std::unordered_map{{"input", {3, 100, 100}}
   * @param outputInfo
   * Output name and size(without batch size),
   * e.g. std::unordered_map{{"bbox", {100, 4}}
   * @param batchSize
   * Supported batch size by TensorRT engine file
   * @param logLevel
   * Log level for TensorRT logger,
   * refer to nvinfer1::ILogger::Severity
   */
  InferEngine(
      const void *engineData, std::size_t engineSize,
      const std::unordered_map<std::string, std::vector<int>> &inputInfo,
      const std::unordered_map<std::string, std::vector<int>> &outputInfo,
      int batchSize = 1,
      nvinfer1::ILogger::Severity logLevel =
          nvinfer1::ILogger::Severity::kWARNING);
  /**
   * @brief Destructor
   */
  ~InferEngine();

  /**
   * @brief Get tensor data type by name
   * @param name
   * Tensor name
   * @return
   * Tensor data type,
   * refer to nvinfer1::DataType
   */
  inline nvinfer1::DataType getTensorDataType(const std::string &name);

  /**
   * @brief Inference input data
   * @param input
   * Input name and data,
   * e.g. std::unordered_map{{"input", cv::Mat(size: 1 x 3 x 100 x 100)}}
   * @return
   * Output name and data,
   * e.g. std::unordered_map{{"bbox", cv::Mat(size: 1 x 100 x 4)}}
   */
  std::unordered_map<std::string, cv::Mat>
  infer(const std::unordered_map<std::string, cv::Mat> &input);

private:
  struct BufferInfo {
    void *addr;
    std::uint64_t size;
    std::vector<int> sizes;
  };

  int batchSize;

  std::unique_ptr<nvinfer1::ILogger> logger;
  std::unique_ptr<nvinfer1::IGpuAllocator> allocator;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  std::unordered_map<std::string, BufferInfo> inputBuffer;
  std::unordered_map<std::string, BufferInfo> outputBuffer;
};

#endif
