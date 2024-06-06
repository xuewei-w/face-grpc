#ifndef PROJECT_SRC_UTILS_HPP_
#define PROJECT_SRC_UTILS_HPP_

#include <string>

#include <NvInfer.h>
#include <opencv2/core.hpp>

#include "engine.hpp"

/**
 * @brief Inferece result format for face detection
 */
struct FaceDetectionResult {
  /**
   * @brief Bounding box
   */
  std::vector<cv::Rect2d> bbox;
  /**
   * @brief Confidence score
   */
  std::vector<float> score;
  /**
   * @brief Facial keypoints
   */
  std::vector<cv::Mat> landmark;
};

/**
 * @brief Create TensorRT inference engine for face detection
 * @param engineFilePath
 * Path to TensorRT engine file
 * @param batchSize
 * Supported batch size by TensorRT engine file
 * @param logLevel
 * Log level for TensorRT logger,
 * refer to nvinfer1::ILogger::Severity
 * @return
 * Pointer to InferEngine
 */
InferEngine *createFaceDetector(const std::string &engineFilePath,
                                int batchSize = 1,
                                nvinfer1::ILogger::Severity logLevel =
                                    nvinfer1::ILogger::Severity::kWARNING);

/**
 * @brief Perform face detection
 * @param engine
 * Pointer to InferEngine
 * @param image
 * Input image
 * @param slide
 * Use sliding window to maintain original resolution,
 * otherwise resize image to fit input size,
 * enable this option may increase inference time
 * @param nmsThreshold
 * Non-maximum suppression threshold
 * @param scoreThreshold
 * Confidence score threshold
 * @param keepBeforeNMS
 * Number of bounding boxes to keep before non-maximum suppression
 * @param topK
 * Number of bounding boxes to keep finally
 * @return
 * Face detection result
 */
FaceDetectionResult faceDetection(InferEngine *engine, const cv::Mat &image,
                                  bool slide = false, float nmsThreshold = .5F,
                                  float scoreThreshold = .5F,
                                  int keepBeforeNMS = 1000, int topK = 100);

#endif