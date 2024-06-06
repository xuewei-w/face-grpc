#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "engine.hpp"
#include "utils.hpp"

namespace faceDetectionImpl {
static const int MIN_SIZES[][2] = {{16, 32}, {64, 128}, {256, 512}};
static const int STEPS[] = {8, 16, 32};
static const double VAR[] = {.1, .2};
static const int INPUT_SIZE[] = {640, 640};
static const int OUTPUT_SIZE =
    ((int)std::ceil((double)INPUT_SIZE[0] / STEPS[0]) *
         (int)std::ceil((double)INPUT_SIZE[1] / STEPS[0]) +
     (int)std::ceil((double)INPUT_SIZE[0] / STEPS[1]) *
         (int)std::ceil((double)INPUT_SIZE[1] / STEPS[1]) +
     (int)std::ceil((double)INPUT_SIZE[0] / STEPS[2]) *
         (int)std::ceil((double)INPUT_SIZE[1] / STEPS[2])) *
    2;

static cv::Mat prior;
static bool priorInitialized = false;

void initPrior() {
  if (priorInitialized) {
    return;
  }
  prior.create(OUTPUT_SIZE, 4, CV_64F);
  double *priorData = (double *)prior.data;
  for (int k = 0; k < 3; k++) {
    double step = STEPS[k];
    for (int i = 0; i < std::ceil(INPUT_SIZE[0] / step); i++) {
      for (int j = 0; j < std::ceil(INPUT_SIZE[1] / step); j++) {
        for (int l = 0; l < 2; l++) {
          double minSize = MIN_SIZES[k][l];
          *priorData++ = (j + .5) * step / INPUT_SIZE[1];
          *priorData++ = (i + .5) * step / INPUT_SIZE[0];
          *priorData++ = minSize / INPUT_SIZE[1];
          *priorData++ = minSize / INPUT_SIZE[0];
        }
      }
    }
  }
  priorInitialized = true;
}

inline void decodeBbox(cv::Rect2d &bboxItem, const float *rawBboxItem,
                       const double *priorItem) {
  bboxItem.width = std::exp(rawBboxItem[2] * VAR[1]) * priorItem[2];
  bboxItem.height = std::exp(rawBboxItem[3] * VAR[1]) * priorItem[3];
  bboxItem.x = rawBboxItem[0] * VAR[0] * priorItem[2] + priorItem[0] -
               bboxItem.width / 2.;
  bboxItem.y = rawBboxItem[1] * VAR[0] * priorItem[3] + priorItem[1] -
               bboxItem.height / 2.;
}

inline void decodeLandmark(cv::Mat &landmarkItem, const float *rawLandmarkItem,
                           const double *priorItem) {
  landmarkItem.create(5, 2, CV_64F);
  double *landmarkItemData = (double *)landmarkItem.data;
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[2] + priorItem[0];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[3] + priorItem[1];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[2] + priorItem[0];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[3] + priorItem[1];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[2] + priorItem[0];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[3] + priorItem[1];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[2] + priorItem[0];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[3] + priorItem[1];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[2] + priorItem[0];
  *landmarkItemData++ =
      *rawLandmarkItem++ * VAR[0] * priorItem[3] + priorItem[1];
}
} // namespace faceDetectionImpl

InferEngine *createFaceDetector(const std::string &engineFilePath,
                                int batchSize,
                                nvinfer1::ILogger::Severity logLevel) {
  std::ifstream engineFile(engineFilePath, std::ios::binary);
  engineFile.seekg(0, std::ifstream::end);
  auto engineFileSize = engineFile.tellg();
  engineFile.seekg(0, std::ifstream::beg);
  std::unique_ptr<char[]> engineData(new char[engineFileSize]);
  engineFile.read(engineData.get(), engineFileSize);
  return new InferEngine(engineData.get(), engineFileSize,
                         {{"input",
                           {3, faceDetectionImpl::INPUT_SIZE[0],
                            faceDetectionImpl::INPUT_SIZE[1]}}},
                         {{"bbox", {faceDetectionImpl::OUTPUT_SIZE, 4}},
                          {"score", {faceDetectionImpl::OUTPUT_SIZE, 2}},
                          {"landmark", {faceDetectionImpl::OUTPUT_SIZE, 10}}},
                         batchSize, logLevel);
}

FaceDetectionResult faceDetection(InferEngine *engine, const cv::Mat &image,
                                  bool slide, float nmsThreshold,
                                  float scoreThreshold, int keepBeforeNMS,
                                  int topK) {
  cv::Mat inputData;
  int rows = 0, cols = 0;
  if (slide) {
    int halfWidth = std::ceil((double)faceDetectionImpl::INPUT_SIZE[1] / 2),
        halfHeight = std::ceil((double)faceDetectionImpl::INPUT_SIZE[0] / 2);
    rows = (int)std::ceil((double)std::max(image.rows, halfHeight * 2) /
                          halfHeight) -
           1;
    cols = (int)std::ceil((double)std::max(image.cols, halfWidth * 2) /
                          halfWidth) -
           1;
    std::vector<cv::Mat> windows(rows * cols + 1);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        windows[i * cols + j] = image(
            cv::Rect(j * image.cols / (cols + 1), i * image.rows / (rows + 1),
                     2 * image.cols / (cols + 1), 2 * image.rows / (rows + 1)));
      }
    }
    windows[rows * cols] = image;
    inputData = cv::dnn::blobFromImages(
        windows, 1.,
        cv::Size(faceDetectionImpl::INPUT_SIZE[0],
                 faceDetectionImpl::INPUT_SIZE[1]),
        cv::Scalar(104., 117., 123.), false, false, CV_32F);
  } else {
    inputData = cv::dnn::blobFromImage(
        image, 1.,
        cv::Size(faceDetectionImpl::INPUT_SIZE[0],
                 faceDetectionImpl::INPUT_SIZE[1]),
        cv::Scalar(104., 117., 123.), false, false, CV_32F);
  }
  auto rawOutput = engine->infer({{"input", inputData}});
  faceDetectionImpl::initPrior();
  std::vector<cv::Rect2d> bbox(inputData.size[0] * topK);
  std::vector<float> score(inputData.size[0] * topK);
  std::vector<cv::Mat> landmark(inputData.size[0] * topK);
  std::size_t curIndex = 0;
  for (int batch = 0; batch < inputData.size[0]; batch++) {
    int row = 0, col = 0;
    if (slide) {
      row = batch / cols;
      col = batch % cols;
    }
    float *rawBboxData = rawOutput["bbox"].ptr<float>(batch);
    float *rawScoreData = rawOutput["score"].ptr<float>(batch);
    float *rawLandmarkData = rawOutput["landmark"].ptr<float>(batch);
    std::unique_ptr<std::pair<float, int>[]> scoreIndex(
        new std::pair<float, int>[faceDetectionImpl::OUTPUT_SIZE]);
    auto scoreIndexPtr = scoreIndex.get();
    float *rawScoreDataPtr = rawScoreData;
    for (int i = 0; i < faceDetectionImpl::OUTPUT_SIZE; i++) {
      scoreIndexPtr->first = *(++rawScoreDataPtr)++;
      (scoreIndexPtr++)->second = i;
    }
    std::sort(
        scoreIndex.get(), scoreIndexPtr,
        [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
          if (a.first == b.first) {
            return a.second < b.second;
          }
          return a.first > b.first;
        });
    std::size_t sizeBeforeNMS =
        std::min(keepBeforeNMS, faceDetectionImpl::OUTPUT_SIZE);
    std::vector<int> indicesBeforeNMS(sizeBeforeNMS);
    std::vector<cv::Rect2d> bboxBeforeNMS(sizeBeforeNMS);
    auto bboxBeforeNMSIt = bboxBeforeNMS.begin();
    std::vector<float> scoreBeforeNMS(sizeBeforeNMS);
    auto scoreBeforeNMSIt = scoreBeforeNMS.begin();
    scoreIndexPtr = scoreIndex.get();
    for (int &indexBeforeNMS : indicesBeforeNMS) {
      indexBeforeNMS = scoreIndexPtr->second;
      faceDetectionImpl::decodeBbox(
          *bboxBeforeNMSIt++, rawBboxData + indexBeforeNMS * 4,
          faceDetectionImpl::prior.ptr<double>(indexBeforeNMS));
      *scoreBeforeNMSIt++ = (scoreIndexPtr++)->first;
    }
    std::vector<int> indicesAfterNMS;
    cv::dnn::NMSBoxes(bboxBeforeNMS, scoreBeforeNMS, scoreThreshold,
                      nmsThreshold, indicesAfterNMS, 1.F, topK);
    for (int indexAfterNMS : indicesAfterNMS) {
      bbox[curIndex] = bboxBeforeNMS[indexAfterNMS];
      faceDetectionImpl::decodeLandmark(
          landmark[curIndex],
          rawLandmarkData + indicesBeforeNMS[indexAfterNMS] * 10,
          faceDetectionImpl::prior.ptr<double>(
              indicesBeforeNMS[indexAfterNMS]));
      if (batch < rows * cols) {
        auto &bboxItem = bbox[curIndex];
        bboxItem.width *= 2. / (cols + 1);
        bboxItem.height *= 2. / (rows + 1);
        bboxItem.x *= 2. / (cols + 1);
        bboxItem.x += (double)col / (cols + 1);
        bboxItem.y *= 2. / (rows + 1);
        bboxItem.y += (double)row / (rows + 1);
        double *landmarkData = (double *)landmark[curIndex].data;
        *landmarkData *= 2. / (cols + 1);
        *landmarkData++ += (double)col / (cols + 1);
        *landmarkData *= 2. / (rows + 1);
        *landmarkData++ += (double)row / (rows + 1);
        *landmarkData *= 2. / (cols + 1);
        *landmarkData++ += (double)col / (cols + 1);
        *landmarkData *= 2. / (rows + 1);
        *landmarkData++ += (double)row / (rows + 1);
        *landmarkData *= 2. / (cols + 1);
        *landmarkData++ += (double)col / (cols + 1);
        *landmarkData *= 2. / (rows + 1);
        *landmarkData++ += (double)row / (rows + 1);
        *landmarkData *= 2. / (cols + 1);
        *landmarkData++ += (double)col / (cols + 1);
        *landmarkData *= 2. / (rows + 1);
        *landmarkData++ += (double)row / (rows + 1);
        *landmarkData *= 2. / (cols + 1);
        *landmarkData++ += (double)col / (cols + 1);
        *landmarkData *= 2. / (rows + 1);
        *landmarkData++ += (double)row / (rows + 1);
      }
      score[curIndex++] = scoreBeforeNMS[indexAfterNMS];
    }
  }
  bbox.resize(curIndex);
  score.resize(curIndex);
  landmark.resize(curIndex);
  std::vector<int> indices;
  cv::dnn::NMSBoxes(bbox, score, scoreThreshold, nmsThreshold, indices, 1.F,
                    topK);
  FaceDetectionResult result = {std::vector<cv::Rect2d>(indices.size()),
                                std::vector<float>(indices.size()),
                                std::vector<cv::Mat>(indices.size())};
  for (int i = 0; i < indices.size(); i++) {
    result.bbox[i] = bbox[indices[i]];
    result.score[i] = score[indices[i]];
    result.landmark[i] = landmark[indices[i]];
  }
  return result;
}
