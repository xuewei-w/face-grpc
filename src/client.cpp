#include <cstdio>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <grpcpp/grpcpp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "service.grpc.pb.h"

int main(int argc, char **argv) {
  auto stub = FaceDetectionService::NewStub(
      grpc::CreateChannel(argv[1], grpc::InsecureChannelCredentials()));
  grpc::ClientContext context;
  FaceDetectionRequest request;
  std::ifstream imageFile(argv[2], std::ios::binary);
  std::stringstream buffer;
  buffer << imageFile.rdbuf();
  request.set_image(buffer.str());
  FaceDetectionResponse response;
  auto start = std::chrono::steady_clock::now();
  stub->serve(&context, request, &response);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Inference used "
            << std::chrono::duration<double, std::milli>(end - start).count()
            << "ms, detected " << response.bbox_size() << " faces" << std::endl;
  auto image = cv::imread(argv[2]);
  std::unique_ptr<char[]> scoreText(new char[8]);
  for (int i = 0; i < response.bbox_size(); i++) {
    auto rawBbox = response.bbox(i);
    cv::Rect bbox(rawBbox.x() * image.cols, rawBbox.y() * image.rows,
                  rawBbox.width() * image.cols, rawBbox.height() * image.rows);
    cv::rectangle(image, bbox, cv::Scalar(0, 255, 0));
    cv::rectangle(image, cv::Point(bbox.x, bbox.y - 8),
                  cv::Point(bbox.x + bbox.width, bbox.y), cv::Scalar(0, 255, 0),
                  cv::FILLED);
    std::snprintf(scoreText.get(), 8, "%.2f", response.score(i) * 100);
    cv::putText(image, scoreText.get(), cv::Point(bbox.x, bbox.y - 1),
                cv::FONT_HERSHEY_SIMPLEX, .3, cv::Scalar(255, 0, 0));
    auto landmark = response.landmark(i);
    for (int j = 0; j < landmark.point_size(); j++) {
      auto point = landmark.point(j);
      cv::circle(image,
                 cv::Point(point.x() * image.cols, point.y() * image.rows), 3,
                 cv::Scalar(0, 0, 255), -1);
    }
  }
  cv::imwrite(argv[3], image);
  return 0;
}