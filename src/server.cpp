#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <opencv2/imgcodecs.hpp>

#include "service.grpc.pb.h"

#include "engine.hpp"
#include "utils.hpp"

class FaceDetectionServiceImpl final : public FaceDetectionService::Service {
public:
  explicit FaceDetectionServiceImpl(const std::string &engineFilePath)
      : engine(createFaceDetector(engineFilePath)) {}

  grpc::Status serve(grpc::ServerContext *context,
                     const FaceDetectionRequest *request,
                     FaceDetectionResponse *response) override {
    auto start = std::chrono::steady_clock::now();
    auto image = cv::imdecode(
        std::vector<uchar>(request->image().begin(), request->image().end()),
        cv::IMREAD_COLOR);
    auto result = faceDetection(engine.get(), image, true, .1, .9);
    for (auto &bboxItem : result.bbox) {
      auto bbox = response->add_bbox();
      bbox->set_x(bboxItem.x);
      bbox->set_y(bboxItem.y);
      bbox->set_width(bboxItem.width);
      bbox->set_height(bboxItem.height);
    }
    for (auto scoreItem : result.score) {
      response->add_score(scoreItem);
    }
    for (auto &landmarkItem : result.landmark) {
      auto landmark = response->add_landmark();
      for (int i = 0; i < 5; i++) {
        auto point = landmark->add_point();
        point->set_x(landmarkItem.at<double>(i, 0));
        point->set_y(landmarkItem.at<double>(i, 1));
      }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Inference used "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << "ms, detected " << result.bbox.size() << " faces" << std::endl;
    return grpc::Status::OK;
  }

private:
  std::unique_ptr<InferEngine> engine;
};

void runServer(const std::string &serverAddress,
               const std::string &engineFilePath) {
  FaceDetectionServiceImpl service(engineFilePath);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << serverAddress << std::endl;
  server->Wait();
}

int main(int argc, char **argv) {
  runServer(argv[1], argv[2]);
  return 0;
}
