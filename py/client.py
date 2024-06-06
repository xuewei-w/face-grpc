import sys
import time

import cv2
import grpc
from service_pb2 import FaceDetectionRequest
from service_pb2_grpc import FaceDetectionServiceStub

channel = grpc.insecure_channel(sys.argv[1])
stub = FaceDetectionServiceStub(channel)
request = None
with open(sys.argv[2], "rb") as image_file:
    request = FaceDetectionRequest(image=image_file.read())
start = time.perf_counter()
response = stub.serve(request)
end = time.perf_counter()
print(f"Inference used {(end - start) * 1000}ms, detected {len(response.bbox)} faces")
image = cv2.imread(sys.argv[2])
for i in range(len(response.bbox)):
    x, y, width, height = (
        int(response.bbox[i].x * image.shape[1]),
        int(response.bbox[i].y * image.shape[0]),
        int(response.bbox[i].width * image.shape[1]),
        int(response.bbox[i].height * image.shape[0]),
    )
    cv2.rectangle(image, (x, y, width, height), (0, 255, 0))
    cv2.rectangle(image, (x, y - 8), (x + width, y), (0, 255, 0), -1)
    cv2.putText(
        image,
        f"{response.score[i] * 100:.2f}",
        (x, y - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 0, 0),
    )
    for point in response.landmark[i].point:
        x, y = (
            int(point.x * image.shape[1]),
            int(point.y * image.shape[0]),
        )
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
cv2.imwrite(sys.argv[3], image)
