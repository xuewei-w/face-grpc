cmake_minimum_required(VERSION 3.0)

project(Face)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(service
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service.grpc.pb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service.pb.cc)
target_link_libraries(service ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF})

add_executable(client ${CMAKE_CURRENT_SOURCE_DIR}/src/client.cpp)
target_link_libraries(client
    service
    absl::flags absl::flags_parse
    ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF}
    opencv_core opencv_imgcodecs opencv_imgproc)

add_library(engine SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/engine.cpp)
target_link_libraries(engine nvinfer cudart opencv_core)
add_library(utils SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp)
target_link_libraries(utils engine opencv_dnn)


add_executable(server ${CMAKE_CURRENT_SOURCE_DIR}/src/server.cpp)
target_link_libraries(server
    service engine utils 
    absl::flags absl::flags_parse
    ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF}
    opencv_imgcodecs)

