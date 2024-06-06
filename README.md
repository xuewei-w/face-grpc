# 人脸检测gRPC服务Demo

## Pytorch人脸检测模型导出

- Python依赖: [pytorch](https://pytorch.org/), torchvision

- 所需文件:

    ```
    project
    |-- py
    |   |-- config.py
    |   |-- export.py
    |   |-- model.py
    |   `-- onnx_export.sh
    `-- static
        `-- Resnet50_Final.pth
    ```

    `static/Resnet50_Final.pth`来自[Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

- 导出onnx模型:

    1. 将`py/onnx_export.sh`文件中

        ```
        echoExec source ~/miniconda3/bin/activate retinaface
        ```

        `~/miniconda3/bin/activate`改成目标系统中conda的`activate`的路径

        将`retinaface`改成满足Python依赖的conda环境名

    2. 导出onnx模型

        ```
        $ chmod 775 py/onnx_export.sh
        $ ./py/onnx_export.sh
        ```

        将产生模型文件`static/FaceDetector.onnx`

## C++ 服务Demo (NVIDIA GPU) + 客户端Demo

- 依赖：[CMake](https://cmake.org/), [OpenCV](https://opencv.org/releases/), [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing)10.0+ (CUDA runtime), [gRPC](https://grpc.io/docs/languages/cpp/quickstart/)

    <p style="background-color:yellow">&#10071<b style="color:red">安装依赖, 编译, 运行等过程需要全程deactivate所有conda环境</font></b>&#10071</p>

    安装依赖后记得在`~/.bashrc`添加环境变量 (也可以不添加环境变量,在`sh/trt_export.sh`写明`trtexec`的路径,在`sh/protoc.sh`写明`protoc` (gRPC附带安装) 的路径,手动在`CMakeLists.txt`中写明链接路径)

    以下只涉及部分依赖,仅供参考,根据实际情况修改

    ```
    # CUDA
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
    export C_INCLUDE_PATH="/usr/local/cuda/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="/usr/local/cuda/include:$CPLUS_INCLUDE_PATH"

    # TensorRT exec
    export PATH="/usr/src/tensorrt/bin:$PATH"

    # gRPC
    export PATH="~/.local/share/grpc/bin:$PATH"
    export LD_LIBRARY_PATH="~/.local/share/grpc/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="~/.local/share/grpc/lib:$LIBRARY_PATH"
    export C_INCLUDE_PATH="~/.local/share/grpc/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="~/.local/share/grpc/include:$CPLUS_INCLUDE_PATH"
    ```

- 所需文件：

    ```
    project
    |-- cmake
    |   `-- common.cmake
    |-- sh
    |   |-- make.sh
    |   |-- protoc.sh
    |   |-- run_client.sh
    |   |-- run_server.sh
    |   `-- trt_export.sh
    |-- src
    |   |-- client.cpp  # 客户端实现源码
    |   |-- engine.cpp  # TensorRT engine初始化,通用推理实现源码
    |   |-- engine.hpp  # TensorRT engine初始化,通用推理接口头文件
    |   |-- server.cpp  # 服务端实现源码
    |   |-- service.proto  # gRPC数据结构定义
    |   |-- utils.cpp  # 人脸检测模型初始化,(普通/滑窗)预处理&后处理实现源码
    |   `-- utils.hpp  # 人脸检测模型初始化,推理接口头文件
    |-- static
    |   |-- FaceDetector.onnx
    |   `-- test.jpg  # 自行放置推理图片
    `-- CMakeLists.txt
    ```

- 编译运行:

    1. 为脚本添加运行权限

        ```
        $ chmod -R 775 sh/
        ```

    2. 将onnx模型导出为TensorRT engine

        ```
        $ ./sh/trt_export.sh
        ```

        将产生模型文件`static/FaceDetector.engine`

    3. 生成gRPC代码

        ```
        $ ./sh/protoc.sh
        ```

        将产生代码文件

        ```
        `-- src
            |-- service.grpc.pb.cc
            |-- service.grpc.pb.h
            |-- service.pb.cc
            `-- service.pb.h
        ```

    4. 编译

        ```
        $ ./sh/make.sh
        ```

        将产生文件
        
        ```
        |-- bin
        |   |-- client
        |   `-- server
        `-- lib
            |-- libengine.so
            `-- libutils.so
        ```

    5. 运行服务

        ```
        $ ./sh/run_server.sh
        ```
    
    6. 运行客户端

        ```
        $ ./sh/run_client.sh
        ```

        将产生推理结果`output/output.jpg`

## Python客户端Demo

- Python依赖: grpcio, grpcio-tools, opencv

- 所需文件:

    ```
    project
    |-- py
    |   |-- client.py  # 客户端实现源码
    |   |-- protoc.sh
    |   `-- run_client.sh
    |-- src
    |   `-- service.proto  # gRPC数据结构定义
    `-- static
        `-- test.jpg  # 自行放置推理图片
    ```

- 调用gRPC服务:

    1. 将`py/protoc.sh`和`py/run_client.sh`文件中

        ```
        echoExec source ~/miniconda3/bin/activate grpc
        ```

        `~/miniconda3/bin/activate`改成目标系统中conda的`activate`的路径

        将`gprc`改成满足Python依赖的conda环境名

    2. 生成gRPC代码

        ```
        $ chmod 775 py/protoc.sh
        $ ./py/protoc.sh
        ```

        将产生代码文件

        ```
        `-- py
            |-- service_pb2_grpc.py
            `-- service_pb2.py
        ```

    3. 运行客户端

        ```
        $ chmod 775 py/run_client.sh
        $ ./py/run_client.sh
        ```

        将产生推理结果`py/output.jpg`


