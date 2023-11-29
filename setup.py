import os
import os.path as osp
from loguru import logger


_commands = [
    "sudo apt install -y libyaml-cpp0.6",
    "sudo apt install -y libopencv-dev",
    "pip install -r requirements.txt",
    "wget https://github.com/LSH9832/SOT-MNN/releases/download/v0.0.1/weights.zip",
    "unzip weights.zip && rm weights.zip",
    "mkdir build",
    "cd build && cmake .. && make && cd ..",
]


if __name__ == "__main__":
    while True:
        mnn_root = osp.abspath(input("please input your MNN path(example: /home/user/MNN-2.7.1, include and build in it)\n>>> "))

        if not osp.isfile(osp.join(mnn_root, "build/MNNConvert")):
            logger.error(f"file MNNConvert does not exist in {mnn_root}/build !")
            continue
        if not osp.isdir(osp.join(mnn_root, "include")):
            logger.error(f"dir include does not exist in {mnn_root} !")
            continue
            
        break

    cmake_file_content = "cmake_minimum_required(VERSION 3.0.2)\n" \
                         "project(mnn_sot)\n" \
                         "include_directories(\n" \
                         "    include\n" \
                         ")\n\n" \
                         "find_package(OpenCV 4 REQUIRED)\n" \
                         "include_directories(${OpenCV_INCLUDE_DIRS})\n\n" \
                         "include_directories(/usr/include/eigen3)\n\n" \
                         "# --------------modify your own mnn path---------------\n" \
                         f"include_directories(/home/lsh/code/mnn_dev/MNN-2.7.1/include)\n" \
                         f"link_directories(/home/lsh/code/mnn_dev/MNN-2.7.1/build)\n" \
                         "# --------------modify your own mnn path---------------\n\n\n" \
                         "add_executable(mnn_sot\n" \
                         "    src/demo.cpp\n" \
                         ")\n\n" \
                         "target_link_libraries(mnn_sot ${OpenCV_LIBRARIES})\n" \
                         "target_link_libraries(mnn_sot MNN)\n" \
                         "target_link_libraries(mnn_sot yaml-cpp)" \
    
    open("CMakeLists.txt", "w").write(cmake_file_content)

    for cmd in _commands:
        os.system(cmd)
