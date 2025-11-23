# 一、MSVC的设定

MSVC工具链：全部自动检测
都是CLion自带的ninja.exe、cl.exe

名称：
CLion-MSVC-Release

构建类型：
Release

生成器：让CMake决定

CMake选项：

--preset "windows-msvc-release"

构建目录：
build\clion-msvc-release

构建选项：
（空）



# 二、MinGW的设定

MinGW工具链：全部自动检测
都是CLion自带的ninja.exe、gcc.exe、g++.exe

名称：
CLion-MinGW-Release

构建类型：
Release

工具链：MinGW

生成器：Ninja

CMake选项：
-G Ninja -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=FALSE -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -fopenmp" -DCMAKE_EXE_LINKER_FLAGS_RELEASE="-fopenmp"

构建目录：
build\clion-mingw-release

构建选项：
-j 30