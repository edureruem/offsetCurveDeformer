#!/bin/bash

echo "========================================"
echo "Offset Curve Deformer Build Test"
echo "========================================"

# 기존 빌드 폴더 정리
if [ -d "build.clean" ]; then
    echo "Cleaning previous build..."
    rm -rf "build.clean"
fi

# 새 빌드 폴더 생성
mkdir "build.clean"
cd "build.clean"

# CMake 설정
echo "Configuring with CMake..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    cmake .. -DMAYA_VERSION=2020
else
    # Linux
    cmake .. -DMAYA_VERSION=2020
fi

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# 빌드 실행
echo "Building project..."
cmake --build . --config Release

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Plugin location: build.clean/src/offsetCurveDeformer"

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Plugin extension: .bundle"
else
    echo "Plugin extension: .so"
fi

echo ""
echo "To install to Maya:"
echo "1. Copy the plugin to your Maya plug-ins folder"
echo "2. Restart Maya"
echo "3. Load the plugin: Plug-in Manager"
echo ""
