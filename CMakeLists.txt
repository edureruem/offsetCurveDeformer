cmake_minimum_required(VERSION 3.16)
project(offsetCurveDeformer)

set(PROJECT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT})
set(CMAKE_INSTALL_PREFIX ${PROJECT_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# Maya 버전별 설정
if(DEFINED MAYA_VERSION)
    set(MAYA_VERSION_STRING ${MAYA_VERSION})
else()
    set(MAYA_VERSION_STRING "2025")
endif()

# AVX 최적화 플래그
if (WIN32)
    set(COMPILE_FLAGS "/arch:AVX")
else()
    set(COMPILE_FLAGS "-mavx")
endif()

# Maya 찾기
find_package(Maya REQUIRED)

# 소스 파일들
set(SOURCE_FILES
    "src/pluginMain.cpp"
    "src/offsetCurveDeformer.cpp"
    "src/offsetCurveDeformer.h"
    "src/offsetCurveCmd.cpp"
    "src/offsetCurveCmd.h"
    "src/offsetCurveAlgorithm.cpp"
    "src/offsetCurveAlgorithm.h"
    "src/offsetCurveControlParams.cpp"
    "src/offsetCurveControlParams.h"
    "src/offsetCurveSystems.cpp"
    "src/offsetCurveSystems.h"
    "src/common.cpp"
    "src/common.h"
)

# 라이브러리 생성
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE Maya::Maya)
target_include_directories(${PROJECT_NAME} PRIVATE Maya::Maya)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_FLAGS})

# Maya 플러그인으로 등록
MAYA_PLUGIN(${PROJECT_NAME})

# 설치
install(TARGETS ${PROJECT_NAME} ${MAYA_TARGET_TYPE} DESTINATION plug-ins)
