# 소스 파일 목록 (cvwrap 방식)
set(SOURCE_FILES
    "pluginMain.cpp"
    "offsetCurveCmd.cpp"
    "offsetCurveCmd.h"
    "offsetCurveDeformerNode.cpp"
    "offsetCurveDeformerNode.h"
    "offsetCurveAlgorithm.cpp"
    "offsetCurveAlgorithm.h"
    "offsetCurveControlParams.cpp"
    "offsetCurveControlParams.h"
    "offsetCurveSystems.cpp"
    "offsetCurveSystems.h"
    "offsetCurveTypes.h"
)

# AVX 최적화 플래그 설정
if (WIN32)
    set(COMPILE_FLAGS "/arch:AVX")
else()
    set(COMPILE_FLAGS "-mavx")
endif()

find_package(Maya REQUIRED)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_FLAGS})

# UTF-8 인코딩 설정 (한글 문자 문제 해결)
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE Maya::Maya)
target_include_directories(${PROJECT_NAME} 
    PRIVATE Maya::Maya
    PUBLIC "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
)
MAYA_PLUGIN(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} ${MAYA_TARGET_TYPE} DESTINATION plug-ins)

