# 테스트 시스템 설정
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# 테스트 실행 파일들
add_executable(OCDAlgorithmTests 
    test_offsetCurveAlgorithm.cpp 
    test_offsetCurveTypes.cpp 
    test_performance.cpp
)

add_executable(OCDPluginTests 
    test_plugin_loading.cpp 
    test_node_creation.cpp 
    test_deformation.cpp
)

add_executable(OCDIntegrationTests 
    test_integration.cpp 
    test_performance_benchmark.cpp 
    test_memory_usage.cpp
)

# 링크 설정 - 실제 존재하는 라이브러리만 링크
target_link_libraries(OCDAlgorithmTests 
    gtest gtest_main
)

target_link_libraries(OCDPluginTests 
    gtest gtest_main 
    offsetCurveDeformer
)

target_link_libraries(OCDIntegrationTests 
    gtest gtest_main 
    offsetCurveDeformer
)

# include 경로 설정 - 소스 헤더 파일들을 찾을 수 있도록
target_include_directories(OCDAlgorithmTests 
    PRIVATE 
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_BINARY_DIR}/src"
)

target_include_directories(OCDPluginTests 
    PRIVATE 
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_BINARY_DIR}/src"
)

target_include_directories(OCDIntegrationTests 
    PRIVATE 
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_BINARY_DIR}/src"
)

# 테스트 등록
add_test(NAME OCDAlgorithmTests COMMAND OCDAlgorithmTests)
add_test(NAME OCDPluginTests COMMAND OCDPluginTests)
add_test(NAME OCDIntegrationTests COMMAND OCDIntegrationTests)

# 테스트 속성 설정
set_tests_properties(OCDAlgorithmTests PROPERTIES 
    TIMEOUT 300 
    ENVIRONMENT "MAYA_PLUG_IN_PATH=${CMAKE_BINARY_DIR}/plug-ins"
)

set_tests_properties(OCDPluginTests PROPERTIES 
    TIMEOUT 300 
    ENVIRONMENT "MAYA_PLUG_IN_PATH=${CMAKE_BINARY_DIR}/plug-ins"
)

set_tests_properties(OCDIntegrationTests PROPERTIES 
    TIMEOUT 300 
    ENVIRONMENT "MAYA_PLUG_IN_PATH=${CMAKE_BINARY_DIR}/plug-ins"
)
