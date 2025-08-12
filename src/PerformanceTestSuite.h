/**
 * PerformanceTestSuite.h
 * Offset Curve Deformer 성능 벤치마크 및 최적화 검증 시스템
 */

#ifndef PERFORMANCETESTSUITE_H
#define PERFORMANCETESTSUITE_H

#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MFnMesh.h>
#include <maya/MDagPath.h>
#include <maya/MTime.h>
#include <maya/MGlobal.h>
#include <chrono>
#include <vector>
#include <string>
#include <memory>

// 성능 테스트 결과 구조체
struct PerformanceResult {
    std::string testName;
    int vertexCount;
    int curveCount;
    double cpuTime;      // CPU 처리 시간 (ms)
    double gpuTime;      // GPU 처리 시간 (ms)
    double speedup;      // GPU 가속 비율
    double memoryUsage;  // 메모리 사용량 (MB)
    bool success;        // 테스트 성공 여부
    std::string errorMessage;
};

// 성능 테스트 스위트 클래스
class PerformanceTestSuite {
public:
    PerformanceTestSuite();
    ~PerformanceTestSuite();
    
    // 메인 테스트 실행
    bool runAllTests();
    
    // 개별 테스트들
    bool testSmallMesh();      // 1K 정점 테스트
    bool testMediumMesh();     // 10K 정점 테스트
    bool testLargeMesh();      // 100K 정점 테스트
    bool testHugeMesh();       // 1M+ 정점 테스트
    
    // 성능 비교 테스트
    bool testCPUvsGPU();       // CPU vs GPU 성능 비교
    bool testStrategyPattern(); // 전략 패턴 성능 비교
    bool testMemoryEfficiency(); // 메모리 효율성 테스트
    
    // 결과 출력
    void printResults() const;
    void exportResultsToCSV(const std::string& filename) const;
    
private:
    // 테스트 헬퍼 메서드들
    MPointArray generateTestMesh(int vertexCount, double size = 10.0);
    std::vector<MDagPath> generateTestCurves(int curveCount);
    
    // 성능 측정 헬퍼
    double measureExecutionTime(std::function<void()> func);
    double getMemoryUsage();
    
    // 테스트 결과 저장
    std::vector<PerformanceResult> mResults;
    
    // 테스트 설정
    struct TestConfig {
        int maxVertices = 1000000;  // 최대 테스트 정점 수
        int maxCurves = 10;         // 최대 테스트 곡선 수
        int iterations = 5;         // 각 테스트 반복 횟수
        bool enableGPU = true;      // GPU 테스트 활성화
        bool enableCPU = true;      // CPU 테스트 활성화
    } mConfig;
};

#endif // PERFORMANCETESTSUITE_H
