/**
 * PerformanceMonitor.h
 * OCD 성능 모니터링 및 프로파일링 시스템
 * 실시간 성능 추적 및 최적화 가이드
 */

#ifndef PERFORMANCEMONITOR_H
#define PERFORMANCEMONITOR_H

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

// Maya 헤더
#include <maya/MStatus.h>
#include <maya/MString.h>

/**
 * 성능 측정 단위
 */
enum class PerformanceUnit {
    NANOSECONDS,
    MICROSECONDS,
    MILLISECONDS,
    SECONDS
};

/**
 * 성능 메트릭 구조체
 */
struct PerformanceMetric {
    std::string name;                    // 메트릭 이름
    double value;                        // 측정값
    PerformanceUnit unit;                // 단위
    std::chrono::high_resolution_clock::time_point timestamp;  // 타임스탬프
    std::map<std::string, double> additionalData;  // 추가 데이터
    
    PerformanceMetric(const std::string& n, double v, PerformanceUnit u) 
        : name(n), value(v), unit(u) {
        timestamp = std::chrono::high_resolution_clock::now();
    }
};

/**
 * 성능 섹션 클래스 (RAII 기반)
 */
class PerformanceSection {
private:
    std::string mSectionName;
    std::chrono::high_resolution_clock::time_point mStartTime;
    std::function<void(const std::string&, double)> mCallback;
    
public:
    PerformanceSection(const std::string& name, 
                      std::function<void(const std::string&, double)> callback = nullptr)
        : mSectionName(name), mCallback(callback) {
        mStartTime = std::chrono::high_resolution_clock::now();
    }
    
    ~PerformanceSection() {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - mStartTime);
        
        if (mCallback) {
            mCallback(mSectionName, duration.count());
        }
    }
    
    // 섹션 이름 반환
    const std::string& getName() const { return mSectionName; }
    
    // 경과 시간 반환 (마이크로초)
    double getElapsedTime() const {
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - mStartTime);
        return duration.count();
    }
};

/**
 * 메인 성능 모니터링 클래스
 */
class PerformanceMonitor {
private:
    // 싱글톤 인스턴스
    static std::unique_ptr<PerformanceMonitor> sInstance;
    
    // 성능 메트릭 저장소
    std::vector<PerformanceMetric> mMetrics;
    std::map<std::string, std::vector<double>> mSectionTimes;
    
    // 설정
    bool mEnabled;
    bool mAutoReport;
    PerformanceUnit mDefaultUnit;
    size_t mMaxMetrics;
    
    // 통계 계산
    struct Statistics {
        double min, max, mean, median, stdDev;
        size_t count;
    };
    
public:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    // 싱글톤 접근
    static PerformanceMonitor& getInstance();
    
    // 기본 설정
    void enable(bool enabled = true) { mEnabled = enabled; }
    void setAutoReport(bool autoReport) { mAutoReport = autoReport; }
    void setDefaultUnit(PerformanceUnit unit) { mDefaultUnit = unit; }
    void setMaxMetrics(size_t max) { mMaxMetrics = max; }
    
    // 성능 측정
    void startSection(const std::string& sectionName);
    void endSection(const std::string& sectionName);
    void recordMetric(const std::string& name, double value, PerformanceUnit unit = PerformanceUnit::MICROSECONDS);
    void recordMetricWithData(const std::string& name, double value, 
                             const std::map<std::string, double>& additionalData,
                             PerformanceUnit unit = PerformanceUnit::MICROSECONDS);
    
    // 성능 섹션 생성 (RAII)
    std::unique_ptr<PerformanceSection> createSection(const std::string& name);
    
    // 통계 계산
    Statistics calculateStatistics(const std::string& sectionName) const;
    Statistics calculateStatistics(const std::vector<double>& values) const;
    
    // 성능 분석
    std::vector<std::string> getTopSlowestSections(size_t count = 10) const;
    std::vector<std::string> getTopFastestSections(size_t count = 10) const;
    double getAverageTime(const std::string& sectionName) const;
    double getTotalTime(const std::string& sectionName) const;
    
    // 병목 지점 감지
    std::vector<std::string> detectBottlenecks(double threshold = 1000.0) const;  // 1ms 이상
    std::vector<std::string> detectPerformanceRegressions() const;
    
    // 리포트 생성
    std::string generateReport(bool includeDetails = true) const;
    std::string generateCSVReport() const;
    void exportReportToFile(const std::string& filename) const;
    
    // 메모리 사용량 모니터링
    size_t getCurrentMemoryUsage() const;
    size_t getPeakMemoryUsage() const;
    void resetMemoryTracking();
    
    // GPU 성능 모니터링 (CUDA)
    bool isGPUAvailable() const;
    size_t getGPUMemoryUsage() const;
    double getGPUUtilization() const;
    
    // 실시간 모니터링
    void startRealTimeMonitoring();
    void stopRealTimeMonitoring();
    bool isRealTimeMonitoringActive() const;
    
    // 데이터 정리
    void clearMetrics();
    void clearSection(const std::string& sectionName);
    void removeOldMetrics(size_t maxAgeInSeconds);
    
private:
    // 내부 헬퍼 메서드
    void addMetric(const PerformanceMetric& metric);
    std::string formatTime(double time, PerformanceUnit unit) const;
    std::string formatMemory(size_t bytes) const;
    void checkMemoryLimits();
    
    // 통계 계산 헬퍼
    double calculatePercentile(const std::vector<double>& values, double percentile) const;
    double calculateStandardDeviation(const std::vector<double>& values, double mean) const;
    
    // 실시간 모니터링
    bool mRealTimeMonitoring;
    std::chrono::high_resolution_clock::time_point mLastReportTime;
    double mReportInterval;  // 초 단위
};

/**
 * 성능 측정 매크로 (디버그 모드에서만 활성화)
 */
#ifdef _DEBUG
    #define PERFORMANCE_SECTION(name) \
        auto perfSection = PerformanceMonitor::getInstance().createSection(name)
    
    #define PERFORMANCE_RECORD(name, value) \
        PerformanceMonitor::getInstance().recordMetric(name, value)
    
    #define PERFORMANCE_START(name) \
        PerformanceMonitor::getInstance().startSection(name)
    
    #define PERFORMANCE_END(name) \
        PerformanceMonitor::getInstance().endSection(name)
#else
    #define PERFORMANCE_SECTION(name) \
        do {} while(0)
    
    #define PERFORMANCE_RECORD(name, value) \
        do {} while(0)
    
    #define PERFORMANCE_START(name) \
        do {} while(0)
    
    #define PERFORMANCE_END(name) \
        do {} while(0)
#endif

/**
 * 자동 성능 측정 클래스 (RAII)
 */
class AutoPerformanceTimer {
private:
    std::string mSectionName;
    std::chrono::high_resolution_clock::time_point mStartTime;
    
public:
    AutoPerformanceTimer(const std::string& sectionName) 
        : mSectionName(sectionName) {
        mStartTime = std::chrono::high_resolution_clock::now();
    }
    
    ~AutoPerformanceTimer() {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - mStartTime);
        
        PerformanceMonitor::getInstance().recordMetric(mSectionName, duration.count());
    }
    
    // 경과 시간 확인 (타이머 중단 없이)
    double getElapsedTime() const {
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - mStartTime);
        return duration.count();
    }
};

#endif // PERFORMANCEMONITOR_H
