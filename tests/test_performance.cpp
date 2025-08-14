#include <gtest/gtest.h>

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 성능 테스트 설정
    }
};

// 테스트 케이스 1: 기본 성능 측정
TEST_F(PerformanceTest, BasicPerformanceMeasurement) {
    // 간단한 연산 수행
    volatile int result = 0;
    for (int i = 0; i < 1000; ++i) {
        result += i;
    }
    
    // 결과 검증
    EXPECT_EQ(result, 499500);
}

// 테스트 케이스 2: 메모리 할당 성능
TEST_F(PerformanceTest, MemoryAllocationPerformance) {
    // 메모리 할당/해제 테스트
    std::vector<int> testVector;
    for (int i = 0; i < 100; ++i) {
        testVector.push_back(i);
    }
    
    // 결과 검증
    EXPECT_EQ(testVector.size(), 100);
    EXPECT_EQ(testVector[0], 0);
    EXPECT_EQ(testVector[99], 99);
}

// 테스트 케이스 3: 수학 연산 성능
TEST_F(PerformanceTest, MathOperationsPerformance) {
    // 수학 연산 테스트
    volatile double sum = 0.0;
    for (int i = 0; i < 10000; ++i) {
        sum += i * 2.0;
    }
    
    // 결과 검증
    EXPECT_GT(sum, 0.0);
    EXPECT_DOUBLE_EQ(sum, 99990000.0); // 0부터 9999까지 * 2의 합
}

// 테스트 케이스 4: 배열 접근 성능
TEST_F(PerformanceTest, ArrayAccessPerformance) {
    // 배열 접근 테스트
    std::vector<double> testArray(1000, 1.0);
    volatile double sum = 0.0;
    
    for (size_t i = 0; i < testArray.size(); ++i) {
        sum += testArray[i];
    }
    
    // 결과 검증
    EXPECT_DOUBLE_EQ(sum, 1000.0);
}

// 테스트 케이스 5: 문자열 연산 성능
TEST_F(PerformanceTest, StringOperationsPerformance) {
    // 문자열 연산 테스트
    std::string testString = "Hello";
    for (int i = 0; i < 100; ++i) {
        testString += "World";
    }
    
    // 결과 검증
    EXPECT_EQ(testString.length(), 505); // "Hello" + 100 * "World"
    EXPECT_EQ(testString.substr(0, 5), "Hello");
}
