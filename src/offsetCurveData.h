/**
 * offsetCurveData.h
 * 오프셋 곡선 데이터 구조 정의
 * 특허 요구사항에 맞춰 확장
 */

#ifndef OFFSETCURVEDATA_H
#define OFFSETCURVEDATA_H

#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MVector.h>
#include <maya/MVectorArray.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MDagPath.h>
#include <maya/MDoubleArray.h>
#include <vector>
#include <limits>

// 곡선-정점 바인딩 클래스를 포함
#include "offsetCurveBinding.h"

// 곡선 데이터 클래스 (확장)
class offsetCurveData {
public:
    offsetCurveData();
    ~offsetCurveData();

    // 초기화 및 설정
    void initialize(const MDagPath& curvePath);
    void cacheBindPoseData();
    
    // 곡선 정보 액세스
    inline const MDagPath& getCurvePath() const { return mCurvePath; }
    inline const MPointArray& getBindCVs() const { return mBindCVs; }
    inline const MMatrixArray& getBindMatrices() const { return mBindMatrices; }
    
    // 곡선 조작 메서드
    MStatus updateCurveData();
    MStatus getCVs(MPointArray& cvArray) const;
    MStatus getPoint(double paramU, MPoint& point) const;
    MStatus getTangent(double paramU, MVector& tangent) const;
    
    // 곡률 계산 (추가)
    MStatus getCurvature(double paramU, double& curvature) const;
    
    // 방향 벡터 액세스 및 계산
    MStatus getOrientation(int cvIndex, double& orientation) const;
    MStatus computeOrientations();
    
    // 사용자 지정 방향 설정 (추가)
    MStatus setOrientation(int cvIndex, double orientation);
    
    // 가장 가까운 점 찾기 (최적화)
    double findClosestPoint(const MPoint& point, double& paramU) const;
    
    // 세그먼트 정보 (아크 구현용)
    int getSegmentIndex(double paramU) const;
    bool getSegmentPoints(int segmentIdx, MPoint& start, MPoint& end) const;
    bool getSegmentVectors(int segmentIdx, MVector& startTangent, MVector& endTangent) const;
    
    // 곡선 매개변수 조회
    int getNumCVs() const;
    int getNumSegments() const;
    int getDegree() const;
    double getLength() const;
    bool isClosed() const;
    
    // B-스플라인 특정 메서드
    MStatus getKnotVector(MDoubleArray& knots) const;
    MStatus evaluateBasis(double u, MDoubleArray& basis) const;

private:
    // 내부 유틸리티 메서드
    void initializeMatrices();
    void computeFrenetFrames();
    void computeSegmentData();   // 세그먼트 정보 계산 (아크 모드용)
    
    // 곡률 계산 헬퍼
    double calculateLocalCurvature(const MPoint& p1, const MPoint& p2, const MPoint& p3) const;

private:
    MDagPath mCurvePath;          // 곡선 객체 경로
    MPointArray mBindCVs;         // 바인드 포즈의 CV 위치
    MPointArray mCurrentCVs;      // 현재 CV 위치
    MMatrixArray mBindMatrices;   // 바인드 포즈 행렬
    MDoubleArray mOrientations;   // CV 방향값
    double mLength;               // 곡선 길이
    bool mIsClosed;               // 폐곡선 여부
    
    // 세그먼트 데이터 (아크 모드용)
    struct SegmentData {
        MPoint startPoint;
        MPoint endPoint;
        MVector startTangent;
        MVector endTangent;
        double length;
        double curvature;
    };
    std::vector<SegmentData> mSegments;
    
    // 곡률 캐시 (성능 최적화)
    MDoubleArray mCurvatureCache;
};

#endif // OFFSETCURVEDATA_H