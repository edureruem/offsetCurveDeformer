/**
 * offsetCurveData.cpp
 * 오프셋 곡선 데이터 구조 구현
 */

#include "offsetCurveData.h"
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnNurbsCurveData.h>
#include <maya/MPlug.h>
#include <maya/MDagPath.h>
#include <maya/MGlobal.h>
#include <algorithm>
#include <cmath>

// 생성자
offsetCurveData::offsetCurveData()
    : mLength(0.0)
    , mIsClosed(false)
{
    // 기본 초기화
}

// 소멸자
offsetCurveData::~offsetCurveData()
{
    // 필요한 정리 작업
}

// 곡선 초기화
void offsetCurveData::initialize(const MDagPath& curvePath)
{
    mCurvePath = curvePath;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath);
    
    // 기본 곡선 정보 캐시
    mIsClosed = fnCurve.form() == MFnNurbsCurve::kClosed || 
                fnCurve.form() == MFnNurbsCurve::kPeriodic;
    
    mLength = fnCurve.length();
    
    // CV 정보 가져오기
    fnCurve.getCVs(mCurrentCVs);
    
    // 행렬 초기화
    initializeMatrices();
    
    // 방향 계산
    computeOrientations();
    
    // 세그먼트 데이터 계산
    computeSegmentData();
    
    // 곡률 캐시 초기화
    mCurvatureCache.setLength(100); // 100개 샘플 포인트에 대한 곡률 캐싱
    
    for (unsigned int i = 0; i < mCurvatureCache.length(); i++) {
        double paramU = static_cast<double>(i) / (mCurvatureCache.length() - 1);
        MPoint prev, curr, next;
        
        // 샘플링 포인트
        double step = 0.01;
        getPoint(paramU - step, prev);
        getPoint(paramU, curr);
        getPoint(paramU + step, next);
        
        // 곡률 계산 및 캐싱
        mCurvatureCache[i] = calculateLocalCurvature(prev, curr, next);
    }
}

// 바인드 포즈 데이터 캐싱
void offsetCurveData::cacheBindPoseData()
{
    // 현재 CV들을 바인드 포즈로 저장
    mBindCVs.copy(mCurrentCVs);
    
    // 프레넷 프레임 계산 (탄젠트/노멀/바이노멀)
    computeFrenetFrames();
}

// 곡선 데이터 업데이트
MStatus offsetCurveData::updateCurveData()
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 현재 CV 정보 업데이트
    status = fnCurve.getCVs(mCurrentCVs);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 길이 업데이트
    mLength = fnCurve.length();
    
    // 세그먼트 데이터 재계산
    computeSegmentData();
    
    return MS::kSuccess;
}

// CV 배열 반환
MStatus offsetCurveData::getCVs(MPointArray& cvArray) const
{
    cvArray.copy(mCurrentCVs);
    return MS::kSuccess;
}

// 매개변수 위치의 점 계산
MStatus offsetCurveData::getPoint(double paramU, MPoint& point) const
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 매개변수 위치의 점 계산
    status = fnCurve.getPointAtParam(paramU, point);
    
    return status;
}

// 매개변수 위치의 접선 벡터 계산
MStatus offsetCurveData::getTangent(double paramU, MVector& tangent) const
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 접선 벡터 계산
    status = fnCurve.getTangent(paramU, tangent);
    tangent.normalize();
    
    return status;
}

// 곡률 계산
MStatus offsetCurveData::getCurvature(double paramU, double& curvature) const
{
    // 인덱스 계산 (캐시에서 근사값 찾기)
    unsigned int index = static_cast<unsigned int>(paramU * (mCurvatureCache.length() - 1));
    
    // 범위 검사
    if (index >= mCurvatureCache.length()) {
        index = mCurvatureCache.length() - 1;
    }
    
    // 캐시된 값 반환
    curvature = mCurvatureCache[index];
    
    return MS::kSuccess;
}

// 곡률 계산 헬퍼 메소드
double offsetCurveData::calculateLocalCurvature(const MPoint& p1, const MPoint& p2, const MPoint& p3) const
{
    // 삼각형 세 점으로 원의 곡률 계산
    MVector v1 = p1 - p2;
    MVector v2 = p3 - p2;
    
    double v1Length = v1.length();
    double v2Length = v2.length();
    
    // 벡터 길이가 너무 작으면 곡률은 0
    if (v1Length < 1e-6 || v2Length < 1e-6) {
        return 0.0;
    }
    
    // 정규화
    v1 /= v1Length;
    v2 /= v2Length;
    
    // 사인 계산
    double sinTheta = (v1 ^ v2).length(); // 외적의 크기 = sin(theta)
    
    // 내적으로 코사인 계산
    double cosTheta = v1 * v2;  // 내적 = |v1|*|v2|*cos(theta)
    
    // 곡률 = 2 * sin(theta) / 현의 길이
    double chordLength = (p3 - p1).length();
    
    if (chordLength < 1e-6) {
        return 0.0;
    }
    
    double k = 2.0 * sinTheta / chordLength;
    
    // 방향 결정 (시계/반시계)
    MVector normal = v1 ^ v2;
    if (normal.y < 0) {
        k = -k; // 시계 방향이면 부호 반전
    }
    
    return k;
}

// CV 방향 계산
MStatus offsetCurveData::computeOrientations()
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 방향 배열 초기화
    int numCVs = fnCurve.numCVs();
    mOrientations.setLength(numCVs);
    
    // 각 CV마다 방향 계산
    for (int i = 0; i < numCVs; i++) {
        // 양쪽 점에서 접선 계산
        double paramU = static_cast<double>(i) / (numCVs - 1);
        MVector tangent;
        
        status = fnCurve.getTangent(paramU, tangent);
        if (status != MS::kSuccess) {
            // 접선을 얻지 못하면 기본 방향 사용
            mOrientations[i] = 0.0;
            continue;
        }
        
        // 방향 각도 계산 (XZ 평면의 각도)
        double angle = atan2(tangent.z, tangent.x);
        mOrientations[i] = angle;
    }
    
    return MS::kSuccess;
}

// CV 방향 반환
MStatus offsetCurveData::getOrientation(int cvIndex, double& orientation) const
{
    if (cvIndex < 0 || cvIndex >= static_cast<int>(mOrientations.length())) {
        return MS::kInvalidParameter;
    }
    
    orientation = mOrientations[cvIndex];
    return MS::kSuccess;
}

// CV 방향 설정 (아티스트 제어용)
MStatus offsetCurveData::setOrientation(int cvIndex, double orientation)
{
    if (cvIndex < 0 || cvIndex >= static_cast<int>(mOrientations.length())) {
        return MS::kInvalidParameter;
    }
    
    mOrientations[cvIndex] = orientation;
    return MS::kSuccess;
}

// 가장 가까운 점 찾기
double offsetCurveData::findClosestPoint(const MPoint& point, double& paramU) const
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    
    if (status != MS::kSuccess) {
        paramU = 0.0;
        return std::numeric_limits<double>::max();
    }
    
    // 가장 가까운 점 찾기
    double parameter;
    MPoint closestPoint;
    
    // Maya API를 사용하여 가장 가까운 점 찾기
    status = fnCurve.closestPoint(point, &closestPoint, &parameter);
    
    if (status != MS::kSuccess) {
        paramU = 0.0;
        return std::numeric_limits<double>::max();
    }
    
    // 결과 반환
    paramU = parameter;
    return (point - closestPoint).length();
}

// 세그먼트 인덱스 찾기
int offsetCurveData::getSegmentIndex(double paramU) const
{
    // 곡선 세그먼트 수 가져오기
    int numSegments = getNumSegments();
    
    // 파라미터를 세그먼트 인덱스로 변환
    int segmentIndex = static_cast<int>(paramU * numSegments);
    
    // 범위 체크
    if (segmentIndex >= numSegments) {
        segmentIndex = numSegments - 1;
    }
    
    return segmentIndex;
}

// 세그먼트 점 반환
bool offsetCurveData::getSegmentPoints(int segmentIdx, MPoint& start, MPoint& end) const
{
    if (segmentIdx < 0 || segmentIdx >= static_cast<int>(mSegments.size())) {
        return false;
    }
    
    start = mSegments[segmentIdx].startPoint;
    end = mSegments[segmentIdx].endPoint;
    
    return true;
}

// 세그먼트 벡터 반환
bool offsetCurveData::getSegmentVectors(int segmentIdx, MVector& startTangent, MVector& endTangent) const
{
    if (segmentIdx < 0 || segmentIdx >= static_cast<int>(mSegments.size())) {
        return false;
    }
    
    startTangent = mSegments[segmentIdx].startTangent;
    endTangent = mSegments[segmentIdx].endTangent;
    
    return true;
}

// CV 수 반환
int offsetCurveData::getNumCVs() const
{
    MStatus status;
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    
    if (status != MS::kSuccess) {
        return 0;
    }
    
    return fnCurve.numCVs();
}

// 세그먼트 수 반환
int offsetCurveData::getNumSegments() const
{
    return static_cast<int>(mSegments.size());
}

// 곡선 차수 반환
int offsetCurveData::getDegree() const
{
    MStatus status;
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    
    if (status != MS::kSuccess) {
        return 3; // 기본값 (3차 곡선)
    }
    
    return fnCurve.degree();
}

// 곡선 길이 반환
double offsetCurveData::getLength() const
{
    return mLength;
}

// 폐곡선 여부 반환
bool offsetCurveData::isClosed() const
{
    return mIsClosed;
}

// 매트릭스 초기화
void offsetCurveData::initializeMatrices()
{
    // CV 수에 맞게 행렬 배열 크기 설정
    mBindMatrices.setLength(mCurrentCVs.length());
    
    // 모든 행렬을 단위 행렬로 초기화
    for (unsigned int i = 0; i < mBindMatrices.length(); i++) {
        mBindMatrices[i] = MMatrix::identity;
    }
}

// 프레넷 프레임 계산 (로컬 좌표계)
void offsetCurveData::computeFrenetFrames()
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    
    if (status != MS::kSuccess) {
        return;
    }
    
    // 각 CV 위치의 프레넷 프레임 계산
    for (unsigned int i = 0; i < mBindMatrices.length(); i++) {
        double paramU = static_cast<double>(i) / (mBindMatrices.length() - 1);
        
        // 접선 계산
        MVector tangent;
        status = fnCurve.getTangent(paramU, tangent);
        tangent.normalize();
        
        // 초기 업 벡터
        MVector upVector(0.0, 1.0, 0.0);
        
        // 노멀 벡터
        MVector normal = upVector - (upVector * tangent) * tangent;
        if (normal.length() < 1e-6) {
            // 접선이 업 벡터와 평행하면 대체 업 벡터 사용
            upVector = MVector(0.0, 0.0, 1.0);
            normal = upVector - (upVector * tangent) * tangent;
        }
        normal.normalize();
        
        // 바이노멀 벡터
        MVector binormal = tangent ^ normal;
        binormal.normalize();
        
        // 프레넷 프레임 행렬 구성
        mBindMatrices[i][0][0] = tangent.x;
        mBindMatrices[i][0][1] = tangent.y;
        mBindMatrices[i][0][2] = tangent.z;
        
        mBindMatrices[i][1][0] = normal.x;
        mBindMatrices[i][1][1] = normal.y;
        mBindMatrices[i][1][2] = normal.z;
        
        mBindMatrices[i][2][0] = binormal.x;
        mBindMatrices[i][2][1] = binormal.y;
        mBindMatrices[i][2][2] = binormal.z;
    }
}

// 세그먼트 데이터 계산 (아크 모드용)
void offsetCurveData::computeSegmentData()
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    
    if (status != MS::kSuccess) {
        return;
    }
    
    // 세그먼트 샘플 수 결정
    int numSegments = 20; // 세그먼트 수 조정 가능
    mSegments.clear();
    mSegments.resize(numSegments);
    
    double stepSize = 1.0 / numSegments;
    
    // 각 세그먼트에 대한 데이터 계산
    for (int i = 0; i < numSegments; i++) {
        double startU = i * stepSize;
        double endU = (i + 1) * stepSize;
        
        if (i == numSegments - 1 && mIsClosed) {
            endU = 0.0; // 폐곡선이면 마지막 세그먼트는 처음으로 연결
        }
        
        // 세그먼트 시작과 끝점
        MPoint startPoint, endPoint;
        fnCurve.getPointAtParam(startU, startPoint);
        fnCurve.getPointAtParam(endU, endPoint);
        
        // 세그먼트 접선
        MVector startTangent, endTangent;
        fnCurve.getTangent(startU, startTangent);
        fnCurve.getTangent(endU, endTangent);
        
        // 정규화
        startTangent.normalize();
        endTangent.normalize();
        
        // 세그먼트 길이 계산
        double length = fnCurve.length(startU, endU);
        
        // 세그먼트 곡률 계산 (3점 근사)
        MPoint midPoint;
        fnCurve.getPointAtParam((startU + endU) * 0.5, midPoint);
        double curvature = calculateLocalCurvature(startPoint, midPoint, endPoint);
        
        // 세그먼트 데이터 저장
        mSegments[i].startPoint = startPoint;
        mSegments[i].endPoint = endPoint;
        mSegments[i].startTangent = startTangent;
        mSegments[i].endTangent = endTangent;
        mSegments[i].length = length;
        mSegments[i].curvature = curvature;
    }
}

// 매듭 벡터 반환
MStatus offsetCurveData::getKnotVector(MDoubleArray& knots) const
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 매듭 벡터 가져오기
    status = fnCurve.getKnots(knots);
    
    return status;
}

// B-스플라인 기저 함수 평가
MStatus offsetCurveData::evaluateBasis(double u, MDoubleArray& basis) const
{
    MStatus status;
    
    // 곡선 함수 세트 객체 생성
    MFnNurbsCurve fnCurve(mCurvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 기저 함수 개수 설정
    basis.setLength(fnCurve.numCVs());
    
    // 매듭 벡터 가져오기
    MDoubleArray knots;
    status = fnCurve.getKnots(knots);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    int degree = fnCurve.degree();
    int numCVs = fnCurve.numCVs();
    
    // 모든 기저 함수 초기화
    for (int i = 0; i < numCVs; i++) {
        basis[i] = 0.0;
    }
    
    // 매개변수가 유효한 범위에 있는지 확인
    if (u < knots[degree] || u > knots[numCVs]) {
        return MS::kFailure;
    }
    
    // 해당 매개변수에 영향을 주는 CV 스팬 찾기
    int span = degree;
    while (span < numCVs && knots[span+1] <= u) {
        span++;
    }
    
    // 기본 재귀 알고리즘 대신 Maya API의 계산 사용
    MPointArray cvs;
    fnCurve.getCVs(cvs);
    
    // 영향을 주는 기저 함수 계산
    MPoint result;
    fnCurve.getPointAtParam(u, result);
    
    // 영향력 계산은 복잡하므로 대략적인 근사값 사용
    // 실제 B-스플라인 기저 함수 계산은 더 복잡함
    double sum = 0.0;
    for (int i = span - degree; i <= span; i++) {
        if (i >= 0 && i < numCVs) {
            // 거리 기반 가중치 (간단한 근사)
            basis[i] = 1.0 - (result - cvs[i]).length() / (degree + 1);
            if (basis[i] < 0.0) basis[i] = 0.0;
            sum += basis[i];
        }
    }
    
    // 정규화
    if (sum > 0.0) {
        for (int i = 0; i < numCVs; i++) {
            basis[i] /= sum;
        }
    }
    
    return MS::kSuccess;
}