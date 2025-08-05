/**
 * offsetCurveBinding.cpp
 * 곡선-정점 바인딩 정보 클래스 구현
 */

#include "offsetCurveBinding.h"

// 생성자 - 모든 멤버 초기화
offsetCurveBinding::offsetCurveBinding()
    : mCurveIndex(-1),
      mParamU(0.0),
      mWeight(0.0),
      mBindLocalPoint(0.0, 0.0, 0.0),
      mBindMatrix(),
      mTangent(1.0, 0.0, 0.0),
      mNormal(0.0, 1.0, 0.0),
      mBinormal(0.0, 0.0, 1.0),
      mCurvature(0.0),
      mSegmentLength(0.0),
      mSegmentIndex(0),
      mIsJunction(false),
      mJunctionRadius(0.0)
{
}

// 소멸자
offsetCurveBinding::~offsetCurveBinding()
{
    // 특별한 정리 작업 없음
}

//---------- 데이터 설정 메서드 구현 ----------//

void offsetCurveBinding::setCurveIndex(int index)
{
    mCurveIndex = index;
}

void offsetCurveBinding::setParamU(double param)
{
    mParamU = param;
}

void offsetCurveBinding::setWeight(double weight)
{
    mWeight = weight;
}

void offsetCurveBinding::setBindLocalPoint(const MPoint& point)
{
    mBindLocalPoint = point;
}

void offsetCurveBinding::setBindMatrix(const MMatrix& matrix)
{
    mBindMatrix = matrix;
}

void offsetCurveBinding::setTangent(const MVector& tangent)
{
    mTangent = tangent;
}

void offsetCurveBinding::setNormal(const MVector& normal)
{
    mNormal = normal;
}

void offsetCurveBinding::setBinormal(const MVector& binormal)
{
    mBinormal = binormal;
}

void offsetCurveBinding::setCurvature(double curvature)
{
    mCurvature = curvature;
}

void offsetCurveBinding::setSegmentIndex(int index)
{
    mSegmentIndex = index;
}

void offsetCurveBinding::setIsJunction(bool isJunction)
{
    mIsJunction = isJunction;
}

void offsetCurveBinding::setJunctionRadius(double radius)
{
    mJunctionRadius = radius;
}

void offsetCurveBinding::setSegmentLength(double length)
{
    mSegmentLength = length;
}

//---------- 데이터 접근 메서드 구현 ----------//

int offsetCurveBinding::getCurveIndex() const
{
    return mCurveIndex;
}

double offsetCurveBinding::getParamU() const
{
    return mParamU;
}

double offsetCurveBinding::getWeight() const
{
    return mWeight;
}

MPoint offsetCurveBinding::getBindLocalPoint() const
{
    return mBindLocalPoint;
}

const MMatrix& offsetCurveBinding::getBindMatrix() const
{
    return mBindMatrix;
}

const MVector& offsetCurveBinding::getTangent() const
{
    return mTangent;
}

const MVector& offsetCurveBinding::getNormal() const
{
    return mNormal;
}

const MVector& offsetCurveBinding::getBinormal() const
{
    return mBinormal;
}

double offsetCurveBinding::getCurvature() const
{
    return mCurvature;
}

int offsetCurveBinding::getSegmentIndex() const
{
    return mSegmentIndex;
}

bool offsetCurveBinding::isJunction() const
{
    return mIsJunction;
}

double offsetCurveBinding::getJunctionRadius() const
{
    return mJunctionRadius;
}

double offsetCurveBinding::getSegmentLength() const
{
    return mSegmentLength;
}