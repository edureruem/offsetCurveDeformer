/**
 * offsetCurveControlParams.cpp
 * Offset Curve Deformer의 아티스트 제어 파라미터 클래스 구현
 * 소니 특허(US8400455) 기반 구현
 */

#include "offsetCurveControlParams.h"

// 생성자
offsetCurveControlParams::offsetCurveControlParams()
{
    // 기본값 초기화
    resetToDefaults();
}

// 소멸자
offsetCurveControlParams::~offsetCurveControlParams()
{
    // 필요한 정리 작업 (현재 없음)
}

// 볼륨 및 슬라이딩 설정 메서드
void offsetCurveControlParams::setVolumeStrength(double strength)
{
    mVolumeStrength = strength;
}

void offsetCurveControlParams::setSlideEffect(double effect)
{
    mSlideEffect = effect;
}

// 분포 설정 메서드
void offsetCurveControlParams::setRotationDistribution(double distribution)
{
    mRotationDistribution = distribution;
}

void offsetCurveControlParams::setScaleDistribution(double distribution)
{
    mScaleDistribution = distribution;
}

void offsetCurveControlParams::setTwistDistribution(double distribution)
{
    mTwistDistribution = distribution;
}

// 추가 변형 설정 메서드
void offsetCurveControlParams::setAxialSliding(double sliding)
{
    mAxialSliding = sliding;
}

void offsetCurveControlParams::setNormalOffset(double offset)
{
    mNormalOffset = offset;
}

// 포즈 블렌딩 설정 메서드
void offsetCurveControlParams::setEnablePoseBlending(bool enable)
{
    mEnablePoseBlending = enable;
}

void offsetCurveControlParams::setPoseWeight(double weight)
{
    mPoseWeight = weight;
}

// 값 접근자 메서드
double offsetCurveControlParams::getVolumeStrength() const
{
    return mVolumeStrength;
}

double offsetCurveControlParams::getSlideEffect() const
{
    return mSlideEffect;
}

double offsetCurveControlParams::getRotationDistribution() const
{
    return mRotationDistribution;
}

double offsetCurveControlParams::getScaleDistribution() const
{
    return mScaleDistribution;
}

double offsetCurveControlParams::getTwistDistribution() const
{
    return mTwistDistribution;
}

double offsetCurveControlParams::getAxialSliding() const
{
    return mAxialSliding;
}

double offsetCurveControlParams::getNormalOffset() const
{
    return mNormalOffset;
}

bool offsetCurveControlParams::isPoseBlendingEnabled() const
{
    return mEnablePoseBlending;
}

double offsetCurveControlParams::getPoseWeight() const
{
    return mPoseWeight;
}

// 기본값 초기화 메서드
void offsetCurveControlParams::resetToDefaults()
{
    mVolumeStrength = 1.0;       // 기본 볼륨 강도 (1.0 = 100%)
    mSlideEffect = 0.0;          // 기본 슬라이딩 없음
    mRotationDistribution = 0.5;  // 균등 회전 분포
    mScaleDistribution = 0.5;     // 균등 스케일 분포
    mTwistDistribution = 0.5;     // 균등 꼬임 분포
    mAxialSliding = 0.0;         // 기본 축 방향 슬라이딩 없음
    mNormalOffset = 1.0;         // 기본 법선 오프셋 (1.0 = 100%)
    mEnablePoseBlending = false;  // 기본 포즈 블렌딩 비활성화
    mPoseWeight = 0.0;           // 기본 포즈 가중치 없음
}
