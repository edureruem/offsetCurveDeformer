/**
 * offsetCurveControlParams.cpp
 * 간단한 기본 제어 파라미터 구현
 */

#include "offsetCurveControlParams.h"

// 기본 생성자
offsetCurveControlParams::offsetCurveControlParams()
{
    resetToDefaults();
}

// 기본 소멸자
offsetCurveControlParams::~offsetCurveControlParams()
{
}

// 기본값으로 초기화
void offsetCurveControlParams::resetToDefaults()
{
    mVolumeStrength = 0.0;
    mSlideEffect = 0.0;
    mRotationDistribution = 0.0;
    mScaleDistribution = 1.0;
    mTwistDistribution = 0.0;
    mAxialSliding = 0.0;
    mNormalOffset = 0.0;
    mEnablePoseBlending = false;
    mPoseWeight = 0.5;
}

// 볼륨 강도 설정
void offsetCurveControlParams::setVolumeStrength(double strength)
{
    mVolumeStrength = strength;
}

// 슬라이드 효과 설정
void offsetCurveControlParams::setSlideEffect(double effect)
{
    mSlideEffect = effect;
}

// 회전 분포 설정
void offsetCurveControlParams::setRotationDistribution(double distribution)
{
    mRotationDistribution = distribution;
}

// 스케일 분포 설정
void offsetCurveControlParams::setScaleDistribution(double distribution)
{
    mScaleDistribution = distribution;
}

// 비틀림 분포 설정
void offsetCurveControlParams::setTwistDistribution(double distribution)
{
    mTwistDistribution = distribution;
}

// 축방향 슬라이딩 설정
void offsetCurveControlParams::setAxialSliding(double sliding)
{
    mAxialSliding = sliding;
}

// 법선 오프셋 설정
void offsetCurveControlParams::setNormalOffset(double offset)
{
    mNormalOffset = offset;
}

// 포즈 블렌딩 활성화 설정
void offsetCurveControlParams::setEnablePoseBlending(bool enable)
{
    mEnablePoseBlending = enable;
}

// 포즈 가중치 설정
void offsetCurveControlParams::setPoseWeight(double weight)
{
    mPoseWeight = std::max(0.0, std::min(1.0, weight));
}

// 값 접근자 메서드들
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

bool offsetCurveControlParams::getEnablePoseBlending() const
{
    return mEnablePoseBlending;
}

double offsetCurveControlParams::getPoseWeight() const
{
    return mPoseWeight;
}
