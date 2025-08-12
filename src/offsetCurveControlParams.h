/**
 * offsetCurveControlParams.h
 * 간단한 기본 제어 파라미터 헤더
 */

#ifndef OFFSETCURVECONTROLPARAMS_H
#define OFFSETCURVECONTROLPARAMS_H

// 기본 제어 파라미터 클래스
class offsetCurveControlParams {
public:
    offsetCurveControlParams();
    ~offsetCurveControlParams();
    
    // 기본값으로 초기화
    void resetToDefaults();
    
    // 설정 메서드들
    void setVolumeStrength(double strength);
    void setSlideEffect(double effect);
    void setRotationDistribution(double distribution);
    void setScaleDistribution(double distribution);
    void setTwistDistribution(double distribution);
    void setAxialSliding(double sliding);
    void setNormalOffset(double offset);
    void setEnablePoseBlending(bool enable);
    void setPoseWeight(double weight);
    
    // 접근자 메서드들
    double getVolumeStrength() const;
    double getSlideEffect() const;
    double getRotationDistribution() const;
    double getScaleDistribution() const;
    double getTwistDistribution() const;
    double getAxialSliding() const;
    double getNormalOffset() const;
    bool getEnablePoseBlending() const;
    double getPoseWeight() const;
    
private:
    // 기본 파라미터들
    double mVolumeStrength;
    double mSlideEffect;
    double mRotationDistribution;
    double mScaleDistribution;
    double mTwistDistribution;
    double mAxialSliding;
    double mNormalOffset;
    bool mEnablePoseBlending;
    double mPoseWeight;
};

#endif // OFFSETCURVECONTROLPARAMS_H