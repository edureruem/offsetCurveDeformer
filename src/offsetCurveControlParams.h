/**
 * offsetCurveControlParams.h
 * Offset Curve Deformer의 아티스트 제어 파라미터 클래스
 * 소니 특허(US8400455) 기반 구현
 */

#ifndef OFFSETCURVECONTROLPARAMS_H
#define OFFSETCURVECONTROLPARAMS_H

class offsetCurveControlParams {
public:
    offsetCurveControlParams();
    ~offsetCurveControlParams();

    // 볼륨 및 슬라이딩 설정
    void setVolumeStrength(double strength);
    void setSlideEffect(double effect);

    // 분포 설정
    void setRotationDistribution(double distribution);
    void setScaleDistribution(double distribution);
    void setTwistDistribution(double distribution);

    // 추가 변형 설정
    void setAxialSliding(double sliding);
    void setNormalOffset(double offset);

    // 포즈 블렌딩 설정
    void setEnablePoseBlending(bool enable);
    void setPoseWeight(double weight);

    // 값 접근자
    double getVolumeStrength() const;
    double getSlideEffect() const;
    double getRotationDistribution() const;
    double getScaleDistribution() const;
    double getTwistDistribution() const;
    double getAxialSliding() const;
    double getNormalOffset() const;
    bool isPoseBlendingEnabled() const;
    double getPoseWeight() const;

    // 기본값 초기화
    void resetToDefaults();

private:
    // 볼륨 및 슬라이딩 속성
    double mVolumeStrength;       // 볼륨 보존 강도 (0.0~2.0)
    double mSlideEffect;          // 슬라이딩 효과 (-1.0~1.0)

    // 분포 속성
    double mRotationDistribution; // 회전 분포 (0.0~1.0)
    double mScaleDistribution;    // 스케일 분포 (0.0~1.0)
    double mTwistDistribution;    // 꼬임 분포 (0.0~1.0)

    // 추가 변형 속성
    double mAxialSliding;         // 축 방향 슬라이딩 (-1.0~1.0)
    double mNormalOffset;         // 법선 방향 오프셋 (0.0~2.0)

    // 포즈 블렌딩 속성
    bool mEnablePoseBlending;     // 포즈 블렌딩 활성화 여부
    double mPoseWeight;           // 포즈 가중치 (0.0~1.0)
};

#endif // OFFSETCURVECONTROLPARAMS_H