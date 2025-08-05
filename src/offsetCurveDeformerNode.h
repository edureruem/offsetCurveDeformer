/**
 * offsetCurveDeformerNode.h
 * Maya 2020용 Offset Curve Deformer 노드
 * 소니 특허(US8400455) 기반 구현
 */

#pragma once

#include <maya/MPxDeformerNode.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MFnMesh.h>
#include <maya/MItGeometry.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <vector>
#include <map>

#include "offsetCurveAlgorithm.h"
#include "offsetCurveData.h"

class offsetCurveDeformerNode : public MPxDeformerNode {
public:
    offsetCurveDeformerNode();
    virtual ~offsetCurveDeformerNode();

    // 필수 오버라이드 메서드
    static void* creator();
    static MStatus initialize();
    
    // 변형 메서드
    virtual MStatus deform(MDataBlock& dataBlock,
                         MItGeometry& iter,
                         const MMatrix& matrix,
                         unsigned int multiIndex);
    
    // 다른 메서드
    virtual MStatus connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc);
    virtual MStatus connectionBroken(const MPlug& plug, const MPlug& otherPlug, bool asSrc);
    
    // 스킨 바인딩
    MStatus bindSkin();
    
public:
    // 노드 속성
    static MTypeId id;
    static MString nodeName;

    // 기본 속성
    static MObject aCurveData;           // 영향 곡선 정보
    static MObject aCurveMessage;        // 곡선 메시지 연결
    static MObject aOffsetMode;          // 오프셋 모드 (아크/B-스플라인)
    static MObject aBindPoseMatrix;      // 바인드 포즈 매트릭스
    static MObject aBindMode;            // 바인드 모드 (자동/수동)
    static MObject aMaxInfluences;       // 최대 영향 개수
    static MObject aFalloffRadius;       // 영향 반경
    
    // 볼륨 및 변형 제어 속성
    static MObject aVolumeStrength;      // 볼륨 보존 강도
    static MObject aSlideEffect;         // 슬라이딩 효과 조절
    static MObject aRebind;              // 재바인딩 트리거
    static MObject aUseParallelCompute;  // 병렬 계산 활성화
    
    // 고급 아티스트 제어 속성
    static MObject aRotationDistribution;   // 회전 분포
    static MObject aScaleDistribution;      // 스케일 분포
    static MObject aTwistDistribution;      // 꼬임 분포
    static MObject aAxialSliding;           // 축 방향 슬라이딩
    static MObject aNormalOffsetStrength;   // 법선 오프셋 강도
    
    // 포즈 기반 블렌딩 속성
    static MObject aEnablePoseBlending;     // 포즈 블렌딩 활성화
    static MObject aPoseTarget;             // 포즈 타겟
    static MObject aPoseWeight;             // 포즈 가중치

private:
    // Offset Curve 알고리즘 인스턴스
    offsetCurveAlgorithm mOffsetCurveAlgorithm;
    
    // 바인딩 상태 관리
    bool mNeedsRebind;
    bool mIsInitialBindDone;
    
    // 작업 캐시
    std::vector<MDagPath> mInfluenceCurves;
    MPointArray mOriginalPoints;
};

// offsetCurveDeformerNode.h에 추가
// 기존 include 문 아래에 배치

// offsetCurveControlParams 클래스 선언
class offsetCurveControlParams {
public:
    offsetCurveControlParams();
    ~offsetCurveControlParams();

    void setVolumeStrength(double strength);
    void setSlideEffect(double effect);
    void setRotationDistribution(double distribution);
    void setScaleDistribution(double distribution);
    void setTwistDistribution(double distribution);
    void setAxialSliding(double sliding);
    void setNormalOffset(double offset);
    void setEnablePoseBlending(bool enable);
    void setPoseWeight(double weight);

    double getVolumeStrength() const;
    double getSlideEffect() const;
    double getRotationDistribution() const;
    double getScaleDistribution() const;
    double getTwistDistribution() const;
    double getAxialSliding() const;
    double getNormalOffset() const;
    bool isPoseBlendingEnabled() const;
    double getPoseWeight() const;

    void resetToDefaults();

private:
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

// offsetCurveDeformerNode 클래스 선언 계속...