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
#include <maya/MFnDependencyNode.h>
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
#include "offsetCurveControlParams.h"  // 아티스트 제어 파라미터

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
    
    // 추가 메서드 선언 (Maya 2020 호환성)
    virtual MStatus compute(const MPlug& plug, MDataBlock& dataBlock);
    MStatus updateParameters(MDataBlock& dataBlock);
    MStatus rebindDeformer(MDataBlock& dataBlock, MItGeometry& iter);
    MStatus getCurvesFromInputs(MDataBlock& dataBlock, std::vector<MDagPath>& curves);
    MStatus getPoseTargetMesh(MDataBlock& dataBlock, MPointArray& points);
    MStatus initializeBinding(MDataBlock& dataBlock, MItGeometry& iter);
    
    // 다른 메서드
    virtual MStatus connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc);
    virtual MStatus connectionBroken(const MPlug& plug, const MPlug& otherPlug, bool asSrc);
    
    // 스킨 바인딩
    MStatus bindSkin();
    
    // 특허 기술 관련 메서드
    MStatus applyVolumePreservationCorrection(MPointArray& points, 
                                            const offsetCurveControlParams& params);
    
public:
    // 노드 속성
    static MTypeId id;
    static MString nodeName;

    // 기본 속성
    static MObject aOffsetMode;          // 오프셋 모드 (아크/B-스플라인) - 사용자 직접 제어
    static MObject aOffsetCurves;        // 오프셋 곡선들
    static MObject aCurvesData;          // 곡선 데이터
    static MObject aBindPose;            // 바인드 포즈
    static MObject aMaxInfluences;       // 최대 영향 개수
    static MObject aFalloffRadius;       // 영향 반경
    static MObject aRebindMesh;          // 메시 리바인드
    static MObject aRebindCurves;        // 곡선 리바인드
    static MObject aUseParallel;         // 병렬 처리
    static MObject aDebugDisplay;        // 디버그 표시
    
    // 아티스트 제어 속성
    static MObject aVolumeStrength;         // 볼륨 보존 강도
    static MObject aSlideEffect;            // 슬라이딩 효과 조절
    static MObject aRotationDistribution;   // 회전 분포
    static MObject aScaleDistribution;      // 스케일 분포
    static MObject aTwistDistribution;      // 꼬임 분포
    static MObject aAxialSliding;           // 축 방향 슬라이딩
    
    // 포즈 타겟 속성
    static MObject aEnablePoseBlend;        // 포즈 블렌딩 활성화
    static MObject aPoseTarget;             // 포즈 타겟
    static MObject aPoseWeight;             // 포즈 가중치

private:
    // Offset Curve 알고리즘 인스턴스 (컴포지션 패턴 사용)
    offsetCurveAlgorithm* mAlgorithm;
    
    // 바인딩 상태 관리 (컴포지션 데이터에서 관리)
    bool mNeedsRebind;
    
    // Maya 2020 호환성을 위한 데이터 멤버
    MPointArray mOriginalPoints;
    MPointArray mPoseTargetPoints;
    bool mBindingInitialized;
    
    // 레거시 호환성을 위한 캐시 (점진적으로 제거 예정)
    std::vector<MDagPath> mCurvePaths;
};

// offsetCurveControlParams는 offsetCurveAlgorithm.h에서 정의됨