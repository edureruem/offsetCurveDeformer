/**
 * offsetCurveDeformerNode.h
 * Maya 2020용 Offset Curve Deformer 노드
 * 소니 특허(US8400455) 기반 구현
 */

#pragma once

// Maya 헤더들
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

// C++ 표준 라이브러리
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
    
    // 🚀 1단계: 기본 동작 복구 함수들
    MStatus applyBasicDeformation(MPointArray& points, 
                                 const std::vector<MDagPath>& curves);
    double calculateDistanceToCurve(const MPoint& point, const MDagPath& curve);
    MVector calculateBasicOffset(const MPoint& point, const MDagPath& curve);
    
    // 🚨 Maya 권장 방식: compute() 오버라이드하지 않음
    // Maya가 자동으로 compute()에서 deform()을 호출
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
    
    // 🔴 추가: 에러 처리 및 검증 메서드들
    bool validateInputData(MDataBlock& dataBlock);
    bool checkMemoryStatus();
    bool checkGPUStatus();
    bool validateOutputData(MItGeometry& iter);
    
    // 🔴 추가: 안전한 메모리 관리
    void cleanupResources();
    bool initializeResources();
    
public:
    // 노드 속성
    static MTypeId id;
    static const MString nodeName;

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

    // ✅ 추가: 새로운 시스템들을 위한 속성들
    
    // Bind Remapping 속성
    static MObject aEnableBindRemapping;    // Bind Remapping 활성화
    static MObject aBindRemappingStrength;  // Bind Remapping 강도
    
    // Pose Space Deformation 속성
    static MObject aEnablePoseSpaceDeform;  // Pose Space Deformation 활성화
    static MObject aSkeletonJoints;         // 스켈레톤 관절들
    static MObject aJointWeights;           // 관절별 가중치
    
    // Adaptive Subdivision 속성
    static MObject aEnableAdaptiveSubdiv;   // Adaptive Subdivision 활성화
    static MObject aCurvatureThreshold;     // 곡률 임계값
    static MObject aMaxSegmentLength;       // 최대 세그먼트 길이
    static MObject aMinSegmentLength;       // 최소 세그먼트 길이
    
    // ✅ 추가: 가중치 맵 관련 속성들
    static MObject aEnableWeightMaps;       // 가중치 맵 시스템 활성화
    static MObject aWeightMapStrength;      // 가중치 맵 강도
    static MObject aWeightMapTransform;     // 가중치 맵 변환 행렬
    static MObject aWeightMapFalloff;       // 가중치 맵 폴오프
    
    // ✅ 추가: 영향력 혼합 관련 속성들
    static MObject aEnableInfluenceBlending;    // 영향력 혼합 시스템 활성화
    static MObject aBlendingQuality;            // 혼합 품질 (0.0 ~ 1.0)
    static MObject aConflictResolution;         // 충돌 해결 방식
    static MObject aMaxInfluenceDistance;       // 최대 영향 거리
    
    // ✅ 추가: 공간적 보간 관련 속성들
    static MObject aEnableSpatialInterpolation; // 공간적 보간 시스템 활성화
    static MObject aInterpolationQuality;       // 보간 품질 (0.0 ~ 1.0)
    static MObject aSmoothnessFactor;           // 부드러움 계수 (0.0 ~ 1.0)
    static MObject aMaxInterpolationSteps;     // 최대 보간 단계 수
    static MObject aInfluenceRadius;            // 영향 반경

private:
    // Offset Curve 알고리즘 인스턴스 (컴포지션 패턴 사용)
    std::unique_ptr<offsetCurveAlgorithm> mAlgorithm;
    
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