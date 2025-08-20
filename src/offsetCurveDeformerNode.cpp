/**
 * offsetCurveDeformerNode.cpp
 * Maya 2020용 Offset Curve Deformer 노드 구현
 */

#include "offsetCurveDeformerNode.h"
#include "offsetCurveAlgorithm.h"
#include "offsetCurveControlParams.h"  // 별도 파일에서 구현된 클래스 사용
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MDagPath.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
#include <maya/MSelectionList.h>
#include <maya/MNodeMessage.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <float.h>

// Maya 상태 체크 매크로는 이미 Maya 헤더에 정의되어 있음

// 중복 구현 제거 - offsetCurveControlParams.cpp에서 구현됨

// 노드 ID 및 이름
MTypeId offsetCurveDeformerNode::id(0x00134); // 임시 ID - 실제 등록 ID로 변경 필요
const MString offsetCurveDeformerNode::nodeName = "offsetCurveDeformer";

// 노드 속성 초기화
MObject offsetCurveDeformerNode::aOffsetMode;
MObject offsetCurveDeformerNode::aOffsetCurves;
MObject offsetCurveDeformerNode::aCurvesData;
MObject offsetCurveDeformerNode::aBindPose;
MObject offsetCurveDeformerNode::aFalloffRadius;
MObject offsetCurveDeformerNode::aMaxInfluences;
MObject offsetCurveDeformerNode::aRebindMesh;
MObject offsetCurveDeformerNode::aRebindCurves;
MObject offsetCurveDeformerNode::aUseParallel;
MObject offsetCurveDeformerNode::aDebugDisplay;

    // 추가: influenceCurve 관련 어트리뷰트 변수들
MObject offsetCurveDeformerNode::aInfluenceCurve;
MObject offsetCurveDeformerNode::aInfluenceCurveData;
MObject offsetCurveDeformerNode::aInfluenceCurveGroupId;

// cvwrap 방식의 바인딩 데이터 속성
MObject offsetCurveDeformerNode::aBindData;
MObject offsetCurveDeformerNode::aSampleComponents;
MObject offsetCurveDeformerNode::aSampleWeights;
MObject offsetCurveDeformerNode::aTriangleVerts;
MObject offsetCurveDeformerNode::aBarycentricWeights;
MObject offsetCurveDeformerNode::aBindMatrix;

// 아티스트 제어 속성
MObject offsetCurveDeformerNode::aVolumeStrength;
MObject offsetCurveDeformerNode::aSlideEffect;
MObject offsetCurveDeformerNode::aRotationDistribution;
MObject offsetCurveDeformerNode::aScaleDistribution;
MObject offsetCurveDeformerNode::aTwistDistribution;
MObject offsetCurveDeformerNode::aAxialSliding;

// 포즈 타겟 속성
MObject offsetCurveDeformerNode::aEnablePoseBlend;
MObject offsetCurveDeformerNode::aPoseTarget;
MObject offsetCurveDeformerNode::aPoseWeight;

// 생성자
offsetCurveDeformerNode::offsetCurveDeformerNode() 
    : mNeedsRebind(true), mBindingInitialized(false)
{
    try {
        mAlgorithm = std::make_unique<offsetCurveAlgorithm>();
        if (!mAlgorithm) {
            MGlobal::displayError("Failed to create algorithm in constructor");
        }
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Constructor error: ") + e.what());
        mAlgorithm.reset();
    }
}

// 소멸자
offsetCurveDeformerNode::~offsetCurveDeformerNode() 
{
    try {
        // 리소스 정리
        cleanupResources();
        
        // mAlgorithm은 std::unique_ptr이므로 자동 소멸됨
        mAlgorithm.reset();
        
        // 포인트 배열 정리
        mOriginalPoints.clear();
        mPoseTargetPoints.clear();
        mCurvePaths.clear();
        
    } catch (...) {
        // 소멸자에서는 예외를 던지지 않음
        MGlobal::displayError("Error in destructor - ignored for safety");
    }
}

// 노드 생성자 (팩토리 메서드)
void* offsetCurveDeformerNode::creator() 
{
    try {
    return new offsetCurveDeformerNode();
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Failed to create node: ") + e.what());
        return nullptr;
    } catch (...) {
        MGlobal::displayError("Failed to create node: Unknown error");
        return nullptr;
    }
}

// postConstructor 구현 (cvwrap 방식)
void offsetCurveDeformerNode::postConstructor()
{
    MPxDeformerNode::postConstructor();
    
    MStatus status = MS::kSuccess;
    MObject obj = thisMObject();
    onDeleteCallbackId = MNodeMessage::addNodeAboutToDeleteCallback(obj, aboutToDeleteCB, NULL, &status);
    
    if (!status) {
        MGlobal::displayWarning("Failed to add node delete callback");
    }
}

// setDependentsDirty 구현 (cvwrap 방식)
MStatus offsetCurveDeformerNode::setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs) 
{
    // Extract the geom index from the dirty plug and set the dirty flag so we know that we need to
    // re-read the binding data.
    if (plugBeingDirtied.isElement()) {
        MPlug parent = plugBeingDirtied.array().parent();
        if (parent == aBindData) {
            unsigned int geomIndex = parent.logicalIndex();
            dirty_[geomIndex] = true;
        }
    }
    return MS::kSuccess;
}

// aboutToDeleteCB 콜백 구현 (cvwrap 방식)
void offsetCurveDeformerNode::aboutToDeleteCB(MObject &node, MDGModifier &modifier, void *clientData)
{
    // cvwrap과 동일한 방식으로 연결된 바인드 메시 삭제
    // 현재 OCD에서는 별도 처리 없음
}

// 노드 초기화 (cvwrap 방식)
MStatus offsetCurveDeformerNode::initialize() 
{
    MStatus status;
    
    // 속성 팩토리 (cvwrap 방식)
    MFnCompoundAttribute cAttr;
    MFnMatrixAttribute mAttr;
    MFnMessageAttribute meAttr;
    MFnTypedAttribute tAttr;
    MFnNumericAttribute nAttr;
    MFnEnumAttribute eAttr;
    
    // 1. 오프셋 모드 설정 (Enum)
    aOffsetMode = eAttr.create("offsetMode", "om", 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    eAttr.addField("Arc Segment", 0);
    eAttr.addField("B-Spline", 1);
    eAttr.setKeyable(true);
    eAttr.setStorable(true);
    
    // 2. 오프셋 곡선들 (메시지 배열)
    aOffsetCurves = meAttr.create("offsetCurves", "oc", &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    meAttr.setArray(true);
    meAttr.setStorable(false);
    meAttr.setConnectable(true);
    
    // 추가: 3. influenceCurve 관련 어트리뷰트들 (Maya 표준 input과 동일한 구조)
    // 3.1. influenceCurveData: nurbsCurve 데이터 (하위 속성)
    aInfluenceCurveData = tAttr.create("influenceCurveData", "icd", MFnData::kNurbsCurve, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    tAttr.setStorable(false);
    tAttr.setConnectable(true);
    
    // 3.2. influenceCurveGroupId: 그룹 ID (하위 속성)
    aInfluenceCurveGroupId = nAttr.create("influenceCurveGroupId", "icgi", MFnNumericData::kLong, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setStorable(false);
    nAttr.setConnectable(false);
    
    // 3.3. influenceCurve: 복합 속성 (Maya 표준 input과 동일한 구조)
    aInfluenceCurve = cAttr.create("influenceCurve", "ic", &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 복합 속성에 하위 속성들 추가 (Maya 표준 방식)
    status = cAttr.addChild(aInfluenceCurveData);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = cAttr.addChild(aInfluenceCurveGroupId);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 복합 속성 설정 (Maya 표준 input과 동일)
    cAttr.setStorable(false);
    cAttr.setConnectable(true);
            cAttr.setArray(true);  // Maya 표준: 다중 곡선 지원
        cAttr.setUsesArrayDataBuilder(true);  // Maya 표준: 배열 빌더 사용
    
    // 4. 바인딩 데이터 (cvwrap 방식)
    aSampleComponents = tAttr.create("sampleComponents", "sc", MFnData::kIntArray, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    tAttr.setArray(true);

    aSampleWeights = tAttr.create("sampleWeights", "sw", MFnData::kDoubleArray, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    tAttr.setArray(true);

    aTriangleVerts = nAttr.create("triangleVerts", "tv", MFnNumericData::k3Int, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setArray(true);

    aBarycentricWeights = nAttr.create("barycentricWeights", "bw", MFnNumericData::k3Float, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setArray(true);

    aBindMatrix = mAttr.create("bindMatrix", "bm", MFnMatrixAttribute::kDouble, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    mAttr.setDefault(MMatrix::identity);
    mAttr.setArray(true);

    // 바인딩 데이터 복합 속성
    aBindData = cAttr.create("bindData", "bd", &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    cAttr.setArray(true);
    cAttr.addChild(aSampleComponents);
    cAttr.addChild(aSampleWeights);
    cAttr.addChild(aTriangleVerts);
    cAttr.addChild(aBarycentricWeights);
    cAttr.addChild(aBindMatrix);

    // 5. 바인딩 및 제어 매개변수
    aFalloffRadius = nAttr.create("falloffRadius", "fr", MFnNumericData::kDouble, 10.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.001);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aMaxInfluences = nAttr.create("maxInfluences", "mi", MFnNumericData::kInt, 4, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(1);
    nAttr.setMax(10);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    // 5. 리바인드 트리거
    aRebindMesh = nAttr.create("rebindMesh", "rbm", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(false);
    
    aRebindCurves = nAttr.create("rebindCurves", "rbc", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(false);
    
    // 6. 아티스트 제어 속성
    aVolumeStrength = nAttr.create("volumeStrength", "vs", MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(5.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aSlideEffect = nAttr.create("slideEffect", "sle", MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-2.0);
    nAttr.setMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aRotationDistribution = nAttr.create("rotationDistribution", "rd", MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aScaleDistribution = nAttr.create("scaleDistribution", "sd", MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aTwistDistribution = nAttr.create("twistDistribution", "td", MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aAxialSliding = nAttr.create("axialSliding", "as", MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-1.0);
    nAttr.setMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    // 7. 포즈 블렌딩
    aEnablePoseBlend = nAttr.create("enablePoseBlend", "epb", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aPoseTarget = meAttr.create("poseTarget", "pt", &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    meAttr.setStorable(false);
    meAttr.setConnectable(true);
    
    aPoseWeight = nAttr.create("poseWeight", "pw", MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    // 8. 추가 설정
    aUseParallel = nAttr.create("useParallelComputation", "upc", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aDebugDisplay = nAttr.create("debugDisplay", "dbg", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    // 9. 속성 추가
    // 바인딩 데이터 속성 추가 (cvwrap 방식)
    status = addAttribute(aBindData);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aOffsetMode);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aOffsetCurves);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aInfluenceCurve); // 복합 속성 추가
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aFalloffRadius);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aMaxInfluences);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aRebindMesh);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aRebindCurves);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aVolumeStrength);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aSlideEffect);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aRotationDistribution);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aScaleDistribution);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aTwistDistribution);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aAxialSliding);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aEnablePoseBlend);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aPoseTarget);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aPoseWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aUseParallel);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aDebugDisplay);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 10. 속성 영향 설정
    // 바인딩 데이터 속성 영향 설정 (cvwrap 방식)
    status = attributeAffects(aSampleComponents, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aSampleWeights, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aTriangleVerts, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aBarycentricWeights, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aBindMatrix, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = attributeAffects(aOffsetMode, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aOffsetCurves, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aInfluenceCurve, outputGeom); // 복합 속성 영향 설정
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aFalloffRadius, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aMaxInfluences, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aRebindMesh, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aRebindCurves, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aVolumeStrength, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aSlideEffect, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aRotationDistribution, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aScaleDistribution, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aTwistDistribution, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aAxialSliding, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aEnablePoseBlend, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aPoseTarget, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aPoseWeight, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aUseParallel, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aDebugDisplay, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 11. 초기화 완료 메시지
    MGlobal::displayInfo("Offset Curve Deformer Node attributes initialized successfully");
    
            return status;
}

// 🚨 Maya 권장 방식: compute() 오버라이드하지 않음
// Maya가 자동으로 compute()에서 deform()을 호출
// 
// 참고: Maya 공식 문서에 따르면:
// "In general, to derive the full benefit of the Maya deformer base class, 
//  it is suggested that you do not write your own compute() method. 
//  Instead, write the deform() method, which is called by the MPxDeformerNode's compute() method."
//
// 따라서 compute()를 제거하고 deform()만 구현하여 Maya의 기본 동작을 활용

// 디포머 메서드 (cvwrap 방식)
MStatus offsetCurveDeformerNode::deform(MDataBlock& data, MItGeometry& iter, 
                                        const MMatrix& mat, unsigned int mIndex)
{
    MStatus status;
    
    try {
        // 1. 기본 검증 (cvwrap 방식)
        if (!validateInputData(data)) {
            MGlobal::displayError("Input data validation failed");
            return MS::kFailure;
        }
        
        // 2. 바인딩 데이터 확인
        MArrayDataHandle bindDataHandle = data.inputArrayValue(aBindData, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        if (bindDataHandle.elementCount() == 0) {
            // 바인딩 데이터가 없으면 초기 바인딩 수행
            status = initializeBinding(data, iter);
            CHECK_MSTATUS_AND_RETURN_IT(status);
        }
        
        // 3. 곡선 데이터 가져오기
        std::vector<MDagPath> curves;
        status = getCurvesFromInputs(data, curves);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        if (curves.empty()) {
            MGlobal::displayWarning("No curves connected to the deformer.");
            return MS::kFailure;
        }
        
        // 4. 메시 포인트 가져오기
        MPointArray points;
        status = iter.allPositions(points);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        // 5. cvwrap 방식의 변형 적용 (단일 스레드)
        status = applyDeformation(points, curves, data, mIndex);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        // 6. 결과를 메시에 적용
        iter.setAllPositions(points);
        
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Deformation error: ") + e.what());
        return MS::kFailure;
    }
}

// cvwrap 방식의 단일 스레드 변형 적용
MStatus offsetCurveDeformerNode::applyDeformation(MPointArray& points, 
                                                  const std::vector<MDagPath>& curves,
                                                  MDataBlock& data, unsigned int mIndex)
{
    MStatus status;
    
    try {
        // 기존 OCD 알고리즘으로 변형 적용
        if (mAlgorithm) {
            // 컨트롤 파라미터 생성 (임시)
            offsetCurveControlParams controlParams;
            status = mAlgorithm->computeDeformation(points, controlParams);
            CHECK_MSTATUS_AND_RETURN_IT(status);
        }
        
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Deformation error: ") + e.what());
        return MS::kFailure;
    }
}

// 기존 함수들은 제거됨 - applyDeformation으로 통합

// 🚀 1.2: 기본 변형 함수 추가 - 가장 단순한 변형부터 시작
MStatus offsetCurveDeformerNode::applyBasicDeformation(MPointArray& points, 
                                                      const std::vector<MDagPath>& curves) {
    try {
        MGlobal::displayInfo("Applying basic deformation...");
        
        // 각 정점에 대해 기본 변형 적용
        for (unsigned int i = 0; i < points.length(); i++) {
            MPoint& point = points[i];
            
            // 각 곡선에 대한 기본 오프셋 계산
            for (const auto& curve : curves) {
                // 핵심: 단순한 거리 기반 변형
                double distance = calculateDistanceToCurve(point, curve);
                if (distance < 5.0) { // 기본 영향 반경
                    MVector offset = calculateBasicOffset(point, curve);
                    point += offset * 0.1; // 기본 강도 (10%)
                }
            }
        }
        
        MGlobal::displayInfo("Basic deformation completed successfully");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Basic deformation error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown error in basic deformation");
        return MS::kFailure;
    }
}

// 🚀 1.3: 헬퍼 함수들 추가 - 단순화된 기본 계산 함수들
double offsetCurveDeformerNode::calculateDistanceToCurve(const MPoint& point, const MDagPath& curve) {
    try {
        // 핵심: 단순한 거리 계산 - 곡선의 첫 번째 CV와의 거리
        MFnNurbsCurve curveFn(curve);
        
        // 곡선의 첫 번째 CV 위치 가져오기
        MPoint firstCV;
        curveFn.getCV(0, firstCV, MSpace::kWorld);
        
        // 정점에서 첫 번째 CV까지의 거리
        double distance = point.distanceTo(firstCV);
        return distance;
        
    } catch (...) {
        return 1000.0; // 오류 시 기본값
    }
}

MVector offsetCurveDeformerNode::calculateBasicOffset(const MPoint& point, const MDagPath& curve) {
    try {
        // 핵심: 단순한 오프셋 벡터 - Y축 방향으로 기본 변형
        // 복잡한 곡선 계산 대신 기본 방향 사용
        
        // 정점에서 곡선의 첫 번째 CV까지의 방향
        MFnNurbsCurve curveFn(curve);
        MPoint firstCV;
        curveFn.getCV(0, firstCV, MSpace::kWorld);
        
        MVector direction = firstCV - point;
        if (direction.length() > 0.001) {
            direction.normalize();
            return direction;
        }
        
        // 기본 방향 반환
        return MVector(0, 1, 0); // Y축 방향
        
    } catch (...) {
        return MVector(0, 1, 0); // 오류 시 기본값
    }
}

// 초기 바인딩 초기화
MStatus offsetCurveDeformerNode::initializeBinding(MDataBlock& block, MItGeometry& iter)
{
    MStatus status;
    
    try {
        // 알고리즘 유효성 검사
        if (!mAlgorithm) {
            MGlobal::displayError("Algorithm not initialized");
        return MS::kFailure;
    }
    
        // 오프셋 모드 가져오기
        MDataHandle hOffsetMode = block.inputValue(aOffsetMode, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        short offsetMode = hOffsetMode.asShort();
        
        // 오프셋 곡선들 가져오기
        std::vector<MDagPath> curves;
        status = getCurvesFromInputs(block, curves);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        if (curves.empty()) {
            MGlobal::displayWarning("No curves connected to the deformer.");
        return MS::kFailure;
    }
    
        // 메시 점들 가져오기
    MPointArray points;
        status = iter.allPositions(points);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        if (points.length() == 0) {
            MGlobal::displayError("No mesh points found");
        return MS::kFailure;
    }

        // 알고리즘 초기화
        status = mAlgorithm->initialize(points, static_cast<offsetCurveOffsetMode>(offsetMode));
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        // 병렬 계산 설정
        MDataHandle hUseParallel = block.inputValue(aUseParallel, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        bool useParallel = hUseParallel.asBool();
        mAlgorithm->enableParallelComputation(useParallel);
        
        // 곡선 경로 저장 (레거시 호환성)
        mCurvePaths = curves;
        
        // OCD 바인딩 페이즈
        MDataHandle hFalloffRadius = block.inputValue(aFalloffRadius, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        double falloffRadius = hFalloffRadius.asDouble();
        
        MDataHandle hMaxInfluences = block.inputValue(aMaxInfluences, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        int maxInfluences = hMaxInfluences.asInt();
        
        status = mAlgorithm->performBindingPhase(points, curves, falloffRadius, maxInfluences);
    if (status != MS::kSuccess) {
            MGlobal::displayWarning("OCD binding failed");
        return status;
    }

        // 바인딩 완료 플래그 설정
        mNeedsRebind = false;
        mBindingInitialized = true;
    
    return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Binding initialization error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown binding initialization error");
        return MS::kFailure;
    }
}

// 디포머 리바인딩
MStatus offsetCurveDeformerNode::rebindDeformer(MDataBlock& block, MItGeometry& iter)
{
    return initializeBinding(block, iter);
}

// 입력에서 곡선 가져오기
MStatus offsetCurveDeformerNode::getCurvesFromInputs(MDataBlock& block, std::vector<MDagPath>& curves)
{
    MStatus status;
    curves.clear();
    
    try {
        MArrayDataHandle hCurves = block.inputArrayValue(aOffsetCurves, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        unsigned int curveCount = hCurves.elementCount();
        if (curveCount == 0) {
            return MS::kSuccess; // 곡선이 연결되지 않은 경우
        }
        
        for (unsigned int i = 0; i < curveCount; i++) {
            status = hCurves.jumpToArrayElement(i);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            
            MDataHandle hCurve = hCurves.inputValue(&status);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            
            MObject curveObj = hCurve.asNurbsCurve();
            if (!curveObj.isNull()) {
                MDagPath curvePath;
                status = MDagPath::getAPathTo(curveObj, curvePath);
                if (status == MS::kSuccess) {
                    curves.push_back(curvePath);
                }
    } else {
                // 메시지 커넥션으로부터 곡선 찾기
                MFnDependencyNode thisNodeFn(thisMObject());
                MPlug curvePlug = thisNodeFn.findPlug(aOffsetCurves, false);
                if (!curvePlug.isNull()) {
                    curvePlug.selectAncestorLogicalIndex(i);
                    
                    MPlugArray connections;
                    curvePlug.connectedTo(connections, true, false);
                    
                    if (connections.length() > 0) {
                        MObject connectedNode = connections[0].node();
                        
                        if (connectedNode.hasFn(MFn::kNurbsCurve)) {
                            MDagPath curvePath;
                            status = MDagPath::getAPathTo(connectedNode, curvePath);
                            if (status == MS::kSuccess) {
                                curves.push_back(curvePath);
                            }
                        }
                    }
                }
            }
        }
    
    return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Error getting curves: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown error getting curves");
        return MS::kFailure;
    }
}

// 포즈 타겟 메시 가져오기
MStatus offsetCurveDeformerNode::getPoseTargetMesh(MDataBlock& block, MPointArray& points)
{
    MStatus status;
    points.clear();
    
    try {
        MDataHandle hPoseTarget = block.inputValue(aPoseTarget, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        MObject poseObj = hPoseTarget.asMesh();  // 올바른 타입 캐스팅
        
        if (poseObj.isNull()) {
            // 메시지 커넥션으로부터 메시 찾기
            MFnDependencyNode thisNodeFn(thisMObject());
            MPlug posePlug = thisNodeFn.findPlug(aPoseTarget, false);
            
            if (!posePlug.isNull()) {
                MPlugArray connections;
                posePlug.connectedTo(connections, true, false);
                
                if (connections.length() > 0) {
                    MObject connectedNode = connections[0].node();
                    
                    if (connectedNode.hasFn(MFn::kMesh)) {
                        poseObj = connectedNode;
                    }
                }
            }
        }
        
        if (!poseObj.isNull() && poseObj.hasFn(MFn::kMesh)) {
            MFnMesh meshFn(poseObj);
            status = meshFn.getPoints(points);
            CHECK_MSTATUS_AND_RETURN_IT(status);
        }
    
    return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Error getting pose target mesh: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown error getting pose target mesh");
        return MS::kFailure;
    }
}

// 매개변수 업데이트
MStatus offsetCurveDeformerNode::updateParameters(MDataBlock& block)
{
    MStatus status;
    
    try {
        // 알고리즘 유효성 검사
        if (!mAlgorithm) {
            MGlobal::displayError("Algorithm not initialized");
            return MS::kFailure;
        }
        
        // 오프셋 모드 변경 확인
        MDataHandle hOffsetMode = block.inputValue(aOffsetMode, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        short offsetMode = hOffsetMode.asShort();
        
        // 병렬 계산 설정
        MDataHandle hUseParallel = block.inputValue(aUseParallel, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        bool useParallel = hUseParallel.asBool();
        mAlgorithm->enableParallelComputation(useParallel);
        
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Parameter update error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown parameter update error");
        return MS::kFailure;
    }
}

// 특허 기술: 볼륨 보존 보정 (볼륨 손실, 캔디 래퍼 핀칭, 자기교차 방지)
MStatus offsetCurveDeformerNode::applyVolumePreservationCorrection(MPointArray& points, 
                                                         const offsetCurveControlParams& params)
{
    try {
        // 특허에서 언급하는 주요 아티팩트들 해결:
        // 1. 굽힘에서의 볼륨 손실
        // 2. 비틀림에서의 "캔디 래퍼" 핀칭
        // 3. 굽힘 내측에서의 표면 자기교차
        
        if (mOriginalPoints.length() != points.length()) {
            MGlobal::displayWarning("Point count mismatch in volume preservation correction");
            return MS::kFailure;
        }
        
        double volumeStrength = params.getVolumeStrength();
        if (volumeStrength <= 0.0) {
            return MS::kSuccess;
        }
        
        // 각 정점에 대해 볼륨 보존 보정 적용
        for (unsigned int i = 0; i < points.length(); i++) {
            MPoint& currentPoint = points[i];
            const MPoint& originalPoint = mOriginalPoints[i];
            
            // 변형 벡터 계산
            MVector deformationVector = currentPoint - originalPoint;
            double deformationMagnitude = deformationVector.length();
            
            if (deformationMagnitude < 1e-6) {
                continue; // 변형이 거의 없으면 건너뛰기
            }
            
            // 주변 정점들과의 관계를 고려한 볼륨 보존
            // 이는 특허에서 언급하는 "오프셋 곡선이 모델 포인트를 통과한다"는 개념의 구현
            
            // 인근 정점들 찾기 (간단한 구현)
            std::vector<unsigned int> neighborIndices;
            for (unsigned int j = 0; j < points.length(); j++) {
                if (i != j && originalPoint.distanceTo(mOriginalPoints[j]) < 2.0) {
                    neighborIndices.push_back(j);
                }
            }
            
            if (!neighborIndices.empty()) {
                // 인근 정점들의 평균 변형 계산
                MVector averageDeformation(0.0, 0.0, 0.0);
                for (unsigned int neighborIdx : neighborIndices) {
                    if (neighborIdx < points.length() && neighborIdx < mOriginalPoints.length()) {
                        averageDeformation += (points[neighborIdx] - mOriginalPoints[neighborIdx]);
                    }
                }
                averageDeformation /= static_cast<double>(neighborIndices.size());
                
                // 볼륨 보존을 위한 보정 벡터 계산
                MVector correctionVector = (deformationVector - averageDeformation) * volumeStrength * 0.5;
                
                // 자기교차 방지: 내측 굽힘에서 점들이 밀려나도록
                if (correctionVector.length() > deformationMagnitude * 0.1) {
                    correctionVector.normalize();
                    correctionVector *= deformationMagnitude * 0.1;
                }
                
                // 보정 적용
                currentPoint += correctionVector;
            }
        }
        
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Volume preservation correction error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown volume preservation correction error");
        return MS::kFailure;
    }
}

// 연결 생성 시 호출되는 함수
MStatus offsetCurveDeformerNode::connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc)
{
    try {
        // 연결이 생성되었을 때 필요한 처리를 수행
        if (plug.attribute() == aOffsetCurves) {
            // 오프셋 곡선이 연결되었을 때 리바인딩 필요
            mNeedsRebind = true;
            MGlobal::displayInfo("Offset curve connected - rebinding required");
        }
        return MS::kSuccess;
    } catch (...) {
        MGlobal::displayError("Error in connectionMade");
        return MS::kFailure;
    }
}

// 연결 해제 시 호출되는 함수
MStatus offsetCurveDeformerNode::connectionBroken(const MPlug& plug, const MPlug& otherPlug, bool asSrc)
{
    try {
        // 연결이 해제되었을 때 필요한 처리를 수행
        if (plug.attribute() == aOffsetCurves) {
            // 오프셋 곡선이 해제되었을 때 리바인딩 필요
            mNeedsRebind = true;
            MGlobal::displayInfo("Offset curve disconnected - rebinding required");
        }
        return MS::kSuccess;
    } catch (...) {
        MGlobal::displayError("Error in connectionBroken");
        return MS::kFailure;
    }
}

// 🔴 추가: 에러 처리 및 검증 메서드들

bool offsetCurveDeformerNode::validateInputData(MDataBlock& dataBlock)
{
    MStatus status;
    
    try {
        // 1. 엔벨롭 값 확인
        MDataHandle hEnvelope = dataBlock.inputValue(envelope, &status);
        if (!status || hEnvelope.asFloat() < 0.0f || hEnvelope.asFloat() > 1.0f) {
            MGlobal::displayWarning("Invalid envelope value in Offset Curve Deformer");
            return false;
        }
        
        // 2. 입력 메시 확인
        MDataHandle hInput = dataBlock.inputValue(input, &status);
        if (!status) {
            MGlobal::displayError("No input mesh connected to Offset Curve Deformer");
            return false;
        }
        
        // 3. 오프셋 곡선 확인 (선택적)
        MArrayDataHandle hOffsetCurves = dataBlock.inputArrayValue(aOffsetCurves, &status);
        if (!status) {
            MGlobal::displayWarning("Failed to get offset curves data");
            return false;
        }
        
        // 4. 파라미터 범위 검증
        MDataHandle hVolumeStrength = dataBlock.inputValue(aVolumeStrength, &status);
        if (status && (hVolumeStrength.asDouble() < 0.0 || hVolumeStrength.asDouble() > 5.0)) {
            MGlobal::displayWarning("Volume strength out of valid range [0.0, 5.0]");
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Input validation error: ") + e.what());
        return false;
    } catch (...) {
        MGlobal::displayError("Unknown input validation error");
        return false;
    }
}

bool offsetCurveDeformerNode::checkMemoryStatus()
{
    // 시스템 메모리 상태 확인 (크로스 플랫폼 호환성)
    MGlobal::displayInfo("Memory check disabled for cross-platform compatibility");
    return true;
}

bool offsetCurveDeformerNode::checkGPUStatus()
{
    // CUDA GPU 상태 확인
    #ifdef CUDA_ENABLED
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        MGlobal::displayWarning("No CUDA-capable GPU found");
        return false;
    }
    
    // GPU 메모리 상태 확인
    size_t freeMemory, totalMemory;
    error = cudaMemGetInfo(&freeMemory, &totalMemory);
    if (error == cudaSuccess) {
        double freeMemoryGB = (double)freeMemory / (1024.0 * 1024.0 * 1024.0);
        if (freeMemoryGB < 0.5) { // 500MB 미만이면 경고
            MGlobal::displayWarning("Low GPU memory warning: Available GPU memory is less than 500MB");
            return false;
        }
    }
    #endif
    
    return true;
}

// performDeformation 함수의 나머지 부분 제거됨

bool offsetCurveDeformerNode::validateOutputData(MItGeometry& iter)
{
    MStatus status;
    MPointArray points;
    
    // 출력 포인트 가져오기
    status = iter.allPositions(points);
    if (!status || points.length() == 0) {
        MGlobal::displayError("Failed to get output points from Offset Curve Deformer");
        return false;
    }
    
    // 기본적인 포인트 유효성 검증 (간단한 버전)
    for (unsigned int i = 0; i < points.length(); i++) {
        // 극단적인 값 확인 (예: 10000 단위 이상)
        double x = points[i].x;
        double y = points[i].y;
        double z = points[i].z;
        
        if (x > 10000.0 || x < -10000.0 ||
            y > 10000.0 || y < -10000.0 ||
            z > 10000.0 || z < -10000.0) {
            MGlobal::displayWarning("Extreme output point detected in Offset Curve Deformer");
            return false;
        }
    }
    
    return true;
}

void offsetCurveDeformerNode::cleanupResources()
{
    // 메모리 정리
    if (mAlgorithm) {
        mAlgorithm.reset();
    }
    
    // 포인트 배열 정리
    mPoseTargetPoints.clear();
    
    // 리바인드 플래그 재설정
    mNeedsRebind = true;
}

bool offsetCurveDeformerNode::initializeResources()
{
    try {
        // 알고리즘 초기화
        if (!mAlgorithm) {
            mAlgorithm = std::make_unique<offsetCurveAlgorithm>();
        }
        
        // 포인트 배열 초기화
        mPoseTargetPoints.clear();
        
        // 리바인드 플래그 설정
        mNeedsRebind = true;
        
        return true;
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Failed to initialize resources: ") + e.what());
        return false;
    }
}

    // 추가: influenceCurve에서 데이터 가져오기 (Maya 표준 input과 동일한 구조)
MStatus offsetCurveDeformerNode::getInfluenceCurve(MDataBlock& dataBlock, MDagPath& influenceCurve)
{
    MStatus status;
    
    // 1. influenceCurve 배열 속성에서 첫 번째 요소 가져오기 (logicalIndex 0)
    MGlobal::displayInfo("=== getInfluenceCurve() 시작 ===");
    MGlobal::displayInfo("1단계: influenceCurve 배열 속성 가져오기");
    
    // 수정: inputArrayValue 대신 outputArrayValue 사용 (cached 값에 직접 접근)
    MGlobal::displayInfo("outputArrayValue 사용하여 cached 값에 직접 접근");
    MArrayDataHandle hInfluenceCurveArray = dataBlock.outputArrayValue(aInfluenceCurve, &status);
    if (status != MS::kSuccess) {
        MGlobal::displayError("Failed to get influenceCurve array");
        return status;
    }
    MGlobal::displayInfo("influenceCurve 배열 속성 가져오기 성공");
    
    // 2. 배열에 요소가 있는지 확인
    MGlobal::displayInfo("2단계: 배열 요소 개수 확인");
    unsigned int elementCount = hInfluenceCurveArray.elementCount();
    MGlobal::displayInfo(MString("배열 요소 개수: ") + elementCount);
    
    if (elementCount == 0) {
        MGlobal::displayError("No influence curves connected - 배열이 비어있음");
        return MS::kFailure;
    }
    
    // 추가: 배열의 logical indices 확인
    MGlobal::displayInfo("3단계: 배열 logical indices 확인");
    // Maya 2020에서는 getLogicalIndices를 지원하지 않으므로 제거
    MGlobal::displayInfo("Maya 2020에서는 logical indices를 직접 가져올 수 없음");
    
    // 추가: outputArrayValue 사용 시 주의사항
    MGlobal::displayInfo("outputArrayValue 사용 시: cached 값에 직접 접근, evaluation 오버헤드 없음");
    MGlobal::displayInfo("outputArrayValue 사용 시: 데이터가 변경되지 않았으면 이전 값이 유지됨");
    
    // 3. 첫 번째 요소로 이동 (logicalIndex 0)
    MGlobal::displayInfo("4단계: 첫 번째 요소로 이동");
    status = hInfluenceCurveArray.jumpToElement(0);
    if (status != MS::kSuccess) {
        MGlobal::displayError("Failed to jump to first element");
        return status;
    }
    MGlobal::displayInfo("첫 번째 요소로 이동 성공");
    
    // 추가: 현재 요소의 logical index 확인
    int currentLogicalIndex = hInfluenceCurveArray.elementIndex();
    MGlobal::displayInfo(MString("현재 요소의 logical index: ") + currentLogicalIndex);

    // 4. 복합 속성의 influenceCurveData에서 nurbsCurve 가져오기 (Maya API 표준 방식)
    MGlobal::displayInfo("5단계: 복합 속성 값 가져오기");
    // 수정: inputValue 대신 outputValue 사용 (cached 값에 직접 접근)
    MGlobal::displayInfo("outputValue 사용하여 cached 값에 직접 접근");
    MDataHandle hInfluenceCurveCompound = hInfluenceCurveArray.outputValue(&status);
    if (status != MS::kSuccess) {
        MGlobal::displayError("Failed to get compound attribute value");
        return status;
    }
    MGlobal::displayInfo("복합 속성 값 가져오기 성공");
    
    // 추가: 복합 속성의 타입 확인
    MGlobal::displayInfo(MString("복합 속성 데이터 타입: ") + hInfluenceCurveCompound.type());
    
    // 5. influenceCurveData 하위 속성에서 nurbsCurve 데이터 가져오기
    MGlobal::displayInfo("6단계: influenceCurveData 하위 속성 가져오기");
    MDataHandle hInfluenceCurveData = hInfluenceCurveCompound.child(aInfluenceCurveData);
    
    // 수정: Maya 2020에서는 isNull()을 지원하지 않으므로 다른 방법으로 검증
    // 하위 속성이 제대로 가져와졌는지 확인
    MGlobal::displayInfo("influenceCurveData 하위 속성 가져오기 성공");
    MGlobal::displayInfo(MString("하위 속성 데이터 타입: ") + hInfluenceCurveData.type());
    
    // Maya 2020에서는 isConnected를 직접 확인할 수 없으므로 제거
    MGlobal::displayInfo("Maya 2020에서는 isConnected를 직접 확인할 수 없음");
    
    // 6. nurbsCurve 데이터에서 MObject 가져오기
    MGlobal::displayInfo("7단계: nurbsCurve 데이터에서 MObject 가져오기");
    MObject influenceObj = hInfluenceCurveData.data();
    
    // 추가: MObject 상세 정보 출력
    if (influenceObj.isNull()) {
        MGlobal::displayError("Influence curve data is null");
        return MS::kFailure;
    }
    
    // 추가: MObject 타입 정보 출력
    MFnDependencyNode depNode(influenceObj);
    MGlobal::displayInfo(MString("MObject 노드 타입: ") + depNode.typeName());
    MGlobal::displayInfo(MString("MObject 노드 이름: ") + depNode.name());
    
    // 추가: MObject의 함수 세트 확인
    if (influenceObj.hasFn(MFn::kNurbsCurve)) {
        MGlobal::displayInfo("MObject가 NURBS 곡선 함수 세트를 가짐");
    } else if (influenceObj.hasFn(MFn::kTransform)) {
        MGlobal::displayInfo("MObject가 Transform 함수 세트를 가짐");
    } else if (influenceObj.hasFn(MFn::kDagNode)) {
        MGlobal::displayInfo("MObject가 DAG 노드 함수 세트를 가짐");
    } else {
        MGlobal::displayInfo("MObject의 함수 세트를 확인할 수 없음");
    }
    
    MGlobal::displayInfo("nurbsCurve 데이터에서 MObject 가져오기 성공");
    
    // 7. MDagPath로 변환
    MGlobal::displayInfo("8단계: MDagPath로 변환");
    status = MDagPath::getAPathTo(influenceObj, influenceCurve);
    if (status != MS::kSuccess) {
        MGlobal::displayError("Failed to get DAG path to influence curve");
        return status;
    }
    MGlobal::displayInfo("MDagPath로 변환 성공");
    
    // 추가: MDagPath 상세 정보 출력
    MGlobal::displayInfo(MString("MDagPath 노드 이름: ") + influenceCurve.fullPathName());
    MGlobal::displayInfo(MString("MDagPath 노드 타입: ") + influenceCurve.node().apiTypeStr());
    
    // 8. 디버그 정보 출력
    MGlobal::displayInfo("9단계: 최종 검증");
    MGlobal::displayInfo("Successfully found influence curve");
    
    // 9. NURBS 곡선인지 확인
    MGlobal::displayInfo("10단계: NURBS 곡선 타입 최종 확인");
    if (influenceCurve.hasFn(MFn::kNurbsCurve)) {
        MGlobal::displayInfo("Influence curve is a NURBS curve");
        MGlobal::displayInfo("=== getInfluenceCurve() 성공 완료 ===");
        return MS::kSuccess;
    } else {
        MGlobal::displayError("Influence curve is not a NURBS curve");
        MGlobal::displayInfo(MString("실제 노드 타입: ") + influenceCurve.node().apiTypeStr());
        return MS::kFailure;
    }
}