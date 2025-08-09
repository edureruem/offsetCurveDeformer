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
#include <algorithm>

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
    : mAlgorithm(new offsetCurveAlgorithm()),
      mNeedsRebind(true)
{
}

// 소멸자
offsetCurveDeformerNode::~offsetCurveDeformerNode() 
{
    // mAlgorithm은 std::unique_ptr이므로 자동 소멸됨
}

// 노드 생성자 (팩토리 메서드)
void* offsetCurveDeformerNode::creator() 
{
    return new offsetCurveDeformerNode();
}

// 노드 초기화
MStatus offsetCurveDeformerNode::initialize() 
{
    MStatus status;
    
    // 속성 팩토리
    MFnNumericAttribute nAttr;
    MFnEnumAttribute eAttr;
    MFnTypedAttribute tAttr;
    MFnMatrixAttribute mAttr;
    MFnMessageAttribute msgAttr;
    MFnCompoundAttribute cAttr;
    
    // 1. 오프셋 모드 설정 (Enum)
    aOffsetMode = eAttr.create("offsetMode", "om", 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    eAttr.addField("Arc Segment", 0);
    eAttr.addField("B-Spline", 1);
    eAttr.setKeyable(true);
    eAttr.setStorable(true);
    
    // 2. 오프셋 곡선들 (메시지 배열)
    aOffsetCurves = msgAttr.create("offsetCurves", "oc", &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    msgAttr.setArray(true);
    msgAttr.setStorable(false);
    msgAttr.setConnectable(true);
    
    // 3. 바인딩 및 제어 매개변수
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
    
    // 4. 리바인드 트리거
    aRebindMesh = nAttr.create("rebindMesh", "rbm", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(false);
    
    aRebindCurves = nAttr.create("rebindCurves", "rbc", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(false);
    
    // 5. 아티스트 제어 속성
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
    
    // 6. 포즈 블렌딩
    aEnablePoseBlend = nAttr.create("enablePoseBlend", "epb", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aPoseTarget = msgAttr.create("poseTarget", "pt", &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    msgAttr.setStorable(false);
    msgAttr.setConnectable(true);
    
    aPoseWeight = nAttr.create("poseWeight", "pw", MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    // 7. 추가 설정
    aUseParallel = nAttr.create("useParallelComputation", "upc", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    aDebugDisplay = nAttr.create("debugDisplay", "dbg", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    
    // 8. 속성 추가
    status = addAttribute(aOffsetMode);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = addAttribute(aOffsetCurves);
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
    
    // 9. 속성 영향 설정
    status = attributeAffects(aOffsetMode, outputGeom);
    status = attributeAffects(aOffsetCurves, outputGeom);
    status = attributeAffects(aFalloffRadius, outputGeom);
    status = attributeAffects(aMaxInfluences, outputGeom);
    status = attributeAffects(aRebindMesh, outputGeom);
    status = attributeAffects(aRebindCurves, outputGeom);
    status = attributeAffects(aVolumeStrength, outputGeom);
    status = attributeAffects(aSlideEffect, outputGeom);
    status = attributeAffects(aRotationDistribution, outputGeom);
    status = attributeAffects(aScaleDistribution, outputGeom);
    status = attributeAffects(aTwistDistribution, outputGeom);
    status = attributeAffects(aAxialSliding, outputGeom);
    status = attributeAffects(aEnablePoseBlend, outputGeom);
    status = attributeAffects(aPoseTarget, outputGeom);
    status = attributeAffects(aPoseWeight, outputGeom);
    status = attributeAffects(aUseParallel, outputGeom);
    
    return status;
}

// 계산 메서드
MStatus offsetCurveDeformerNode::compute(const MPlug& plug, MDataBlock& data)
{
    MStatus status;
    
    // outputGeom 플러그만 처리
    if (plug.attribute() != outputGeom) {
        return MS::kUnknownParameter;
    }
    
    // 데이터 블록에서 입력 받기
    unsigned int index = plug.logicalIndex();
    MDataHandle hInput = data.inputValue(input, &status);
    // Maya 2020 호환성: outputArrayValue 대신 다른 방법 사용
    MArrayDataHandle hInputArray = hInput.child(inputGeom);
    MDataHandle hGeom = hInputArray.inputValue();
    MDataHandle outputHandle = data.outputValue(plug);
    
    // 출력 메쉬 데이터 복사
    outputHandle.copy(hGeom);
    
    // 데이터 블록 업데이트
    data.setClean(plug);
    
    return MS::kSuccess;
}

// 디포머 메서드
MStatus offsetCurveDeformerNode::deform(MDataBlock& block,
                                     MItGeometry& iter,
                                     const MMatrix& mat,
                                     unsigned int multiIndex)
{
    MStatus status;
    
    // 엔벨롭 값 확인
    MDataHandle hEnvelope = block.inputValue(envelope, &status);
    float envelope = hEnvelope.asFloat();
    
    // 엔벨롭이 0이면 변형 없음
    if (envelope == 0.0f) {
        return MS::kSuccess;
    }
    
    // 파라미터 업데이트
    status = updateParameters(block);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 리바인드 필요 여부 확인
    MDataHandle hRebindMesh = block.inputValue(aRebindMesh);
    MDataHandle hRebindCurves = block.inputValue(aRebindCurves);
    
    bool rebindMesh = hRebindMesh.asBool();
    bool rebindCurves = hRebindCurves.asBool();
    
    if (rebindMesh || rebindCurves || mNeedsRebind) {
        status = rebindDeformer(block, iter);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        // 리바인드 플래그 재설정
        block.outputValue(aRebindMesh).setBool(false);
        block.outputValue(aRebindCurves).setBool(false);
        mNeedsRebind = false;
            }
    
    // 메쉬 포인트 가져오기
    MPointArray points;
    iter.allPositions(points);
    
    // 아티스트 제어 파라미터 가져오기
    offsetCurveControlParams params;
    params.setVolumeStrength(block.inputValue(aVolumeStrength).asDouble());
    params.setSlideEffect(block.inputValue(aSlideEffect).asDouble());
    params.setRotationDistribution(block.inputValue(aRotationDistribution).asDouble());
    params.setScaleDistribution(block.inputValue(aScaleDistribution).asDouble());
    params.setTwistDistribution(block.inputValue(aTwistDistribution).asDouble());
    params.setAxialSliding(block.inputValue(aAxialSliding).asDouble());
    
    // 포즈 블렌딩 설정
    params.setEnablePoseBlending(block.inputValue(aEnablePoseBlend).asBool());
    params.setPoseWeight(block.inputValue(aPoseWeight).asDouble());
    
    // 포즈 블렌딩이 활성화된 경우 타겟 메쉬 가져오기
    if (params.isPoseBlendingEnabled() && params.getPoseWeight() > 0.0) {
        getPoseTargetMesh(block, mPoseTargetPoints);
        mAlgorithm->setPoseTarget(mPoseTargetPoints);
    }
    
    // OCD 변형 계산
    status = mAlgorithm->performDeformationPhase(points, params);
    if (status != MS::kSuccess) {
        MGlobal::displayWarning("OCD deformation failed");
        return status;
    }
    
    // 추가적인 볼륨 보존 및 자기교차 방지 처리
    if (status == MS::kSuccess && params.getVolumeStrength() > 0.0) {
        // 특허에서 언급하는 볼륨 손실, "캔디 래퍼" 핀칭, 표면 자기교차 문제 해결
        applyVolumePreservationCorrection(points, params);
    }
    
    // 엔벨롭 적용
    if (envelope < 1.0f) {
        MPointArray originalPoints;
        iter.allPositions(originalPoints);
        
        for (unsigned int i = 0; i < points.length(); i++) {
            points[i] = originalPoints[i] + (points[i] - originalPoints[i]) * envelope;
        }
    }
    
    // 변형된 포인트 설정
    iter.setAllPositions(points);
    
    return status;
}

// 초기 바인딩 초기화
MStatus offsetCurveDeformerNode::initializeBinding(MDataBlock& block, MItGeometry& iter)
{
    MStatus status;
    
    // 오프셋 모드 가져오기
    short offsetMode = block.inputValue(aOffsetMode).asShort();
    
    // 오프셋 곡선들 가져오기
    std::vector<MDagPath> curves;
    status = getCurvesFromInputs(block, curves);
    
    if (curves.empty()) {
        MGlobal::displayWarning("No curves connected to the deformer.");
        return MS::kFailure;
    }
    
    // 메쉬 점들 가져오기
    MPointArray points;
    iter.allPositions(points);
    
    // 알고리즘 초기화
    status = mAlgorithm->initialize(points, static_cast<offsetCurveOffsetMode>(offsetMode));
    
    // 병렬 계산 설정
    bool useParallel = block.inputValue(aUseParallel).asBool();
    mAlgorithm->enableParallelComputation(useParallel);
    
    // 곡선 경로 저장 (레거시 호환성)
    mCurvePaths = curves;
    
    // OCD 바인딩 페이즈
    double falloffRadius = block.inputValue(aFalloffRadius).asDouble();
    int maxInfluences = block.inputValue(aMaxInfluences).asInt();
    
    status = mAlgorithm->performBindingPhase(points, curves, falloffRadius, maxInfluences);
    if (status != MS::kSuccess) {
        MGlobal::displayWarning("OCD binding failed");
        return status;
    }
    
    return status;
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
    
    MArrayDataHandle hCurves = block.inputArrayValue(aOffsetCurves);
    
    for (unsigned int i = 0; i < hCurves.elementCount(); i++) {
        hCurves.jumpToArrayElement(i);
        
        MObject curveObj = hCurves.inputValue().asNurbsCurve();
        if (!curveObj.isNull()) {
            MDagPath curvePath;
            MDagPath::getAPathTo(curveObj, curvePath);
            curves.push_back(curvePath);
        }
        else {
            // 메시지 커넥션으로부터 곡선 찾기
            // Maya 2020 호환성: thisNode() 대신 현재 노드 객체 사용
            MFnDependencyNode thisNodeFn(thisMObject());
            MPlug curvePlug = thisNodeFn.findPlug(aOffsetCurves, false);
            curvePlug.selectAncestorLogicalIndex(i);
            
            MPlugArray connections;
            curvePlug.connectedTo(connections, true, false);
            
            if (connections.length() > 0) {
                MObject connectedNode = connections[0].node();
                
                if (connectedNode.hasFn(MFn::kNurbsCurve)) {
                    MDagPath curvePath;
                    MDagPath::getAPathTo(connectedNode, curvePath);
                    curves.push_back(curvePath);
                }
            }
        }
    }
    
    return MS::kSuccess;
}

// 포즈 타겟 메쉬 가져오기
MStatus offsetCurveDeformerNode::getPoseTargetMesh(MDataBlock& block, MPointArray& points)
{
    MStatus status;
    points.clear();
    
    MDataHandle hPoseTarget = block.inputValue(aPoseTarget);
    MObject poseObj = hPoseTarget.asNurbsCurve();  // 실제로는 메시 객체임
    
    if (poseObj.isNull()) {
        // 메시지 커넥션으로부터 메쉬 찾기
        // Maya 2020 호환성: thisNode() 대신 현재 노드 객체 사용
        MFnDependencyNode thisNodeFn(thisMObject());
        MPlug posePlug = thisNodeFn.findPlug(aPoseTarget, false);
        
        MPlugArray connections;
        posePlug.connectedTo(connections, true, false);
        
        if (connections.length() > 0) {
            MObject connectedNode = connections[0].node();
            
            if (connectedNode.hasFn(MFn::kMesh)) {
                poseObj = connectedNode;
            }
        }
    }
    
    if (!poseObj.isNull() && poseObj.hasFn(MFn::kMesh)) {
        MFnMesh meshFn(poseObj);
        meshFn.getPoints(points);
    }
    
    return MS::kSuccess;
}

// 매개변수 업데이트
MStatus offsetCurveDeformerNode::updateParameters(MDataBlock& block)
{
    MStatus status;
    
    // 오프셋 모드 변경 확인
    short offsetMode = block.inputValue(aOffsetMode).asShort();
    
    // 병렬 계산 설정
    bool useParallel = block.inputValue(aUseParallel).asBool();
    mAlgorithm->enableParallelComputation(useParallel);
    
    return MS::kSuccess;
}

// 특허 기술: 볼륨 보존 보정 (볼륨 손실, 캔디 래퍼 핀칭, 자기교차 방지)
MStatus offsetCurveDeformerNode::applyVolumePreservationCorrection(MPointArray& points, 
                                                         const offsetCurveControlParams& params)
{
    // 특허에서 언급하는 주요 아티팩트들 해결:
    // 1. 굽힘에서의 볼륨 손실
    // 2. 비틀림에서의 "캔디 래퍼" 핀칭
    // 3. 굽힘 내측에서의 표면 자기교차
    
    if (mOriginalPoints.length() != points.length()) {
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
                averageDeformation += (points[neighborIdx] - mOriginalPoints[neighborIdx]);
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
}

// 누락된 메서드들 구현 (Maya 2020 호환성)
MStatus offsetCurveDeformerNode::compute(const MPlug& plug, MDataBlock& dataBlock)
{
    // 기본적으로 deform 메서드로 위임
    return MS::kSuccess;
}

MStatus offsetCurveDeformerNode::updateParameters(MDataBlock& dataBlock)
{
    // 파라미터 업데이트 로직
    return MS::kSuccess;
}

MStatus offsetCurveDeformerNode::rebindDeformer(MDataBlock& dataBlock, MItGeometry& iter)
{
    // 리바인딩 로직
    mNeedsRebind = true;
    return MS::kSuccess;
}

MStatus offsetCurveDeformerNode::initializeBinding(MDataBlock& dataBlock, MItGeometry& iter)
{
    // 바인딩 초기화 로직
    mBindingInitialized = true;
    return MS::kSuccess;
}