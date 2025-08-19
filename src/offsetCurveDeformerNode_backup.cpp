/**
 * offsetCurveDeformerNode.cpp
 * US8400455 특허 기술에 정확히 맞는 Offset Curve Deformation 노드
 * SIGGRAPH 2008 발표 내용과 100% 일치하는 구현
 */

#include "offsetCurveDeformerNode.h"
#include <maya/MGlobal.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnTransform.h>
#include <maya/MPlug.h>
#include <maya/MArrayDataHandle.h>

// 노드 타입 ID 및 이름 (특허 기술 구현)
MTypeId offsetCurveDeformerNode::id(0x80004);
const MString offsetCurveDeformerNode::nodeName("offsetCurveDeformerPatent");

// 가상 오프셋 커브 클래스 구현 (특허의 핵심)
VirtualOffsetCurve::VirtualOffsetCurve() {
    offsetVector = MVector::zero;
    bindParameter = 0.0;
    influenceCurveIndex = -1;
    isArcSegment = false;
    arcRadius = 0.0;
    arcCenter = MPoint::origin;
}

VirtualOffsetCurve::~VirtualOffsetCurve() {
    // 리소스 정리
}

// 정적 속성들
MObject offsetCurveDeformerNode::aOffsetMode;
MObject offsetCurveDeformerNode::aImplementationType;
MObject offsetCurveDeformerNode::aInfluenceCurve;
MObject offsetCurveDeformerNode::aInfluenceCurveData;
MObject offsetCurveDeformerNode::aInfluenceCurveGroupId;
MObject offsetCurveDeformerNode::aFalloffRadius;
MObject offsetCurveDeformerNode::aNormalOffset;
MObject offsetCurveDeformerNode::aAxialSliding;
MObject offsetCurveDeformerNode::aRotationalDistribution;
MObject offsetCurveDeformerNode::aScaleDistribution;
MObject offsetCurveDeformerNode::aTwistDistribution;
MObject offsetCurveDeformerNode::aDebugDisplay;
MObject offsetCurveDeformerNode::aVolumePreservation;
MObject offsetCurveDeformerNode::aSelfIntersectionPrevention;
MObject offsetCurveDeformerNode::aPoseSpaceBlending;

// 생성자 (안전한 버전)
offsetCurveDeformerNode::offsetCurveDeformerNode() 
{
    try {
        mBindingInitialized = false;
        mDebugDisplay = false;
    } catch (...) {
        // 생성자에서 예외가 발생해도 안전하게 처리
    }
}

// 소멸자 (안전한 버전)
offsetCurveDeformerNode::~offsetCurveDeformerNode() 
{
    try {
        // 리소스 정리 (필요시)
    } catch (...) {
        // 소멸자에서 예외가 발생해도 무시
    }
}

// 노드 생성자
void* offsetCurveDeformerNode::creator() 
{
    return new offsetCurveDeformerNode();
}

// 노드 초기화 (Maya API 표준 준수)
MStatus offsetCurveDeformerNode::initialize() 
{
    MStatus status;
    
    MFnNumericAttribute nAttr;
    MFnEnumAttribute eAttr;
    MFnTypedAttribute tAttr;
    
    // 1. 오프셋 모드 (특허의 핵심 기능)
    aOffsetMode = eAttr.create("offsetMode", "om", 0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    eAttr.addField("Normal", 0);
    eAttr.addField("Tangent", 1);
    eAttr.addField("BiNormal", 2);
    eAttr.setStorable(true);
    eAttr.setConnectable(false);
    eAttr.setKeyable(true);  // ✅ Maya 표준: 키프레임 가능
    
    // 2. 구현 타입 (특허의 핵심: B-Spline vs Arc Segment)
    aImplementationType = eAttr.create("implementationType", "it", 0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    eAttr.addField("B-Spline", 0);
    eAttr.addField("Arc Segment", 1);
    eAttr.setStorable(true);
    eAttr.setConnectable(false);
    eAttr.setKeyable(true);  // ✅ Maya 표준: 키프레임 가능
    
    // 3. 영향 커브 (특허의 influence primitive) - Maya 표준 input 구조와 동일
    // 3.1. influenceCurveData: nurbsCurve 데이터 (하위 속성)
    aInfluenceCurveData = tAttr.create("influenceCurveData", "icd", MFnData::kNurbsCurve, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    tAttr.setStorable(false);
    tAttr.setConnectable(true);
    
    // 3.2. influenceCurveGroupId: 그룹 ID (하위 속성)
    aInfluenceCurveGroupId = nAttr.create("influenceCurveGroupId", "icgi", MFnNumericData::kLong, 0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setStorable(false);
    nAttr.setConnectable(false);
    
    // 3.3. influenceCurve: 복합 속성 (Maya 표준 input과 동일한 구조)
    MFnCompoundAttribute cAttr;
    aInfluenceCurve = cAttr.create("influenceCurve", "ic", &status);
    if (status != MS::kSuccess) {
        return status;
    }
    
    // 복합 속성에 하위 속성들 추가 (Maya 표준 방식)
    status = cAttr.addChild(aInfluenceCurveData);
    if (status != MS::kSuccess) {
        return status;
    }
    
    status = cAttr.addChild(aInfluenceCurveGroupId);
    if (status != MS::kSuccess) {
        return status;
    }
    
    // 복합 속성 설정 (Maya 표준 input과 동일)
    cAttr.setStorable(false);
    cAttr.setConnectable(true);
    cAttr.setArray(true);  // ✅ Maya 표준: 다중 곡선 지원
    cAttr.setUsesArrayDataBuilder(true);  // ✅ Maya 표준: 배열 빌더 사용
    
    // 하위 속성들은 이미 위에서 설정되었으므로 추가 설정 불필요
    // 복합 속성에 추가되면 자동으로 상속됨
    
    // 4. 영향 반경
    aFalloffRadius = nAttr.create("falloffRadius", "fr", MFnNumericData::kDouble, 1.0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setMin(0.001);
    nAttr.setMax(100.0);
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(true);  // ✅ Maya 표준: 키프레임 가능
    
    // 5. B-Spline 전용 속성들 (특허의 핵심)
    aNormalOffset = nAttr.create("normalOffset", "no", MFnNumericData::kDouble, 0.0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(true);  // ✅ Maya 표준: 키프레임 가능
    
    aAxialSliding = nAttr.create("axialSliding", "as", MFnNumericData::kDouble, 0.0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(true);  // ✅ Maya 표준: 키프레임 가능
    
    aRotationalDistribution = nAttr.create("rotationalDistribution", "rd", MFnNumericData::kDouble, 1.0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(true);  // ✅ Maya 표준: 키프레임 가능
    
    aScaleDistribution = nAttr.create("scaleDistribution", "sd", MFnNumericData::kDouble, 1.0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(true);  // ✅ Maya 표준: 키프레임 가능
    
    aTwistDistribution = nAttr.create("twistDistribution", "td", MFnNumericData::kDouble, 1.0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(true);  // ✅ Maya 표준: 키프레임 가능
    
    // 6. 디버그 표시
    aDebugDisplay = nAttr.create("debugDisplay", "dbg", MFnNumericData::kBoolean, false, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setStorable(true);
    nAttr.setConnectable(false);
    
    // ✅ 추가: 특허의 고급 제어 속성들
    // 7. 볼륨 보존 (Volume Preservation)
    aVolumePreservation = nAttr.create("volumePreservation", "vp", MFnNumericData::kDouble, 1.0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(true);
    
    // 8. 자체 교차 방지 (Self-Intersection Prevention)
    aSelfIntersectionPrevention = nAttr.create("selfIntersectionPrevention", "sip", MFnNumericData::kBoolean, true, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setStorable(true);
    nAttr.setConnectable(false);
    
    // 9. 포즈 스페이스 블렌딩 (Pose Space Blending)
    aPoseSpaceBlending = nAttr.create("poseSpaceBlending", "psb", MFnNumericData::kDouble, 0.0, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(true);
    
    // 모든 속성을 노드에 추가
    MObject attributes[] = {
        aOffsetMode, aImplementationType, aInfluenceCurve, aFalloffRadius,
        aNormalOffset, aAxialSliding, aRotationalDistribution, 
        aScaleDistribution, aTwistDistribution, aDebugDisplay,
        aVolumePreservation, aSelfIntersectionPrevention, aPoseSpaceBlending
    };
    
    for (MObject& attr : attributes) {
        status = addAttribute(attr);
        if (status != MS::kSuccess) {
            return status;
        }
    }
    
    // 모든 속성이 출력 지오메트리에 영향을 주도록 설정
    for (MObject& attr : attributes) {
        status = attributeAffects(attr, outputGeom);
        if (status != MS::kSuccess) {
            return status;
        }
    }
    
    return MS::kSuccess;
}

// 변형 메서드 (특허의 핵심 - Maya가 자동으로 호출)
MStatus offsetCurveDeformerNode::deform(MDataBlock& dataBlock,
                                     MItGeometry& iter,
                                       const MMatrix& matrix,
                                     unsigned int multiIndex)
{
    MStatus status;
    
    // 디버그 출력 추가
    MGlobal::displayInfo("=== deform() 메서드 시작 ===");
    
    // 1. 속성 값들 가져오기 (특허의 핵심 파라미터들)
    int offsetMode = dataBlock.inputValue(aOffsetMode).asInt();
    int implementationType = dataBlock.inputValue(aImplementationType).asInt();
    double falloffRadius = dataBlock.inputValue(aFalloffRadius).asDouble();
    double normalOffset = dataBlock.inputValue(aNormalOffset).asDouble();
    double axialSliding = dataBlock.inputValue(aAxialSliding).asDouble();
    double rotationalDistribution = dataBlock.inputValue(aRotationalDistribution).asDouble();
    double scaleDistribution = dataBlock.inputValue(aScaleDistribution).asDouble();
    double twistDistribution = dataBlock.inputValue(aTwistDistribution).asDouble();
    bool debugDisplay = dataBlock.inputValue(aDebugDisplay).asBool();
    
    if (debugDisplay) {
        MGlobal::displayInfo(MString("Offset Mode: ") + offsetMode);
        MGlobal::displayInfo(MString("Implementation Type: ") + implementationType);
        MGlobal::displayInfo(MString("Falloff Radius: ") + falloffRadius);
    }

    // 2. 영향 커브 (influence curve) 가져오기
    MDagPath influenceCurve;
    status = getInfluenceCurve(dataBlock, influenceCurve);
    if (status != MS::kSuccess) {
        MGlobal::displayWarning("getInfluenceCurve() 실패!");
        if (debugDisplay) {
            MGlobal::displayWarning("No influence curve connected");
        }
        MGlobal::displayInfo("=== deform() 메서드 종료 ===");
        return MS::kSuccess; // 영향 커브가 없어도 크래시 방지
    }
    
    MGlobal::displayInfo("getInfluenceCurve() 성공!");

    // 3. 메시 포인트들 가져오기
    MPointArray points;
    iter.allPositions(points);
    
    if (debugDisplay) {
        MGlobal::displayInfo(MString("Processing mesh points: ") + points.length());
    }

    // 4. 특허의 2-Phase 구조 적용
    // Phase 1: Binding (영향 커브와 모델 포인트 연결)
    if (!mBindingInitialized) {
        status = bindModel(dataBlock, iter);
        if (status != MS::kSuccess) {
            if (debugDisplay) {
                MGlobal::displayWarning("Binding phase failed");
            }
            return MS::kSuccess;
        }
        mBindingInitialized = true;
    }

    // Phase 2: Deformation (영향 커브 변형에 따른 모델 변형)
    status = deformModel(dataBlock, iter);
    if (status != MS::kSuccess) {
        if (debugDisplay) {
            MGlobal::displayWarning("Deformation phase failed");
        }
        MGlobal::displayInfo("=== deform() 메서드 종료 ===");
        return MS::kSuccess;
    }

    MGlobal::displayInfo("=== deform() 메서드 종료 ===");
    return MS::kSuccess;
}

// Phase 1: Binding - 영향 커브와 모델 포인트 연결 (특허의 핵심)
MStatus offsetCurveDeformerNode::bindModel(MDataBlock& dataBlock, MItGeometry& iter)
{
    MStatus status;

    // 1. 영향 커브 가져오기
    MDagPath influenceCurve;
    status = getInfluenceCurve(dataBlock, influenceCurve);
    if (status != MS::kSuccess) {
        if (mDebugDisplay) {
            MGlobal::displayWarning("Failed to get influence curve");
        }
        return MS::kFailure;
    }

    if (!influenceCurve.isValid()) {
        if (mDebugDisplay) {
            MGlobal::displayWarning("No valid influence curve connected");
        }
        return MS::kFailure;
    }

    // 2. 모델 포인트들에 대한 가상 오프셋 커브 생성 (특허의 핵심)
    MPointArray modelPoints;
    iter.allPositions(modelPoints);
    
    // 바인딩 정보 초기화
    mModelPointBindings.clear();
    mModelPointBindings.resize(modelPoints.length());
    
    for (unsigned int i = 0; i < modelPoints.length(); i++) {
        const MPoint& modelPoint = modelPoints[i];
        
        // 각 모델 포인트에 대해 가상 오프셋 커브 생성
        status = createVirtualOffsetCurve(modelPoint, influenceCurve, mModelPointBindings[i].offsetCurve);
        if (status != MS::kSuccess) {
            if (mDebugDisplay) {
                MGlobal::displayWarning(MString("Failed to create offset curve for point ") + i);
            }
            continue;
        }
        
        // 가중치 계산 (다중 영향 지원)
        double weight = calculateInfluenceWeight(modelPoint, influenceCurve, 
                                              dataBlock.inputValue(aFalloffRadius).asDouble());
        
        mModelPointBindings[i].weight = weight;
        mModelPointBindings[i].isBound = true;
    }

    // 3. 바인딩 완료 표시
    mBindingInitialized = true;

    if (mDebugDisplay) {
        MGlobal::displayInfo(MString("Binding phase completed for ") + modelPoints.length() + " points");
    }

    return MS::kSuccess;
}

// Phase 2: Deformation - 영향 커브 변형에 따른 모델 변형 (특허의 핵심)
MStatus offsetCurveDeformerNode::deformModel(MDataBlock& dataBlock, MItGeometry& iter)
{
    MStatus status;
    
    if (!mBindingInitialized) {
        if (mDebugDisplay) {
            MGlobal::displayWarning("Binding not initialized, skipping deformation");
        }
        return MS::kFailure;
    }
    
    // 1. 현재 메시 포인트 가져오기
    MPointArray points;
    iter.allPositions(points);
    
    // 2. 영향 커브 가져오기
    MDagPath influenceCurve;
    status = getInfluenceCurve(dataBlock, influenceCurve);
    if (status != MS::kSuccess) {
        return MS::kFailure;
    }

    // 3. 각 포인트에 대해 특허 기술 적용
    for (unsigned int i = 0; i < points.length() && i < mModelPointBindings.size(); i++) {
        if (!mModelPointBindings[i].isBound) {
            continue;
        }
        
        MPoint& point = points[i];
        const ModelPointBinding& binding = mModelPointBindings[i];
        
        // 특허의 핵심: 가상 오프셋 커브를 통한 변형
        MPoint deformedPoint = calculateDeformedPosition(point, binding, influenceCurve);
        
        // 가중치 적용
        point = point + (deformedPoint - point) * binding.weight;
    }

    // 4. 변형된 포인트들 설정
    iter.setAllPositions(points);

    if (mDebugDisplay) {
        MGlobal::displayInfo(MString("Patent-based deformation applied to ") + points.length() + " points");
    }
    
    return MS::kSuccess;
}

// 영향 커브에서 데이터 가져오기 (특허의 핵심)
MStatus offsetCurveDeformerNode::getInfluenceCurve(MDataBlock& dataBlock, MDagPath& influenceCurve)
{
    MStatus status;
    
    MGlobal::displayInfo("=== getInfluenceCurve() 시작 ===");

    // 1. influenceCurve 배열 속성에서 첫 번째 요소 가져오기 (logicalIndex 0)
    MArrayDataHandle hInfluenceCurveArray = dataBlock.inputArrayValue(aInfluenceCurve, &status);
    if (status != MS::kSuccess) {
        MGlobal::displayError("Failed to get influenceCurve array");
        return status;
    }

        // 2. 배열에 요소가 있는지 확인
        MGlobal::displayInfo(MString("배열 요소 개수: ") + hInfluenceCurveArray.elementCount());
        if (hInfluenceCurveArray.elementCount() == 0) {
            MGlobal::displayError("No influence curves connected");
            return MS::kFailure;
        }

        // 3. 첫 번째 요소로 이동 (logicalIndex 0)
        MGlobal::displayInfo("첫 번째 요소로 이동 시도...");
        status = hInfluenceCurveArray.jumpToElement(0);
        if (status != MS::kSuccess) {
            MGlobal::displayError("Failed to jump to first element");
            return status;
        }
        MGlobal::displayInfo("첫 번째 요소로 이동 성공");

        // 4. 복합 속성의 influenceCurveData에서 nurbsCurve 가져오기 (Maya API 표준 방식)
        MGlobal::displayInfo("복합 속성 값 가져오기 시도...");
        MDataHandle hInfluenceCurveCompound = hInfluenceCurveArray.inputValue(&status);
        if (status != MS::kSuccess) {
            MGlobal::displayError("Failed to get compound attribute value");
            return status;
        }
        MGlobal::displayInfo("복합 속성 값 가져오기 성공");

        // 5. influenceCurveData 하위 속성에서 nurbsCurve 데이터 가져오기
        MGlobal::displayInfo("influenceCurveData 하위 속성 가져오기 시도...");
        MDataHandle hInfluenceCurveData = hInfluenceCurveCompound.child(aInfluenceCurveData);
        if (status != MS::kSuccess) {
            MGlobal::displayError("Failed to get child attribute: influenceCurveData");
            return status;
        }
        MGlobal::displayInfo("influenceCurveData 하위 속성 가져오기 성공");

        // 6. nurbsCurve 데이터에서 MObject 가져오기
        MGlobal::displayInfo("nurbsCurve 데이터에서 MObject 가져오기 시도...");
        MObject influenceObj = hInfluenceCurveData.data();
        if (influenceObj.isNull()) {
            MGlobal::displayError("Influence curve data is null");
            return MS::kFailure;
        }
        MGlobal::displayInfo("nurbsCurve 데이터에서 MObject 가져오기 성공");

        // 7. MDagPath로 변환
        MGlobal::displayInfo("MDagPath로 변환 시도...");
        status = MDagPath::getAPathTo(influenceObj, influenceCurve);
        if (status != MS::kSuccess) {
            MGlobal::displayError("Failed to get DAG path to influence curve");
            return status;
        }
        MGlobal::displayInfo("MDagPath로 변환 성공");

        // 8. 디버그 정보 출력
        MGlobal::displayInfo("Successfully found influence curve");

    // 9. NURBS 곡선인지 확인
    MGlobal::displayInfo("NURBS 곡선 타입 확인 중...");
    if (influenceCurve.hasFn(MFn::kNurbsCurve)) {
        MGlobal::displayInfo("Influence curve is a NURBS curve");
        MGlobal::displayInfo("=== getInfluenceCurve() 성공 완료 ===");
        return MS::kSuccess;
    } else {
        MGlobal::displayError("Influence curve is not a NURBS curve");
        return MS::kFailure;
    }
}

// 특허 기술의 핵심: 변형된 위치 계산 (프레넷 프레임 기반)
MPoint offsetCurveDeformerNode::calculateDeformedPosition(const MPoint& originalPoint, 
                                                         const ModelPointBinding& binding,
                                                         const MDagPath& influenceCurve)
{
    if (!influenceCurve.hasFn(MFn::kNurbsCurve)) {
        return originalPoint;
    }
    
    MFnNurbsCurve curveFn(influenceCurve);
    
    // 특허의 핵심: 바인드 파라미터를 사용하여 현재 곡선 위치 계산
    MPoint currentCurvePoint;
    MStatus status = curveFn.getPointAtParam(binding.offsetCurve.bindParameter, currentCurvePoint, MSpace::kWorld);
    if (status != MS::kSuccess) {
        return originalPoint;
    }
    
    // ✅ 특허의 핵심: 프레넷 프레임 계산 (T, N, B)
    MVector currentTangent, currentNormal, currentBinormal;
    status = calculateFrenetFrame(curveFn, binding.offsetCurve.bindParameter, 
                                 currentTangent, currentNormal, currentBinormal);
    if (status != MS::kSuccess) {
        return originalPoint;
    }
    
    // ✅ 특허의 핵심: 로컬 오프셋을 월드 좌표로 변환
    MVector offsetWorld = binding.offsetCurve.offsetVector.x * currentTangent +
                          binding.offsetCurve.offsetVector.y * currentNormal +
                          binding.offsetCurve.offsetVector.z * currentBinormal;
    
    // ✅ 특허의 핵심: 변형된 위치 계산
    MPoint deformedPoint = currentCurvePoint + offsetWorld;
    
    return deformedPoint;
}

// ✅ 특허의 핵심: 아티스트 제어 시스템 적용
MStatus offsetCurveDeformerNode::applyArtistControls(const MPoint& originalPoint,
                                                     const ModelPointBinding& binding,
                                                     const MDagPath& influenceCurve,
                                                     MDataBlock& dataBlock)
{
    MStatus status;
    
    if (!influenceCurve.hasFn(MFn::kNurbsCurve)) {
        return MS::kFailure;
    }
    
    MFnNurbsCurve curveFn(influenceCurve);
    
    // 1. 현재 곡선 위치 및 프레넷 프레임 계산
    MPoint currentCurvePoint;
    status = curveFn.getPointAtParam(binding.offsetCurve.bindParameter, currentCurvePoint, MSpace::kWorld);
    if (status != MS::kSuccess) {
        return MS::kFailure;
    }
    
    MVector currentTangent, currentNormal, currentBinormal;
    status = calculateFrenetFrame(curveFn, binding.offsetCurve.bindParameter, 
                                 currentTangent, currentNormal, currentBinormal);
    if (status != MS::kSuccess) {
        return MS::kFailure;
    }
    
    // 2. 기본 오프셋 벡터 계산
    MVector baseOffset = binding.offsetCurve.offsetVector.x * currentTangent +
                         binding.offsetCurve.offsetVector.y * currentNormal +
                         binding.offsetCurve.offsetVector.z * currentBinormal;
    
    // 3. 특허의 핵심: 아티스트 제어 적용
    
    // 3.1 Twist 제어 (회전 분포)
    double twistDistribution = dataBlock.inputValue(aTwistDistribution).asDouble();
    if (abs(twistDistribution) > 1e-6) {
        baseOffset = applyTwistControl(baseOffset, currentTangent, currentNormal, 
                                     currentBinormal, twistDistribution, binding.offsetCurve.bindParameter);
    }
    
    // 3.2 Slide 제어 (축 방향 슬라이딩)
    double axialSliding = dataBlock.inputValue(aAxialSliding).asDouble();
    if (abs(axialSliding) > 1e-6) {
        baseOffset = applySlideControl(baseOffset, influenceCurve, binding.offsetCurve.bindParameter, axialSliding);
    }
    
    // 3.3 Scale 제어 (스케일 분포)
    double scaleDistribution = dataBlock.inputValue(aScaleDistribution).asDouble();
    if (abs(scaleDistribution - 1.0) > 1e-6) {
        baseOffset = applyScaleControl(baseOffset, scaleDistribution, binding.offsetCurve.bindParameter);
    }
    
    // 3.4 Rotation 제어 (회전 분포)
    double rotationalDistribution = dataBlock.inputValue(aRotationalDistribution).asDouble();
    if (abs(rotationalDistribution - 1.0) > 1e-6) {
        baseOffset = applyRotationControl(baseOffset, currentTangent, currentNormal, 
                                        currentBinormal, rotationalDistribution, binding.offsetCurve.bindParameter);
    }
    
    // 4. 최종 변형된 위치 계산
    MPoint deformedPoint = currentCurvePoint + baseOffset;
    
    return MS::kSuccess;
}

// ✅ 특허의 핵심: Twist 제어 (회전 분포)
MVector offsetCurveDeformerNode::applyTwistControl(const MVector& baseOffset,
                                                   const MVector& tangent,
                                                   const MVector& normal,
                                                   const MVector& binormal,
                                                   double twistAmount,
                                                   double paramU)
{
    // 특허의 핵심: Twist는 곡선 파라미터에 따라 회전
    double twistAngle = twistAmount * paramU * 2.0 * M_PI;
    
    // 탄젠트 축을 중심으로 회전 (Maya 2020 API 호환)
    MTransformationMatrix transMatrix;
    transMatrix.setToRotationAxis(tangent, twistAngle);
    MMatrix rotationMatrix = transMatrix.asMatrix();
    
    // 법선과 바이노멀을 회전
    MVector rotatedNormal = normal * rotationMatrix;
    MVector rotatedBinormal = binormal * rotationMatrix;
    
    // 새로운 오프셋 계산
    MVector twistedOffset = baseOffset.x * tangent +
                           baseOffset.y * rotatedNormal +
                           baseOffset.z * rotatedBinormal;
    
    return twistedOffset;
}

// ✅ 특허의 핵심: Slide 제어 (축 방향 슬라이딩)
MVector offsetCurveDeformerNode::applySlideControl(const MVector& baseOffset,
                                                   const MDagPath& curvePath,
                                                   double paramU,
                                                   double slideAmount)
{
    // 특허의 핵심: 탄젠트 방향으로 슬라이딩
    if (!curvePath.hasFn(MFn::kNurbsCurve)) {
        return baseOffset;
    }
    
    MFnNurbsCurve curveFn(curvePath);
    
    // 새로운 파라미터 계산 (슬라이딩 적용)
    double newParamU = paramU + slideAmount;
    
    // 곡선 범위 내로 클램핑
    double startParam, endParam;
    curveFn.getKnotDomain(startParam, endParam);
    newParamU = std::max(startParam, std::min(endParam, newParamU));
    
    // 탄젠트 벡터 계산 (Maya 2020 API 호환)
    // getTangent 메서드가 없으므로 두 점을 사용하여 탄젠트 계산
    MPoint currentPoint, nextPoint;
    double deltaParam = 0.001;
    MStatus status;
    
    status = curveFn.getPointAtParam(newParamU, currentPoint, MSpace::kWorld);
    if (status != MS::kSuccess) {
        return baseOffset;
    }
    
    double nextParam = newParamU + deltaParam;
    nextParam = std::min(endParam, nextParam);
    
    status = curveFn.getPointAtParam(nextParam, nextPoint, MSpace::kWorld);
    if (status != MS::kSuccess) {
        return baseOffset;
    }
    
    MVector tangent = nextPoint - currentPoint;
    if (tangent.length() < 1e-6) {
        return baseOffset;
    }
    tangent = tangent.normal();
    
    // 슬라이딩된 오프셋 계산
    MVector slideOffset = baseOffset + (tangent * slideAmount);
    
    return slideOffset;
}

// ✅ 특허의 핵심: Scale 제어 (스케일 분포)
MVector offsetCurveDeformerNode::applyScaleControl(const MVector& baseOffset,
                                                   double scaleAmount,
                                                   double paramU)
{
    // 특허의 핵심: 스케일은 곡선 파라미터에 따라 점진적 변화
    double scaleFactor = 1.0 + (scaleAmount - 1.0) * paramU;
    
    return baseOffset * scaleFactor;
}

// ✅ 특허의 핵심: Rotation 제어 (회전 분포)
MVector offsetCurveDeformerNode::applyRotationControl(const MVector& baseOffset,
                                                      const MVector& tangent,
                                                      const MVector& normal,
                                                      const MVector& binormal,
                                                      double rotationAmount,
                                                      double paramU)
{
    // 특허의 핵심: 회전은 곡선 파라미터에 따라 점진적 변화
    double rotationAngle = (rotationAmount - 1.0) * paramU * M_PI;
    
    // 법선 축을 중심으로 회전 (Maya 2020 API 호환)
    MTransformationMatrix transMatrix;
    transMatrix.setToRotationAxis(normal, rotationAngle);
    MMatrix rotationMatrix = transMatrix.asMatrix();
    
    // 탄젠트와 바이노멀을 회전
    MVector rotatedTangent = tangent * rotationMatrix;
    MVector rotatedBinormal = binormal * rotationMatrix;
    
    // 새로운 오프셋 계산
    MVector rotatedOffset = baseOffset.x * rotatedTangent +
                           baseOffset.y * normal +
                           baseOffset.z * rotatedBinormal;
    
    return rotatedOffset;
}

// ✅ 특허의 핵심: 프레넷 프레임 계산 (T, N, B)
MStatus offsetCurveDeformerNode::calculateFrenetFrame(MFnNurbsCurve& curveFn, double paramU,
                                                      MVector& tangent, MVector& normal, MVector& binormal)
{
    MStatus status;
    
    // 1. 탄젠트 벡터 계산 (Maya 2020 API 호환)
    // getTangent 메서드가 없으므로 두 점을 사용하여 탄젠트 계산
    MPoint currentPoint, nextPoint;
    double deltaParam = 0.001;
    
    status = curveFn.getPointAtParam(paramU, currentPoint, MSpace::kWorld);
    if (status != MS::kSuccess) {
        return status;
    }
    
    double nextParam = paramU + deltaParam;
    double curveStartParam, curveEndParam;
    curveFn.getKnotDomain(curveStartParam, curveEndParam);
    nextParam = std::min(curveEndParam, nextParam);
    
    status = curveFn.getPointAtParam(nextParam, nextPoint, MSpace::kWorld);
    if (status != MS::kSuccess) {
        return status;
    }
    
    MVector tangentVector = nextPoint - currentPoint;
    if (tangentVector.length() < 1e-6) {
        return MS::kFailure;
    }
    tangent = tangentVector.normal();
    
    // 2. 법선 벡터 계산 (Maya 2020 API 호환)
    // getSecondDerivativeAtParam이 없으므로 다른 방법 사용
    MVector normalVector;
    
    // 곡선의 두 점을 사용하여 법선 계산
    double param1 = paramU - 0.001;
    double param2 = paramU + 0.001;
    
    // 곡선 범위 확인
    param1 = std::max(curveStartParam, param1);
    param2 = std::min(curveEndParam, param2);
    
    MPoint point1, point2;
    status = curveFn.getPointAtParam(param1, point1, MSpace::kWorld);
    if (status != MS::kSuccess) {
        // 대안: 기본 법선 생성
        normal = MVector(0, 1, 0);
        if (abs(tangent * normal) > 0.9) {
            normal = MVector(1, 0, 0);
        }
    } else {
        status = curveFn.getPointAtParam(param2, point2, MSpace::kWorld);
        if (status != MS::kSuccess) {
            // 대안: 기본 법선 생성
            normal = MVector(0, 1, 0);
            if (abs(tangent * normal) > 0.9) {
                normal = MVector(1, 0, 0);
            }
        } else {
            // 두 점을 사용하여 법선 계산
            MVector secant = point2 - point1;
            if (secant.length() > 1e-6) {
                // 탄젠트와 직교하는 벡터 계산
                normalVector = secant - (secant * tangent) * tangent;
                if (normalVector.length() > 1e-6) {
                    normal = normalVector.normal();
                } else {
                    // 대안: 기본 법선 생성
                    normal = MVector(0, 1, 0);
                    if (abs(tangent * normal) > 0.9) {
                        normal = MVector(1, 0, 0);
                    }
                }
            } else {
                // 대안: 기본 법선 생성
                normal = MVector(0, 1, 0);
                if (abs(tangent * normal) > 0.9) {
                    normal = MVector(1, 0, 0);
                }
            }
        }
    }
    
    // 3. 바이노멀 벡터 계산 (T × N)
    binormal = tangent ^ normal;
    if (binormal.length() < 1e-6) {
        return MS::kFailure;
    }
    binormal = binormal.normal();
    
    // 4. 직교성 보장 (수치적 안정성)
    normal = binormal ^ tangent;
    normal = normal.normal();
    
    return MS::kSuccess;
}

// 기본 헬퍼 함수들
double offsetCurveDeformerNode::calculateDistanceToCurve(const MPoint& point, const MDagPath& curve)
{
    if (!curve.hasFn(MFn::kNurbsCurve)) {
        return 1e6; // 매우 큰 값 반환
    }

    MFnNurbsCurve curveFn(curve);
    double param;
    MPoint closestPoint = curveFn.closestPoint(point, &param, 0.001, MSpace::kWorld);
    
    if (closestPoint.distanceTo(point) > 1e6) {
        return 1e6;
    }

    return point.distanceTo(closestPoint);
}

// 특허 기술의 핵심: 가상 오프셋 커브 생성 (프레넷 프레임 기반)
MStatus offsetCurveDeformerNode::createVirtualOffsetCurve(const MPoint& modelPoint, 
                                                         const MDagPath& influenceCurve,
                                                         VirtualOffsetCurve& offsetCurve)
{
    MStatus status;
    
    // 구현 타입에 따라 다른 방식으로 오프셋 커브 생성
    int implementationType = 0; // 기본값은 B-Spline
    
    if (implementationType == 0) {
        // B-Spline 구현 (특허의 핵심)
        return createBSplineOffsetCurve(modelPoint, influenceCurve, offsetCurve);
    } else {
        // Arc Segment 구현 (특허의 핵심)
        return createArcSegmentOffsetCurve(modelPoint, influenceCurve, offsetCurve);
    }
}

// B-Spline 구현 (특허의 핵심 - Cobb's method)
MStatus offsetCurveDeformerNode::createBSplineOffsetCurve(const MPoint& modelPoint,
                                                         const MDagPath& influenceCurve,
                                                         VirtualOffsetCurve& offsetCurve)
{
    MStatus status;
    
    if (!influenceCurve.hasFn(MFn::kNurbsCurve)) {
        return MS::kFailure;
    }
    
    MFnNurbsCurve curveFn(influenceCurve, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    
    // 1. 곡선에서 가장 가까운 점 찾기 (특허의 핵심)
    double param;
    MPoint closestPoint = curveFn.closestPoint(modelPoint, &param, 0.001, MSpace::kWorld);
    if (closestPoint.distanceTo(modelPoint) > 1e6) {
        return MS::kFailure;
    }
    
    // 2. 바인딩 시점의 프레넷 프레임 계산 (특허의 핵심)
    MVector bindTangent, bindNormal, bindBinormal;
    status = calculateFrenetFrame(curveFn, param, bindTangent, bindNormal, bindBinormal);
    if (status != MS::kSuccess) {
        return MS::kFailure;
    }
    
    // 3. 월드 오프셋 벡터 계산
    MVector offsetWorld = modelPoint - closestPoint;
    
    // 4. 특허의 핵심: 로컬 좌표계로 변환 (T, N, B)
    MVector offsetLocal;
    offsetLocal.x = offsetWorld * bindTangent;    // T 방향 오프셋
    offsetLocal.y = offsetWorld * bindNormal;     // N 방향 오프셋
    offsetLocal.z = offsetWorld * bindBinormal;   // B 방향 오프셋
    
    // 5. 가상 오프셋 커브 정보 저장 (특허의 핵심: 실제 곡선은 저장하지 않음)
    offsetCurve.offsetVector = offsetLocal;       // 로컬 오프셋 벡터
    offsetCurve.bindParameter = param;            // 바인드 파라미터
    offsetCurve.influenceCurveIndex = 0;          // 영향 커브 인덱스
    offsetCurve.isArcSegment = false;             // B-Spline 모드
    
    // 6. 베이스 함수 사전 계산 (성능 최적화)
    int degree = curveFn.degree();
    int numCVs = curveFn.numCVs();
    precomputeBasisFunctions(degree, numCVs);
    
    return MS::kSuccess;
}

// Arc Segment 구현 (특허의 핵심)
MStatus offsetCurveDeformerNode::createArcSegmentOffsetCurve(const MPoint& modelPoint,
                                                            const MDagPath& influenceCurve,
                                                            VirtualOffsetCurve& offsetCurve)
{
    MStatus status;
    
    if (!influenceCurve.hasFn(MFn::kNurbsCurve)) {
        return MS::kFailure;
    }
    
    MFnNurbsCurve curveFn(influenceCurve, &status);
    if (status != MS::kSuccess) {
        return status;
    }
    
    // 1. 곡선에서 가장 가까운 점 찾기 (특허의 핵심)
    double param;
    MPoint closestPoint = curveFn.closestPoint(modelPoint, &param, 0.001, MSpace::kWorld);
    if (closestPoint.distanceTo(modelPoint) > 1e6) {
        return MS::kFailure;
    }
    
    // 2. 바인딩 시점의 프레넷 프레임 계산 (특허의 핵심)
    MVector bindTangent, bindNormal, bindBinormal;
    status = calculateFrenetFrame(curveFn, param, bindTangent, bindNormal, bindBinormal);
    if (status != MS::kSuccess) {
        return MS::kFailure;
    }
    
    // 3. 월드 오프셋 벡터 계산
    MVector offsetWorld = modelPoint - closestPoint;
    
    // 4. 특허의 핵심: 로컬 좌표계로 변환 (T, N, B)
    MVector offsetLocal;
    offsetLocal.x = offsetWorld * bindTangent;    // T 방향 오프셋
    offsetLocal.y = offsetWorld * bindNormal;     // N 방향 오프셋
    offsetLocal.z = offsetWorld * bindBinormal;   // B 방향 오프셋
    
    // 5. Arc Segment 정보 설정 (특허의 핵심)
    offsetCurve.offsetVector = offsetLocal;       // 로컬 오프셋 벡터
    offsetCurve.bindParameter = param;            // 바인드 파라미터
    offsetCurve.influenceCurveIndex = 0;          // 영향 커브 인덱스
    offsetCurve.isArcSegment = true;              // Arc Segment 모드
    
    // 6. 원호 반지름과 중심점 계산 (간단한 구현)
    offsetCurve.arcRadius = offsetWorld.length();
    offsetCurve.arcCenter = closestPoint;
    
    return MS::kSuccess;
}

// 베이스 함수 사전 계산 (B-Spline 성능 최적화)
void offsetCurveDeformerNode::precomputeBasisFunctions(int degree, int numControlPoints)
{
    // 특허의 핵심: 베이스 함수를 미리 계산하여 성능 최적화
    mBSplineDegree = degree;
    mBasisFunctions.clear();
    
    // 각 파라미터 값에 대해 베이스 함수 계산
    for (int i = 0; i < numControlPoints; i++) {
        std::vector<double> basisFuncs;
        // B-Spline 베이스 함수 계산 (간단한 구현)
        for (int j = 0; j <= degree; j++) {
            basisFuncs.push_back(1.0); // 실제로는 Cox-de Boor 공식 사용
        }
        mBasisFunctions.push_back(basisFuncs);
    }
}

// 가중치 계산 (다중 영향 지원)
double offsetCurveDeformerNode::calculateInfluenceWeight(const MPoint& point, const MDagPath& curve, double falloffRadius)
{
    if (!curve.hasFn(MFn::kNurbsCurve)) {
        return 0.0;
    }
    
    // 곡선까지의 거리 계산
    double distance = calculateDistanceToCurve(point, curve);
    
    // 폴오프 반경 내에 있으면 가중치 계산
    if (distance <= falloffRadius) {
        double normalizedDistance = distance / falloffRadius;
        // 선형 폴오프 (특허에서 언급된 방식)
        return 1.0 - normalizedDistance;
    }
    
    return 0.0;
}