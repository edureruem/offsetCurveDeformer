#include "offsetCurveSystems.h"
#include <maya/MFnDependencyNode.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MGlobal.h>

OffsetCurveSystems::OffsetCurveSystems() {
}

OffsetCurveSystems::~OffsetCurveSystems() {
}

MStatus OffsetCurveSystems::initialize() {
    // 시스템 초기화
    return MS::kSuccess;
}

MStatus OffsetCurveSystems::uninitialize() {
    // 시스템 정리
    return MS::kSuccess;
}

MStatus OffsetCurveSystems::registerCurveSystems() {
    // 곡선 타입별 시스템 등록
    MStatus status;
    
    status = registerBSplineSystem();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = registerArcSegmentSystem();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    return MS::kSuccess;
}

MStatus OffsetCurveSystems::unregisterCurveSystems() {
    // 곡선 타입별 시스템 해제
    return MS::kSuccess;
}

MStatus OffsetCurveSystems::registerBSplineSystem() {
    // B-spline 시스템 등록
    return MS::kSuccess;
}

MStatus OffsetCurveSystems::registerArcSegmentSystem() {
    // Arc-segment 시스템 등록
    return MS::kSuccess;
}

bool OffsetCurveSystems::validateSystem(const MObject& systemNode) {
    // 시스템 유효성 검사
    if (systemNode.isNull()) {
        return false;
    }
    
    // 노드가 곡선인지 확인
    MFnDependencyNode fnNode(systemNode);
    if (fnNode.type() == MFn::kNurbsCurve) {
        return true;
    }
    
    return false;
}

int OffsetCurveSystems::getSystemType(const MObject& systemNode) {
    // 시스템 타입 확인
    if (validateSystem(systemNode)) {
        // 기본적으로 B-spline으로 처리
        return 0; // kBSpline
    }
    
    return -1; // 유효하지 않은 시스템
}

MStatus OffsetCurveSystems::extractSystemParameters(const MObject& systemNode,
                                                   MString& systemType,
                                                   MObject& controlObject) {
    // 시스템 파라미터 추출
    MStatus status;
    
    if (!validateSystem(systemNode)) {
        return MS::kFailure;
    }
    
    MFnDependencyNode fnNode(systemNode, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 시스템 타입 설정
    systemType = "B-spline"; // 기본값
    
    // 제어 객체 설정
    controlObject = systemNode;
    
    return MS::kSuccess;
}
