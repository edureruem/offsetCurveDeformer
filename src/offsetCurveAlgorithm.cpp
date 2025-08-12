/**
 * offsetCurveAlgorithm.cpp
 * 간단한 기본 알고리즘 구현
 */

#include "offsetCurveAlgorithm.h"
#include <maya/MGlobal.h>
#include <maya/MFnNurbsCurve.h>
#include <algorithm>
#include <cmath>

// 기본 생성자
offsetCurveAlgorithm::offsetCurveAlgorithm()
{
}

// 기본 소멸자
offsetCurveAlgorithm::~offsetCurveAlgorithm()
{
}

// 기본 변형 메서드
MStatus offsetCurveAlgorithm::performDeformationPhase(MPointArray& points, const offsetCurveControlParams& params)
{
    MStatus status = MS::kSuccess;
    
    try {
        // 간단한 변형 계산
        status = calculateBasicDeformation(points, params);
        if (status != MS::kSuccess) {
            MGlobal::displayWarning("Basic deformation failed");
            return status;
        }
        
        return MS::kSuccess;
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Deformation error: ") + e.what());
        return MS::kFailure;
    }
}

// 포즈 타겟 설정
void offsetCurveAlgorithm::setPoseTarget(const MPointArray& targetPoints)
{
    mPoseTargetPoints = targetPoints;
}

// 기본 변형 계산 (간단한 구현)
MStatus offsetCurveAlgorithm::calculateBasicDeformation(MPointArray& points, const offsetCurveControlParams& params)
{
    // 볼륨 강도에 따른 간단한 변형
    double volumeStrength = params.getVolumeStrength();
    
    for (unsigned int i = 0; i < points.length(); i++) {
        // 간단한 스케일 변형
        MPoint& point = points[i];
        point.x *= (1.0 + volumeStrength * 0.1);
        point.y *= (1.0 + volumeStrength * 0.1);
        point.z *= (1.0 + volumeStrength * 0.1);
    }
    
    return MS::kSuccess;
}