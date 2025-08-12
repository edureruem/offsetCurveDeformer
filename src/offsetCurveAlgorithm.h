/**
 * offsetCurveAlgorithm.h
 * OCD 핵심 알고리즘 구현
 * 소니 특허(US8400455) 기반으로 개선
 */

#ifndef OFFSETCURVEALGORITHM_H
#define OFFSETCURVEALGORITHM_H

#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MVector.h>
#include <maya/MVectorArray.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MDagPath.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MThreadPool.h>
#include <vector>
#include <map>
#include <memory>

#include "offsetCurveControlParams.h"

// 간단한 오프셋 모드 enum 정의
enum class OffsetMode {
    ArcSegment = 0,
    BSpline = 1
};

// 기본 알고리즘 클래스 (Strategy Pattern 제거하고 간단하게)
class offsetCurveAlgorithm {
public:
    offsetCurveAlgorithm();
    ~offsetCurveAlgorithm();
    
    // 기본 변형 메서드
    MStatus performDeformationPhase(MPointArray& points, const offsetCurveControlParams& params);
    
    // 포즈 타겟 설정
    void setPoseTarget(const MPointArray& targetPoints);
    
private:
    // 기본 변형 계산
    MStatus calculateBasicDeformation(MPointArray& points, const offsetCurveControlParams& params);
    
    // 포즈 타겟 포인트
    MPointArray mPoseTargetPoints;
};

#endif // OFFSETCURVEALGORITHM_H