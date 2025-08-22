#include "offsetCurveAlgorithm.h"
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnDagNode.h>
#include <maya/MItGeometry.h>
#include <maya/MGlobal.h>
#include <cmath>

OffsetCurveAlgorithm::OffsetCurveAlgorithm() {
}

OffsetCurveAlgorithm::~OffsetCurveAlgorithm() {
}

MStatus OffsetCurveAlgorithm::bindModelPoints(
    const MPointArray& modelPoints,
    const std::vector<OffsetCurveData>& influenceCurves,
    double offsetDistance,
    double falloffRadius,
    MMatrixArray& bindMatrices,
    MPointArray& samplePoints,
    MFloatArray& sampleWeights,
    MVectorArray& offsetVectors) {
    
    MStatus status;
    
    // 각 모델 포인트에 대해 바인딩 계산
    for (unsigned int i = 0; i < modelPoints.length(); i++) {
        // 1. 각 포인트에 대해 오프셋 곡선 생성
        OffsetCurveData offsetCurve;
        status = createOffsetCurveForPoint(modelPoints[i], influenceCurves, offsetDistance, offsetCurve);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        // 2. 가장 가까운 영향 곡선 찾기
        double minDistance = std::numeric_limits<double>::max();
        int closestCurveIndex = -1;
        double closestParameter = 0.0;
        
        for (size_t j = 0; j < influenceCurves.size(); j++) {
            double param;
            MPoint closestPoint;
            status = findClosestPointOnCurve(modelPoints[i], influenceCurves[j], closestPoint, param);
            if (status == MS::kSuccess) {
                double distance = modelPoints[i].distanceTo(closestPoint);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCurveIndex = j;
                    closestParameter = param;
                }
            }
        }
        
        if (closestCurveIndex == -1) {
            continue; // 유효한 곡선을 찾지 못함
        }
        
        // 3. 바인딩 매트릭스 생성
        MMatrix bindMatrix = MMatrix::identity;
        if (closestCurveIndex < (int)influenceCurves.size()) {
            const OffsetCurveData& curve = influenceCurves[closestCurveIndex];
            
            // 곡선의 접선과 법선 계산
            MVector tangent, normal;
            if (curve.type == kBSpline) {
                tangent = curve.bspline.tangentAt(closestParameter);
                normal = curve.bspline.normalAt(closestParameter);
            } else {
                tangent = curve.arc.tangentAt(closestParameter);
                normal = curve.arc.normalAt(closestParameter);
            }
            
            // 바인딩 매트릭스 구성
            MVector up = tangent ^ normal;
            up.normalize();
            
            // 3x3 회전 매트릭스
            bindMatrix[0][0] = tangent.x; bindMatrix[0][1] = tangent.y; bindMatrix[0][2] = tangent.z;
            bindMatrix[1][0] = normal.x;  bindMatrix[1][1] = normal.y;  bindMatrix[1][2] = normal.z;
            bindMatrix[2][0] = up.x;      bindMatrix[2][1] = up.y;      bindMatrix[2][2] = up.z;
            
            // 위치
            MPoint curvePoint;
            if (curve.type == kBSpline) {
                curvePoint = curve.bspline.evaluateAt(closestParameter);
            } else {
                curvePoint = curve.arc.evaluateAt(closestParameter);
            }
            bindMatrix[3][0] = curvePoint.x;
            bindMatrix[3][1] = curvePoint.y;
            bindMatrix[3][2] = curvePoint.z;
        }
        
        // 4. 바인딩 데이터 저장
        bindMatrices.append(bindMatrix);
        samplePoints.append(modelPoints[i]);
        
        // 가중치 계산 (거리 기반)
        float weight = calculateInfluenceWeight(modelPoints[i], influenceCurves[closestCurveIndex], falloffRadius);
        sampleWeights.append(weight);
        
        // 오프셋 벡터 계산
        MVector offsetVector = modelPoints[i] - samplePoints[i];
        offsetVectors.append(offsetVector);
    }
    
    return MS::kSuccess;
}

MStatus OffsetCurveAlgorithm::deformModelPoints(
    const MPointArray& originalPoints,
    const MMatrixArray& bindMatrices,
    const MPointArray& samplePoints,
    const MFloatArray& sampleWeights,
    const MVectorArray& offsetVectors,
    const std::vector<OffsetCurveData>& deformedInfluenceCurves,
    MPointArray& deformedPoints) {
    
    MStatus status;
    deformedPoints.setLength(originalPoints.length());
    
    for (unsigned int i = 0; i < originalPoints.length(); i++) {
        if (i >= bindMatrices.length()) {
            deformedPoints[i] = originalPoints[i];
            continue;
        }
        
        // 1. 현재 영향 곡선 상태에서 오프셋 곡선 실시간 계산
        OffsetCurveData currentOffsetCurve;
        status = createOffsetCurveForPoint(originalPoints[i], deformedInfluenceCurves, 
                                         offsetVectors[i].length(), currentOffsetCurve);
        if (status != MS::kSuccess) {
            deformedPoints[i] = originalPoints[i];
            continue;
        }
        
        // 2. 바인딩된 파라미터에서 오프셋 곡선의 위치 계산
        // (실제로는 바인딩 시점의 파라미터를 저장해야 하지만, 여기서는 단순화)
        MPoint deformedPoint = currentOffsetCurve.evaluateAt(0.5); // 중간점 사용
        
        // 3. 최종 위치 계산
        deformedPoints[i] = deformedPoint;
    }
    
    return MS::kSuccess;
}

MStatus OffsetCurveAlgorithm::createOffsetCurveForPoint(
    const MPoint& modelPoint,
    const std::vector<OffsetCurveData>& influenceCurves,
    double offsetDistance,
    OffsetCurveData& offsetCurve) {
    
    MStatus status;
    
    // 가장 가까운 영향 곡선 찾기
    double minDistance = std::numeric_limits<double>::max();
    int closestCurveIndex = -1;
    
    for (size_t i = 0; i < influenceCurves.size(); i++) {
        double distance = calculateDistanceToCurve(modelPoint, influenceCurves[i]);
        if (distance < minDistance) {
            minDistance = distance;
            closestCurveIndex = i;
        }
    }
    
    if (closestCurveIndex == -1) {
        return MS::kFailure;
    }
    
    const OffsetCurveData& baseCurve = influenceCurves[closestCurveIndex];
    
    // 곡선 타입에 따라 오프셋 곡선 생성
    if (baseCurve.type == kBSpline) {
        status = createBSplineOffsetCurve(baseCurve.bspline, modelPoint, offsetDistance, offsetCurve.bspline);
        offsetCurve.type = kBSpline;
    } else {
        status = createArcSegmentOffsetCurve(baseCurve.arc, modelPoint, offsetDistance, offsetCurve.arc);
        offsetCurve.type = kArcSegment;
    }
    
    return status;
}

MStatus OffsetCurveAlgorithm::createBSplineOffsetCurve(
    const BSplineCurve& baseCurve,
    const MPoint& offsetPoint,
    double distance,
    BSplineCurve& offsetCurve) {
    
    // B-spline 오프셋 곡선 생성 (단순화된 구현)
    offsetCurve = baseCurve; // 기본적으로 복사
    
    // 각 제어점을 오프셋 방향으로 이동
    for (unsigned int i = 0; i < baseCurve.controlPoints.length(); i++) {
        MVector offsetDirection = (baseCurve.controlPoints[i] - offsetPoint).normal();
        offsetCurve.controlPoints[i] = baseCurve.controlPoints[i] + offsetDirection * distance;
    }
    
    return MS::kSuccess;
}

MStatus OffsetCurveAlgorithm::createArcSegmentOffsetCurve(
    const ArcSegmentCurve& baseCurve,
    const MPoint& offsetPoint,
    double distance,
    ArcSegmentCurve& offsetCurve) {
    
    // Arc-segment 오프셋 곡선 생성
    offsetCurve = baseCurve; // 기본적으로 복사
    
    // 시작점과 끝점을 오프셋
    MVector startOffset = (baseCurve.startPoint - offsetPoint).normal();
    MVector endOffset = (baseCurve.endPoint - offsetPoint).normal();
    
    offsetCurve.startPoint = baseCurve.startPoint + startOffset * distance;
    offsetCurve.endPoint = baseCurve.endPoint + endOffset * distance;
    
    // 중심점도 오프셋
    MVector centerOffset = (baseCurve.centerPoint - offsetPoint).normal();
    offsetCurve.centerPoint = baseCurve.centerPoint + centerOffset * distance;
    
    return MS::kSuccess;
}

double OffsetCurveAlgorithm::calculateDistanceToCurve(
    const MPoint& point, const OffsetCurveData& curve) {
    
    double minDistance = std::numeric_limits<double>::max();
    
    if (curve.type == kBSpline) {
        // B-spline 곡선에서 최단 거리 계산
        for (double t = 0.0; t <= 1.0; t += 0.01) {
            MPoint curvePoint = curve.bspline.evaluateAt(t);
            double distance = point.distanceTo(curvePoint);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    } else {
        // Arc-segment 곡선에서 최단 거리 계산
        for (double t = 0.0; t <= 1.0; t += 0.01) {
            MPoint curvePoint = curve.arc.evaluateAt(t);
            double distance = point.distanceTo(curvePoint);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    }
    
    return minDistance;
}

MStatus OffsetCurveAlgorithm::findClosestPointOnCurve(
    const MPoint& point,
    const OffsetCurveData& curve,
    MPoint& closestPoint,
    double& parameter) {
    
    double minDistance = std::numeric_limits<double>::max();
    double bestParameter = 0.0;
    
    if (curve.type == kBSpline) {
        // B-spline 곡선에서 가장 가까운 점 찾기
        for (double t = 0.0; t <= 1.0; t += 0.01) {
            MPoint curvePoint = curve.bspline.evaluateAt(t);
            double distance = point.distanceTo(curvePoint);
            if (distance < minDistance) {
                minDistance = distance;
                bestParameter = t;
                closestPoint = curvePoint;
            }
        }
    } else {
        // Arc-segment 곡선에서 가장 가까운 점 찾기
        for (double t = 0.0; t <= 1.0; t += 0.01) {
            MPoint curvePoint = curve.arc.evaluateAt(t);
            double distance = point.distanceTo(curvePoint);
            if (distance < minDistance) {
                minDistance = distance;
                bestParameter = t;
                closestPoint = curvePoint;
            }
        }
    }
    
    parameter = bestParameter;
    return MS::kSuccess;
}

float OffsetCurveAlgorithm::calculateInfluenceWeight(
    const MPoint& point,
    const OffsetCurveData& curve,
    double falloffRadius) {
    
    double distance = calculateDistanceToCurve(point, curve);
    
    if (distance >= falloffRadius) {
        return 0.0f;
    }
    
    // 선형 감쇠
    float weight = 1.0f - (float)(distance / falloffRadius);
    return std::max(0.0f, std::min(1.0f, weight));
}

MStatus OffsetCurveAlgorithm::adjustOffsetVectorForVolumePreservation(
    const MPoint& originalPoint,
    const MPoint& deformedPoint,
    const MVector& originalOffset,
    MVector& adjustedOffset) {
    
    // 볼륨 보존을 위한 오프셋 벡터 조정
    // (특허의 핵심 아이디어 구현)
    
    MVector deformationVector = deformedPoint - originalPoint;
    double deformationLength = deformationVector.length();
    
    if (deformationLength < 0.001) {
        adjustedOffset = originalOffset;
        return MS::kSuccess;
    }
    
    // 변형 방향에 따른 오프셋 조정
    MVector deformationDirection = deformationVector.normal();
    double dotProduct = originalOffset * deformationDirection;
    
    if (dotProduct > 0) {
        // 변형 방향과 같은 방향: 스트레칭 감소
        adjustedOffset = originalOffset - deformationDirection * (dotProduct * 0.5);
    } else {
        // 변형 방향과 반대 방향: 자체 교차 방지
        adjustedOffset = originalOffset - deformationDirection * (dotProduct * 0.3);
    }
    
    return MS::kSuccess;
}

// B-spline 곡선 메서드들
MPoint BSplineCurve::evaluateAt(double t) const {
    t = clampParameter(t);
    
    // 단순화된 B-spline 계산 (실제로는 더 정교한 구현 필요)
    if (controlPoints.length() < 2) {
        return MPoint::origin;
    }
    
    if (t <= 0.0) return controlPoints[0];
    if (t >= 1.0) return controlPoints[controlPoints.length() - 1];
    
    // 선형 보간 (실제 B-spline 구현으로 대체 필요)
    int index = (int)(t * (controlPoints.length() - 1));
    double localT = t * (controlPoints.length() - 1) - index;
    
    if (index >= (int)controlPoints.length() - 1) {
        return controlPoints[controlPoints.length() - 1];
    }
    
    return controlPoints[index] * (1.0 - localT) + controlPoints[index + 1] * localT;
}

MVector BSplineCurve::tangentAt(double t) const {
    t = clampParameter(t);
    
    // 단순화된 접선 계산
    double delta = 0.001;
    MPoint p1 = evaluateAt(t);
    MPoint p2 = evaluateAt(t + delta);
    
    return (p2 - p1).normal();
}

MVector BSplineCurve::normalAt(double t) const {
    t = clampParameter(t);
    
    // 단순화된 법선 계산
    MVector tangent = tangentAt(t);
    MVector up(0, 1, 0);
    
    if (std::abs(tangent * up) > 0.9) {
        up = MVector(1, 0, 0);
    }
    
    return (tangent ^ up).normal();
}

// Arc-segment 곡선 메서드들
MPoint ArcSegmentCurve::evaluateAt(double t) const {
    t = clampParameter(t);
    
    if (isArc) {
        // 원호 보간
        double angle = startAngle + (endAngle - startAngle) * t;
        double x = centerPoint.x + radius * cos(angle);
        double y = centerPoint.y + radius * sin(angle);
        double z = centerPoint.z;
        return MPoint(x, y, z);
    } else {
        // 선분 보간
        return startPoint * (1.0 - t) + endPoint * t;
    }
}

MVector ArcSegmentCurve::tangentAt(double t) const {
    t = clampParameter(t);
    
    if (isArc) {
        // 원호 접선
        double angle = startAngle + (endAngle - startAngle) * t;
        double dx = -radius * sin(angle);
        double dy = radius * cos(angle);
        return MVector(dx, dy, 0).normal();
    } else {
        // 선분 접선
        return (endPoint - startPoint).normal();
    }
}

MVector ArcSegmentCurve::normalAt(double t) const {
    t = clampParameter(t);
    
    if (isArc) {
        // 원호 법선 (중심 방향)
        double angle = startAngle + (endAngle - startAngle) * t;
        double dx = radius * cos(angle);
        double dy = radius * sin(angle);
        return MVector(dx, dy, 0).normal();
    } else {
        // 선분 법선 (2D에서 수직)
        MVector direction = (endPoint - startPoint).normal();
        return MVector(-direction.y, direction.x, 0);
    }
}

// 유틸리티 메서드들
double OffsetCurveAlgorithm::clampParameter(double t) const {
    return std::max(0.0, std::min(1.0, t));
}

bool OffsetCurveAlgorithm::isPointInArcSegment(const MPoint& point, const ArcSegmentCurve& arc) const {
    if (!arc.isArc) {
        // 선분인 경우 단순 거리 체크
        MVector toStart = point - arc.startPoint;
        MVector toEnd = point - arc.endPoint;
        MVector segment = arc.endPoint - arc.startPoint;
        
        if (toStart * segment < 0 || toEnd * segment > 0) {
            return false;
        }
        
        MVector perpendicular = segment ^ MVector(0, 0, 1);
        double distance = std::abs(toStart * perpendicular.normal());
        return distance <= arc.radius;
    } else {
        // 원호인 경우 각도 체크
        MVector toCenter = point - arc.centerPoint;
        double distance = toCenter.length();
        
        if (std::abs(distance - arc.radius) > 0.001) {
            return false;
        }
        
        double angle = atan2(toCenter.y, toCenter.x);
        
        // 각도 정규화
        while (angle < 0) angle += 2 * M_PI;
        while (arc.startAngle < 0) arc.startAngle += 2 * M_PI;
        while (arc.endAngle < 0) arc.endAngle += 2 * M_PI;
        
        if (arc.startAngle <= arc.endAngle) {
            return angle >= arc.startAngle && angle <= arc.endAngle;
        } else {
            return angle >= arc.startAngle || angle <= arc.endAngle;
        }
    }
}
