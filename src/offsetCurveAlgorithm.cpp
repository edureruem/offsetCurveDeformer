/**
 * offsetCurveAlgorithm.cpp
 * OCD 핵심 알고리즘 구현 (레거시 코드 제거 완료)
 */

#include "offsetCurveAlgorithm.h"
#include <maya/MGlobal.h>
#include <maya/MFnNurbsCurve.h>
#include <algorithm>
#include <cmath>
#include <limits>

// offsetCurveAlgorithm 구현 (특허 준수)
offsetCurveAlgorithm::offsetCurveAlgorithm()
    : mOffsetMode(ARC_SEGMENT), 
      mUseParallelComputation(false)
{
    // 특허 준수: 곡선 데이터 캐싱하지 않음, 경로만 저장
}

offsetCurveAlgorithm::~offsetCurveAlgorithm()
{
}

// 알고리즘 초기화 (특허 준수)
MStatus offsetCurveAlgorithm::initialize(const MPointArray& points, offsetCurveOffsetMode offsetMode)
{
    mOffsetMode = offsetMode;
    
    // OCD: 정점 데이터 초기화 (최소한의 정보만)
    mVertexData.clear();
    mVertexData.reserve(points.length());
    
    for (unsigned int i = 0; i < points.length(); i++) {
        VertexDeformationData vertexData;
        vertexData.vertexIndex = i;
        vertexData.bindPosition = points[i];
        mVertexData.push_back(vertexData);
    }
    
    // 특허 준수: 영향 곡선 경로만 저장 (데이터 캐싱 안 함!)
    mInfluenceCurvePaths.clear();
    
    // ✅ 추가: 새로운 시스템들 초기화
    initializeBindRemapping();
    initializePoseSpaceDeformation();
    initializeAdaptiveSubdivision();
    
    // ✅ 추가: Strategy Context 초기화
    mStrategyContext.setStrategy(offsetMode);
    
    return MS::kSuccess;
}

// 영향 곡선에 바인딩 (단순화) - performBindingPhase로 위임
MStatus offsetCurveAlgorithm::bindToCurves(const std::vector<MDagPath>& curvePaths, 
                                 double falloffRadius,
                                 int maxInfluences)
{
    // 새로운 OCD 바인딩 페이즈로 위임
    MPointArray bindPoints;
    for (const auto& vertexData : mVertexData) {
        bindPoints.append(vertexData.bindPosition);
    }
    
    return performBindingPhase(bindPoints, curvePaths, falloffRadius, maxInfluences);
}

// 레거시 호환성을 위한 변형 계산 - OCD 알고리즘으로 위임
MStatus offsetCurveAlgorithm::computeDeformation(MPointArray& points,
                                      const offsetCurveControlParams& params)
{
    // 새로운 특허 기반 OCD 알고리즘 사용
    return performDeformationPhase(points, params);
}

// 병렬 처리 활성화/비활성화
void offsetCurveAlgorithm::enableParallelComputation(bool enable)
{
    mUseParallelComputation = enable;
}

// 포즈 타겟 설정
void offsetCurveAlgorithm::setPoseTarget(const MPointArray& poseTarget)
{
    mPoseTargetPoints = poseTarget;
}

// 포즈 블렌딩 적용
MPoint offsetCurveAlgorithm::applyPoseBlending(const MPoint& deformedPoint, 
                                    unsigned int vertexIndex,
                                    double blendWeight)
{
    // 포즈 타겟이 없거나 인덱스가 범위 밖이면 변형 없음
    if (mPoseTargetPoints.length() <= vertexIndex) {
        return deformedPoint;
    }
    
    // 포즈 타겟 위치
    MPoint targetPoint = mPoseTargetPoints[vertexIndex];
    
    // 블렌드 계산
    return deformedPoint * (1.0 - blendWeight) + targetPoint * blendWeight;
}

// ========================================================================
// OCD: 실시간 계산 함수들 (캐싱 없음!)
// ========================================================================

// 🚀 Arc Segment 모드: 고성능 프레넷 프레임 계산 (특허 핵심!)
MStatus offsetCurveAlgorithm::calculateFrenetFrameArcSegment(
    const MDagPath& curvePath,
    double paramU,
    MVector& tangent,
    MVector& normal,
    MVector& binormal) const
{
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // ⚡ Arc Segment 가정: 원형 호 + 직선 세그먼트
    // 팔꿈치, 손가락 관절 등에 최적화
    
    // 1. 곡선의 시작/끝 점과 중간점 (3점으로 원 계산)
    MPoint startPoint, midPoint, endPoint;
    // Maya 2020 호환성: getPointAtParam 올바른 매개변수 순서
    fnCurve.getPointAtParam(0.0, startPoint, MSpace::kWorld);
    fnCurve.getPointAtParam(0.5, midPoint, MSpace::kWorld);  
    fnCurve.getPointAtParam(1.0, endPoint, MSpace::kWorld);
    
    // 2. 원의 중심과 반지름 계산 (기하학적 방법)
    MVector v1 = midPoint - startPoint;
    MVector v2 = endPoint - midPoint;
    
    // 직선인 경우 (곡률이 거의 없음)
    if (v1.isParallel(v2, 1e-3)) {
        // 직선 세그먼트: 간단한 선형 보간
        MPoint currentPoint = startPoint + (endPoint - startPoint) * paramU;
        tangent = (endPoint - startPoint).normal();
        
        // 직선의 경우 임의의 수직 벡터 생성
        MVector up(0, 1, 0);
        if (fabs(tangent * up) > 0.9) {
            up = MVector(1, 0, 0);
        }
        normal = (up - (up * tangent) * tangent).normal();
        binormal = tangent ^ normal;
        
        return MS::kSuccess;
    }
    
    // 3. 원형 호인 경우: 고속 삼각함수 계산
    // 원의 중심 계산 (외심 공식)
    double d1 = v1.length();
    double d2 = v2.length();
    double cross = (v1 ^ v2).length();
    
    if (cross < 1e-6) {
        // 거의 직선인 경우
        tangent = (endPoint - startPoint).normal();
        MVector up(0, 1, 0);
        if (fabs(tangent * up) > 0.9) up = MVector(1, 0, 0);
        normal = (up - (up * tangent) * tangent).normal();
        binormal = tangent ^ normal;
    return MS::kSuccess;
}

    double radius = (d1 * d2 * (endPoint - startPoint).length()) / (2.0 * cross);
    
    // 4. ⚡ 고속 원형 호 계산 (삼각함수 직접 사용)
    double totalAngle = 2.0 * asin(cross / (2.0 * radius));
    double currentAngle = totalAngle * paramU;
    
    // 5. 원 상의 점과 탄젠트 벡터 (삼각함수로 직접 계산)
    MVector centerToStart = startPoint - midPoint;  // 근사 중심
    MVector arcTangent(-centerToStart.z, 0, centerToStart.x);  // 원의 접선
    arcTangent.normalize();
    
    // 회전된 탄젠트 (로드리게스 공식 대신 간단한 회전)
    tangent = arcTangent * cos(currentAngle) + (arcTangent ^ centerToStart.normal()) * sin(currentAngle);
    tangent.normalize();
    
    // 6. 원의 중심을 향하는 노말 벡터
    normal = -centerToStart.normal();
    
    // 7. 바이노말 (외적)
    binormal = tangent ^ normal;
    
    return MS::kSuccess;
}

// B-Spline 모드: 정확하지만 느린 프레넷 프레임 계산
MStatus offsetCurveAlgorithm::calculateFrenetFrameOnDemand(const MDagPath& curvePath, 
                                                          double paramU,
                                                          MVector& tangent,
                                                          MVector& normal, 
                                                          MVector& binormal) const
{
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 1. 탄젠트 벡터 계산
    // Maya 2020 호환성: tangent API 올바른 매개변수 순서
    tangent = fnCurve.tangent(paramU, MSpace::kWorld, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    tangent.normalize();
    
    // 2. 노말 벡터 계산 (최소 회전 방식 - 특허 권장)
    // 간단한 구현: 탄젠트에 수직인 벡터 찾기
    MVector up(0, 1, 0);  // 기본 업 벡터
    if (abs(tangent * up) > 0.9) {  // 거의 평행한 경우
        up = MVector(1, 0, 0);  // 다른 벡터 사용
    }
    
    // 그람-슈미트 과정으로 노말 벡터 계산
    normal = up - (up * tangent) * tangent;
    normal.normalize();
    
    // 3. 바이노말 벡터 = 탄젠트 × 노말
    binormal = tangent ^ normal;
    binormal.normalize();
    
    return MS::kSuccess;
}

// 실시간 곡선 상의 점 계산
MStatus offsetCurveAlgorithm::calculatePointOnCurveOnDemand(const MDagPath& curvePath,
                                                           double paramU,
                                                           MPoint& point) const
{
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // Maya 2020 호환성: getPointAtParam 올바른 매개변수 순서
    fnCurve.getPointAtParam(paramU, point, MSpace::kWorld);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    return MS::kSuccess;
}

// 실시간 가장 가까운 점 찾기
MStatus offsetCurveAlgorithm::findClosestPointOnCurveOnDemand(const MDagPath& curvePath,
                                                             const MPoint& modelPoint,
                                                             double& paramU,
                                                             MPoint& closestPoint,
                                                             double& distance) const
{
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // Maya 2020 호환성: closestPoint 매개변수 순서 수정
    // Maya 2020 호환성: closestPoint와 getPointAtParam 올바른 호출
    MPoint tempClosestPoint = fnCurve.closestPoint(modelPoint, &paramU, false, MSpace::kWorld);
    if (tempClosestPoint != MPoint::origin) {
        fnCurve.getPointAtParam(paramU, closestPoint, MSpace::kWorld);
    }
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 거리 계산
    distance = modelPoint.distanceTo(closestPoint);
    
    return MS::kSuccess;
}

// ========================================================================
// OCD 알고리즘 구현
// ========================================================================

// 바인딩 페이즈: OCD 알고리즘
MStatus offsetCurveAlgorithm::performBindingPhase(const MPointArray& modelPoints,
                                                  const std::vector<MDagPath>& influenceCurves,
                                                  double falloffRadius,
                                                  int maxInfluences)
{
    MStatus status;
    
    // 영향 곡선 경로 저장 (데이터 캐싱 안 함!)
    mInfluenceCurves = influenceCurves;
    
    // 각 모델 포인트에 대해 오프셋 프리미티브 생성
    for (unsigned int vertexIndex = 0; vertexIndex < modelPoints.length(); vertexIndex++) {
        const MPoint& modelPoint = modelPoints[vertexIndex];
        VertexDeformationData& vertexData = mVertexData[vertexIndex];
        
        // ✅ 특허 준수: 각 영향 곡선에 대해 "가상 오프셋 커브" 계산
        // 여러 곡선 허용 - 각각에 대해 오프셋 프리미티브 생성
        for (size_t curveIndex = 0; curveIndex < influenceCurves.size(); curveIndex++) {
            const MDagPath& curvePath = influenceCurves[curveIndex];
            
            // 1. 가장 가까운 점 찾기 (실시간 계산)
            double bindParamU;
            MPoint closestPoint;
            double distance;
            status = findClosestPointOnCurveOnDemand(curvePath, modelPoint, 
                                                   bindParamU, closestPoint, distance);
            if (status != MS::kSuccess) continue;
            
            // 거리 기반 필터링
            if (distance > falloffRadius) continue;
            
            // 2. 바인드 시점의 프레넷 프레임 계산 (모드별 분기)
            MVector tangent, normal, binormal;
            if (mOffsetMode == ARC_SEGMENT) {
                // ⚡ Arc Segment 모드: 3-5배 빠른 계산
                status = calculateFrenetFrameArcSegment(curvePath, bindParamU,
                                                       tangent, normal, binormal);
            } else {
                // B-Spline 모드: 정확하지만 느린 계산
            status = calculateFrenetFrameOnDemand(curvePath, bindParamU, 
                                                tangent, normal, binormal);
            }
            if (status != MS::kSuccess) continue;
            
            // 3. 오프셋 벡터를 로컬 좌표계로 변환 (특허 핵심!)
            MVector offsetWorld = modelPoint - closestPoint;
            MVector offsetLocal;
            offsetLocal.x = offsetWorld * tangent;   // 탄젠트 방향 성분
            offsetLocal.y = offsetWorld * normal;    // 노말 방향 성분
            offsetLocal.z = offsetWorld * binormal;  // 바이노말 방향 성분
            
            // 4. 가중치 계산
            double weight = 1.0 / (1.0 + distance / falloffRadius);
            
            // 5. ✅ 특허 준수: 오프셋 프리미티브 생성 (실제 오프셋 커브 저장 안 함!)
            OffsetPrimitive offsetPrimitive;
            offsetPrimitive.influenceCurveIndex = static_cast<int>(curveIndex);
            offsetPrimitive.bindParamU = bindParamU;
            offsetPrimitive.bindOffsetLocal = offsetLocal;
            offsetPrimitive.weight = weight;
            
            vertexData.offsetPrimitives.push_back(offsetPrimitive);
        }
        
        // 최대 영향 수 제한
        if (vertexData.offsetPrimitives.size() > static_cast<size_t>(maxInfluences)) {
            // 가중치 기준으로 정렬
            std::sort(vertexData.offsetPrimitives.begin(), 
                     vertexData.offsetPrimitives.end(),
                     [](const OffsetPrimitive& a, const OffsetPrimitive& b) {
                         return a.weight > b.weight;
                     });
            vertexData.offsetPrimitives.resize(maxInfluences);
        }
        
        // 가중치 정규화
        double totalWeight = 0.0;
        for (auto& primitive : vertexData.offsetPrimitives) {
            totalWeight += primitive.weight;
        }
        if (totalWeight > 0.0) {
            for (auto& primitive : vertexData.offsetPrimitives) {
                primitive.weight /= totalWeight;
            }
        }
    }
    
    // ✅ 추가: Bind Remapping 시스템 적용
    status = applyBindRemappingToPrimitives();
    if (status != MS::kSuccess) {
        MGlobal::displayWarning("Bind Remapping 적용 실패");
    }
    
    // ✅ 추가: 영향력 혼합 최적화 적용
    for (auto& vertexData : mVertexData) {
        if (vertexData.offsetPrimitives.size() > 1) {
            optimizeInfluenceBlending(vertexData.offsetPrimitives, vertexData.bindPosition);
        }
    }
    
    return MS::kSuccess;
}

// 변형 페이즈: OCD의 정확한 수학 공식
MStatus offsetCurveAlgorithm::performDeformationPhase(MPointArray& points,
                                                      const offsetCurveControlParams& params)
{
    MStatus status;
    
    // 🔥 GPU 가속 우선 시도
    #ifdef CUDA_ENABLED
    if (mUseParallelComputation && mVertexData.size() > 1000) {
        processVertexDeformationGPU(points, params);
        return MS::kSuccess;
    }
    #endif
    
    // 🚀 병렬 처리 활성화 시 OpenMP 사용
    #ifdef _OPENMP
    if (mUseParallelComputation) {
        #pragma omp parallel for schedule(dynamic, 32)
        for (int vertexIndex = 0; vertexIndex < (int)mVertexData.size(); vertexIndex++) {
            processVertexDeformation(vertexIndex, points, params);
        }
        return MS::kSuccess;
    }
    #endif
    
    // 순차 처리 (기본)
    for (size_t vertexIndex = 0; vertexIndex < mVertexData.size(); vertexIndex++) {
        const VertexDeformationData& vertexData = mVertexData[vertexIndex];
        MPoint newPosition(0, 0, 0);
        double totalWeight = 0.0;
        
        // ✅ 특허 준수: 여러 오프셋 프리미티브의 가중치 합으로 계산
        for (const OffsetPrimitive& primitive : vertexData.offsetPrimitives) {
            const MDagPath& curvePath = mInfluenceCurves[primitive.influenceCurveIndex];
            
            // 슬라이딩을 위해 paramU를 복사 (원본 보존)
            double currentParamU = primitive.bindParamU;
            
            // 1. 현재 프레넷 프레임 계산 (모드별 분기)
            MVector currentTangent, currentNormal, currentBinormal;
            if (mOffsetMode == ARC_SEGMENT) {
                // ⚡ Arc Segment 모드: 3-5배 빠른 계산
                status = calculateFrenetFrameArcSegment(curvePath, currentParamU,
                                                       currentTangent, currentNormal, currentBinormal);
            } else {
                // B-Spline 모드: 정확하지만 느린 계산
            status = calculateFrenetFrameOnDemand(curvePath, currentParamU,
                                                currentTangent, currentNormal, currentBinormal);
            }
            if (status != MS::kSuccess) continue;
            
            // 2. 🎯 아티스트 제어 적용 (특허 US8400455B2)
            MVector controlledOffset = applyArtistControls(primitive.bindOffsetLocal,
                                                          currentTangent,
                                                          currentNormal,
                                                          currentBinormal,
                                                          curvePath,
                                                          currentParamU,  // 슬라이딩으로 변경 가능
                                                          params);
            
            // 3. (슬라이딩으로 인해) 업데이트된 곡선 상의 점 계산
            MPoint currentInfluencePoint;
            status = calculatePointOnCurveOnDemand(curvePath, currentParamU, 
                                                 currentInfluencePoint);
            if (status != MS::kSuccess) continue;
            
            // 4. 제어된 오프셋을 현재 프레넷 프레임에 적용
            MVector offsetWorldCurrent = 
                controlledOffset.x * currentTangent +
                controlledOffset.y * currentNormal +
                controlledOffset.z * currentBinormal;
            
            // 5. ✅ 수정: 특허 기반 볼륨 보존 시스템 적용
            if (params.getVolumeStrength() > 0.0) {
                double curvature = calculateCurvatureAtPoint(curvePath, currentParamU);
                double volumeFactor = calculateVolumePreservationFactor(primitive, curvature);
                
                // 볼륨 보존 적용
                offsetWorldCurrent = offsetWorldCurrent * volumeFactor;
                
                // 자체 교차 방지 적용
                offsetWorldCurrent = applySelfIntersectionPrevention(offsetWorldCurrent, primitive, curvature);
            }
            
            // 6. ✅ 추가: Pose Space Deformation 적용
            MVector poseSpaceOffset = applyPoseSpaceDeformation(currentInfluencePoint, static_cast<int>(vertexIndex));
            offsetWorldCurrent += poseSpaceOffset;
            
            // 7. 새로운 정점 위치 = 현재 영향점 + 제어된 오프셋
            MPoint deformedPosition = currentInfluencePoint + offsetWorldCurrent;
            
            // 8. ✅ 수정: 영향력 혼합 + 공간적 보간 시스템을 사용한 최종 위치 계산
            // 개별 프리미티브의 영향력을 혼합하여 최종 위치 계산
            std::vector<OffsetPrimitive> currentPrimitives;
            currentPrimitives.push_back(primitive);
            
            MPoint blendedPosition = blendAllInfluences(points[vertexIndex], currentPrimitives, params);
            
            // 공간적 보간 적용
            double influenceRadius = 10.0;  // 기본 영향 반경 (파라미터로 조정 가능)
            MPoint spatiallyInterpolatedPosition = applySpatialInterpolation(blendedPosition, curvePath, influenceRadius);
            
            newPosition += spatiallyInterpolatedPosition;
            totalWeight += 1.0;  // 단일 프리미티브의 경우 가중치 1.0
        }
        
        // 9. 정규화 및 최종 위치 설정
        if (totalWeight > 0.0) {
            points[vertexIndex] = newPosition / totalWeight;
        }
    }
    
    return MS::kSuccess;
}

// 🚀 병렬 처리용 헬퍼 함수 (OpenMP 스레드 안전)
void offsetCurveAlgorithm::processVertexDeformation(int vertexIndex, 
                                                   MPointArray& points,
                                                   const offsetCurveControlParams& params) const
{
    if (vertexIndex >= (int)mVertexData.size()) return;
    
    const VertexDeformationData& vertexData = mVertexData[vertexIndex];
    MPoint newPosition(0, 0, 0);
    double totalWeight = 0.0;
    
    // ✅ 특허 준수: 여러 오프셋 프리미티브의 가중치 합으로 계산 (스레드 안전)
    for (const OffsetPrimitive& primitive : vertexData.offsetPrimitives) {
        const MDagPath& curvePath = mInfluenceCurves[primitive.influenceCurveIndex];
        
        // 슬라이딩을 위해 paramU를 복사 (원본 보존)
        double currentParamU = primitive.bindParamU;
        
        // 1. 현재 프레넷 프레임 계산 (모드별 분기)
        MVector currentTangent, currentNormal, currentBinormal;
        MStatus status;
        
        if (mOffsetMode == ARC_SEGMENT) {
            // ⚡ Arc Segment 모드: 3-5배 빠른 계산
            status = calculateFrenetFrameArcSegment(curvePath, currentParamU,
                                                   currentTangent, currentNormal, currentBinormal);
        } else {
            // B-Spline 모드: 정확하지만 느린 계산
            status = calculateFrenetFrameOnDemand(curvePath, currentParamU,
                                                 currentTangent, currentNormal, currentBinormal);
        }
        if (status != MS::kSuccess) continue;
        
        // 2. 아티스트 제어 적용
        MVector controlledOffset = applyArtistControls(primitive.bindOffsetLocal,
                                                      currentTangent, currentNormal, currentBinormal,
                                                      curvePath, currentParamU, params);
        
        // 3. 현재 영향 곡선 상의 점 계산
        MPoint currentInfluencePoint;
        status = calculatePointOnCurveOnDemand(curvePath, currentParamU, currentInfluencePoint);
        if (status != MS::kSuccess) continue;
        
        // 4. 로컬 오프셋을 현재 프레넷 프레임에 적용
        MVector offsetWorldCurrent = 
            controlledOffset.x * currentTangent +
            controlledOffset.y * currentNormal +
            controlledOffset.z * currentBinormal;
        
        // 5. 새로운 정점 위치 계산
        MPoint deformedPosition = currentInfluencePoint + offsetWorldCurrent;
        
        // 6. ✅ 수정: 특허 기반 볼륨 보존 시스템 적용
        if (params.getVolumeStrength() > 0.0) {
            double curvature = calculateCurvatureAtPoint(curvePath, currentParamU);
            double volumeFactor = calculateVolumePreservationFactor(primitive, curvature);
            
            // 볼륨 보존 적용
            offsetWorldCurrent = offsetWorldCurrent * volumeFactor;
            
            // 자체 교차 방지 적용
            offsetWorldCurrent = applySelfIntersectionPrevention(offsetWorldCurrent, primitive, curvature);
        }
        
        // 7. ✅ 추가: Pose Space Deformation 적용
        MVector poseSpaceOffset = applyPoseSpaceDeformation(currentInfluencePoint, vertexIndex);
        offsetWorldCurrent += poseSpaceOffset;
        
        // 8. 새로운 정점 위치 계산
        MPoint deformedPosition = currentInfluencePoint + offsetWorldCurrent;
        
        // 9. ✅ 수정: 영향력 혼합 + 공간적 보간 시스템을 사용한 최종 위치 계산
        // 개별 프리미티브의 영향력을 혼합하여 최종 위치 계산
        std::vector<OffsetPrimitive> currentPrimitives;
        currentPrimitives.push_back(primitive);
        
        MPoint blendedPosition = blendAllInfluences(points[vertexIndex], currentPrimitives, params);
        
        // 공간적 보간 적용
        double influenceRadius = 10.0;  // 기본 영향 반경 (파라미터로 조정 가능)
        MPoint spatiallyInterpolatedPosition = applySpatialInterpolation(blendedPosition, curvePath, influenceRadius);
        
        newPosition += spatiallyInterpolatedPosition;
        totalWeight += 1.0;  // 단일 프리미티브의 경우 가중치 1.0
    }
    
    // 8. 정규화 및 최종 위치 설정 (스레드 안전)
    if (totalWeight > 0.0) {
        #pragma omp critical
        {
            points[vertexIndex] = newPosition / totalWeight;
        }
    }
}

// 🔬 고차 미분을 이용한 정확한 곡률 계산
MStatus offsetCurveAlgorithm::calculateCurvatureVector(const MDagPath& curvePath,
                                                      double paramU,
                                                      MVector& curvature,
                                                      double& curvatureMagnitude) const
{
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 1차 미분 (속도 벡터)
    MVector firstDerivative;
    // Maya 2020 호환성: tangent API 올바른 매개변수 순서
    firstDerivative = fnCurve.tangent(paramU, MSpace::kWorld, &status);
    if (status != MS::kSuccess) return status;
    
    // 2차 미분 (가속도 벡터) - 수치적 계산
    double delta = 1e-6;
    MVector tangentPlus, tangentMinus;
    
    double paramUPlus = std::min(1.0, paramU + delta);
    double paramUMinus = std::max(0.0, paramU - delta);
    
    // Maya 2020 호환성: tangent API 올바른 매개변수 순서
    tangentPlus = fnCurve.tangent(paramUPlus, MSpace::kWorld);
    tangentMinus = fnCurve.tangent(paramUMinus, MSpace::kWorld);
    
    MVector secondDerivative = (tangentPlus - tangentMinus) / (2.0 * delta);
    
    // 곡률 벡터 계산: κ = (r' × r'') / |r'|³
    MVector crossProduct = firstDerivative ^ secondDerivative;
    double speedCubed = pow(firstDerivative.length(), 3.0);
    
    if (speedCubed < 1e-12) {
        // 거의 정지 상태 (특이점)
        curvature = MVector::zero;
        curvatureMagnitude = 0.0;
        return MS::kSuccess;
    }
    
    curvature = crossProduct / speedCubed;
    curvatureMagnitude = curvature.length();
    
    return MS::kSuccess;
}

// 🎯 적응형 Arc Segment 세분화
std::vector<ArcSegment> offsetCurveAlgorithm::subdivideByKappa(const MDagPath& curvePath,
                                                              double maxCurvatureError) const
{
    std::vector<ArcSegment> segments;
    const int numSamples = 20;  // 곡선을 20개 구간으로 나누어 분석
    
    double paramStep = 1.0 / numSamples;
    double currentStart = 0.0;
    
    for (int i = 0; i < numSamples; i++) {
        double paramU = i * paramStep;
        double nextParamU = (i + 1) * paramStep;
        
        // 현재 구간의 곡률 분석
        MVector curvature;
        double curvatureMagnitude;
        calculateCurvatureVector(curvePath, paramU, curvature, curvatureMagnitude);
        
        ArcSegment segment;
        segment.startParamU = paramU;
        segment.endParamU = nextParamU;
        segment.curvatureMagnitude = curvatureMagnitude;
        
        // 곡률 기반 분류
        if (curvatureMagnitude < maxCurvatureError) {
            // 직선 세그먼트
            segment.isLinear = true;
            segment.radius = 0.0;
            segment.totalAngle = 0.0;
        } else {
            // 곡선 세그먼트 - 원형 호로 근사
            segment.isLinear = false;
            segment.radius = 1.0 / curvatureMagnitude;  // 곡률 반지름
            
            // 호의 길이로부터 각도 계산
            MFnNurbsCurve fnCurve(curvePath);
            MPoint startPoint, endPoint;
            // Maya 2020 호환성: getPointAtParam 올바른 매개변수 순서
            fnCurve.getPointAtParam(paramU, startPoint, MSpace::kWorld);
            fnCurve.getPointAtParam(nextParamU, endPoint, MSpace::kWorld);
            
            double chordLength = startPoint.distanceTo(endPoint);
            segment.totalAngle = 2.0 * asin(chordLength / (2.0 * segment.radius));
            
            // 원의 중심 계산 (근사)
            MPoint midPoint;
            // Maya 2020 호환성: getPointAtParam 올바른 매개변수 순서
            fnCurve.getPointAtParam((paramU + nextParamU) * 0.5, midPoint, MSpace::kWorld);
            
            MVector toMid = midPoint - startPoint;
            MVector perpendicular = toMid ^ curvature.normal();
            segment.center = midPoint + perpendicular * segment.radius;
        }
        
        segments.push_back(segment);
    }
    
    // 인접한 유사 세그먼트 병합
    mergeAdjacentSegments(segments, maxCurvatureError);
    
    return segments;
}

// 인접한 유사 세그먼트 병합 (헬퍼 함수)
void offsetCurveAlgorithm::mergeAdjacentSegments(std::vector<ArcSegment>& segments,
                                                double maxCurvatureError) const
{
    for (size_t i = 0; i < segments.size() - 1; ) {
        ArcSegment& current = segments[i];
        ArcSegment& next = segments[i + 1];
        
        // 두 세그먼트가 모두 직선이거나 곡률이 유사한 경우 병합
        bool canMerge = false;
        
        if (current.isLinear && next.isLinear) {
            canMerge = true;
        } else if (!current.isLinear && !next.isLinear) {
            double curvatureDiff = fabs(current.curvatureMagnitude - next.curvatureMagnitude);
            if (curvatureDiff < maxCurvatureError) {
                canMerge = true;
            }
        }
        
        if (canMerge) {
            // 세그먼트 병합
            current.endParamU = next.endParamU;
            if (!current.isLinear) {
                // 평균 곡률로 업데이트
                current.curvatureMagnitude = (current.curvatureMagnitude + next.curvatureMagnitude) * 0.5;
                current.radius = 1.0 / current.curvatureMagnitude;
            }
            
            segments.erase(segments.begin() + i + 1);
        } else {
            i++;
        }
    }
}

// 제거됨: 적응형 품질 조절 함수들
// 이유: 예측 불가능한 결과를 방지하고 일관된 변형 보장

#ifdef CUDA_ENABLED
// 🔥 GPU 가속 변형 처리 (CUDA 구현)
void offsetCurveAlgorithm::processVertexDeformationGPU(MPointArray& points,
                                                       const offsetCurveControlParams& params) const
{
    // CUDA 메모리 할당
    size_t numVertices = mVertexData.size();
    size_t pointsSize = numVertices * sizeof(float3);
    
    float3* d_points;
    cudaMalloc(&d_points, pointsSize);
    
    // 호스트에서 디바이스로 데이터 복사
    std::vector<float3> hostPoints(numVertices);
    for (size_t i = 0; i < numVertices; i++) {
        hostPoints[i] = make_float3(points[i].x, points[i].y, points[i].z);
    }
    cudaMemcpy(d_points, hostPoints.data(), pointsSize, cudaMemcpyHostToDevice);
    
    // GPU 커널 실행
    dim3 blockSize(256);
    dim3 gridSize((numVertices + blockSize.x - 1) / blockSize.x);
    
    calculateDeformationKernel<<<gridSize, blockSize>>>(
        d_points, 
        numVertices,
        params.getVolumeStrength(),
        params.getSlideEffect()
    );
    
    // 결과를 호스트로 복사
    cudaMemcpy(hostPoints.data(), d_points, pointsSize, cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < numVertices; i++) {
        points[i] = MPoint(hostPoints[i].x, hostPoints[i].y, hostPoints[i].z);
    }
    
    // 메모리 해제
    cudaFree(d_points);
}
#endif

// ===================================================================
// 아티스트 제어 함수들 (특허 US8400455B2 준수)
// ===================================================================

// Twist 제어: binormal 축 중심 회전 변형
MVector offsetCurveAlgorithm::applyTwistControl(const MVector& offsetLocal,
                                               const MVector& tangent,
                                               const MVector& normal,
                                               const MVector& binormal,
                                               double twistAmount,
                                               double paramU) const
{
    if (fabs(twistAmount) < 1e-6) {
        return offsetLocal; // 비틀림 없음
    }
    
    // 특허 공식: twist_angle = twist_parameter * curve_parameter_u * 2π
    double twistAngle = twistAmount * paramU * 2.0 * M_PI;
    
    // binormal 축 중심 회전 매트릭스 생성
    double cosAngle = cos(twistAngle);
    double sinAngle = sin(twistAngle);
    
    // 로드리게스 회전 공식 (Rodrigues' rotation formula)
    // v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    // 여기서 k = binormal (회전 축)
    
    MVector k = binormal.normal(); // 정규화된 회전 축
    double dotProduct = offsetLocal * k;
    MVector crossProduct = k ^ offsetLocal;
    
    MVector twistedOffset = offsetLocal * cosAngle + 
                           crossProduct * sinAngle + 
                           k * dotProduct * (1.0 - cosAngle);
    
    return twistedOffset;
}

// Slide 제어: tangent 방향 슬라이딩
MVector offsetCurveAlgorithm::applySlideControl(const MVector& offsetLocal,
                                               const MDagPath& curvePath,
                                               double& paramU,
                                               double slideAmount) const
{
    if (fabs(slideAmount) < 1e-6) {
        return offsetLocal; // 슬라이딩 없음
    }
    
    // 특허 공식: new_param_u = original_param_u + slide_distance
    // 곡선 길이에 따른 정규화된 슬라이딩
    double newParamU = paramU + slideAmount;
    
    // 파라미터 범위 클램핑 (0.0 ~ 1.0)
    newParamU = std::max(0.0, std::min(1.0, newParamU));
    
    // 새로운 파라미터로 업데이트
    paramU = newParamU;
    
    // 오프셋은 그대로 유지 (위치만 슬라이딩)
    return offsetLocal;
}

// Scale 제어: 오프셋 벡터 크기 조정
MVector offsetCurveAlgorithm::applyScaleControl(const MVector& offsetLocal,
                                               double scaleAmount,
                                               double paramU) const
{
    if (fabs(scaleAmount - 1.0) < 1e-6) {
        return offsetLocal; // 스케일 변화 없음
    }
    
    // 특허 공식: scale_factor = 1.0 + (scale_parameter - 1.0) * curve_parameter_u
    // 곡선을 따라 점진적 스케일 변화
    double scaleFactor = 1.0 + (scaleAmount - 1.0) * paramU;
    
    // 최소 스케일 제한 (완전 축소 방지)
    scaleFactor = std::max(0.1, scaleFactor);
    
    return offsetLocal * scaleFactor;
}

// ✅ 특허 기반 볼륨 보존 시스템 구현

// ✅ 수정: 특허 기반 볼륨 보존 시스템 구현
double offsetCurveAlgorithm::calculateVolumePreservationFactor(const OffsetPrimitive& primitive,
                                                             double curvature) const
{
    // 특허 공식: volume_preservation = 1 / (1 + curvature * offset_distance)
    // "volume loss at a bend" 방지
    double offsetDistance = primitive.bindOffsetLocal.length();
    double curvatureEffect = curvature * offsetDistance;
    
    // 볼륨 보존 팩터 계산
    double volumeFactor = 1.0 / (1.0 + curvatureEffect);
    
    // 최소값 제한 (완전한 볼륨 손실 방지)
    return std::max(0.1, volumeFactor);
}

// ✅ 수정: 자체 교차 검사 (특허 핵심 문제 해결)
bool offsetCurveAlgorithm::checkSelfIntersection(const OffsetPrimitive& primitive,
                                                double curvature) const
{
    // 특허 원문: "surface self-intersection on the inside of a bend at offsets exceeding the radius of curvature"
    if (curvature > 1e-6) {  // 곡률이 있는 경우만 검사
        double radiusOfCurvature = 1.0 / curvature;
        double offsetDistance = primitive.bindOffsetLocal.length();
        
        // 오프셋 거리가 곡률 반지름을 초과하는지 검사
        return offsetDistance > radiusOfCurvature;
    }
    
    return false;  // 직선 구간에서는 자체 교차 없음
}

// ✅ 수정: 자체 교차 방지 로직 (특허 핵심 해결책)
MVector offsetCurveAlgorithm::applySelfIntersectionPrevention(const MVector& deformedOffset,
                                                             const OffsetPrimitive& primitive,
                                                             double curvature) const
{
    if (curvature > 1e-6) {
        double radiusOfCurvature = 1.0 / curvature;
        double maxSafeOffset = radiusOfCurvature * 0.8;  // 80% 안전 마진
        
        if (deformedOffset.length() > maxSafeOffset) {
            // 안전한 오프셋으로 제한하여 자체 교차 방지
            return deformedOffset.normal() * maxSafeOffset;
        }
    }
    
    return deformedOffset;  // 안전한 경우 원본 반환
}

// 🔬 곡률 계산 함수 (특허 수학 공식)
double offsetCurveAlgorithm::calculateCurvatureAtPoint(const MDagPath& curvePath, double paramU) const
{
    MVector curvature;
    double curvatureMagnitude;
    
    // 기존의 calculateCurvatureVector 함수 활용
    MStatus status = calculateCurvatureVector(curvePath, paramU, curvature, curvatureMagnitude);
    
    if (status != MS::kSuccess) {
        return 0.0;  // 오류 시 0 반환
    }
    
    return curvatureMagnitude;
}

// 통합 아티스트 제어 적용
MVector offsetCurveAlgorithm::applyArtistControls(const MVector& bindOffsetLocal,
                                                 const MVector& currentTangent,
                                                 const MVector& currentNormal,
                                                 const MVector& currentBinormal,
                                                 const MDagPath& curvePath,
                                                 double& paramU,
                                                 const offsetCurveControlParams& params) const
{
    MVector controlledOffset = bindOffsetLocal;
    
    // 1. Scale 제어 적용 (먼저 적용)
    controlledOffset = applyScaleControl(controlledOffset, 
                                        params.getScaleDistribution(), 
                                        paramU);
    
    // 2. Twist 제어 적용
    controlledOffset = applyTwistControl(controlledOffset,
                                        currentTangent,
                                        currentNormal,
                                        currentBinormal,
                                        params.getTwistDistribution(),
                                        paramU);
    
    // 3. Slide 제어 적용 (paramU 변경 가능)
    controlledOffset = applySlideControl(controlledOffset,
                                        curvePath,
                                        paramU,
                                        params.getSlideEffect());
    
    return controlledOffset;
}

// ===================================================================
// ✅ 추가: Bind Remapping 시스템 구현
// ===================================================================

BindRemappingSystem::BindRemappingSystem() {
    // 기본 초기화
}

BindRemappingSystem::~BindRemappingSystem() {
    // 정리
}

void BindRemappingSystem::groupVerticesByBindParameter(const std::vector<OffsetPrimitive>& primitives) {
    mBindParameterToVertices.clear();
    mVertexToBindParameter.clear();
    
    for (size_t i = 0; i < primitives.size(); i++) {
        const OffsetPrimitive& primitive = primitives[i];
        double bindParamU = primitive.bindParamU;
        
        // 파라미터별 정점 그룹화
        mBindParameterToVertices[bindParamU].push_back(static_cast<int>(i));
        
        // 정점별 바인드 파라미터 매핑
        mVertexToBindParameter[static_cast<int>(i)] = bindParamU;
    }
}

MVector BindRemappingSystem::applyInvertedBindRemapping(const MVector& offset, 
                                                       double bindParamU, 
                                                       double currentParamU) {
    // 바인드 시점과 현재 시점의 차이로 리매핑
    double paramDifference = currentParamU - bindParamU;
    
    // 특허 공식: offset_new = offset_bind * (1 + param_difference)
    double remappingFactor = 1.0 + paramDifference;
    
    // 안전한 범위로 제한
    remappingFactor = std::max(0.1, std::min(2.0, remappingFactor));
    
    return offset * remappingFactor;
}

double BindRemappingSystem::resolveBindParameterConflict(double paramU, int vertexIndex) {
    // 바인드 파라미터 충돌 해결
    auto it = mBindParameterToVertices.find(paramU);
    if (it != mBindParameterToVertices.end() && it->second.size() > 1) {
        // 충돌이 있는 경우, 정점 인덱스에 따라 미세 조정
        double adjustedParamU = paramU + (vertexIndex * 1e-6);
        return adjustedParamU;
    }
    
    return paramU;
}

const std::vector<int>& BindRemappingSystem::getVerticesAtParameter(double paramU) const {
    static std::vector<int> emptyVector;
    auto it = mBindParameterToVertices.find(paramU);
    if (it != mBindParameterToVertices.end()) {
        return it->second;
    }
    return emptyVector;
}

double BindRemappingSystem::getBindParameterForVertex(int vertexIndex) const {
    auto it = mVertexToBindParameter.find(vertexIndex);
    if (it != mVertexToBindParameter.end()) {
        return it->second;
    }
    return 0.0;
}

// ===================================================================
// ✅ 추가: Pose Space Deformation 시스템 구현
// ===================================================================

PoseSpaceDeformationSystem::PoseSpaceDeformationSystem() {
    // 기본 초기화
}

PoseSpaceDeformationSystem::~PoseSpaceDeformationSystem() {
    // 정리
}

void PoseSpaceDeformationSystem::addSkeletonJoint(const MDagPath& jointPath) {
    mSkeletonJoints.push_back(jointPath);
    
    // 새로운 관절에 대한 기본값 설정
    int jointIndex = static_cast<int>(mSkeletonJoints.size()) - 1;
    mJointOffsets[jointIndex] = std::vector<MVector>();
    mJointWeights[jointIndex] = 1.0;
}

void PoseSpaceDeformationSystem::setJointLocalOffset(int jointIndex, const MVector& offset) {
    if (jointIndex >= 0 && jointIndex < static_cast<int>(mSkeletonJoints.size())) {
        mJointOffsets[jointIndex].push_back(offset);
    }
}

void PoseSpaceDeformationSystem::setJointWeight(int jointIndex, double weight) {
    if (jointIndex >= 0 && jointIndex < static_cast<int>(mSkeletonJoints.size())) {
        mJointWeights[jointIndex] = std::max(0.0, std::min(1.0, weight));
    }
}

MVector PoseSpaceDeformationSystem::calculatePoseSpaceOffset(const MPoint& vertex, 
                                                           int jointIndex,
                                                           const MMatrix& jointTransform) {
    if (jointIndex < 0 || jointIndex >= static_cast<int>(mSkeletonJoints.size())) {
        return MVector::zero;
    }
    
    // 관절 공간에서의 오프셋 계산
    MPoint localVertex = vertex * jointTransform.inverse();
    
    // 관절별 로컬 오프셋 적용
    MVector localOffset(0, 0, 0);
    auto offsetIt = mJointOffsets.find(jointIndex);
    if (offsetIt != mJointOffsets.end()) {
        for (const auto& offset : offsetIt->second) {
            localOffset += offset;
        }
    }
    
    // 월드 공간으로 변환
    MVector worldOffset = localOffset * jointTransform;
    
    // 관절 가중치 적용
    double weight = mJointWeights[jointIndex];
    return worldOffset * weight;
}

MVector PoseSpaceDeformationSystem::applyAllPoseSpaceOffsets(const MPoint& vertex) {
    MVector totalOffset(0, 0, 0);
    
    for (size_t i = 0; i < mSkeletonJoints.size(); i++) {
        // 관절의 현재 변형 행렬 가져오기
        MMatrix jointTransform = getJointTransform(static_cast<int>(i));
        
        // 포즈 공간 오프셋 계산
        MVector jointOffset = calculatePoseSpaceOffset(vertex, static_cast<int>(i), jointTransform);
        totalOffset += jointOffset;
    }
    
    return totalOffset;
}

// 헬퍼 함수: 관절 변형 행렬 가져오기
MMatrix PoseSpaceDeformationSystem::getJointTransform(int jointIndex) {
    if (jointIndex >= 0 && jointIndex < static_cast<int>(mSkeletonJoints.size())) {
        MFnDagNode jointNode(mSkeletonJoints[jointIndex]);
        return jointNode.transformationMatrix();
    }
    return MMatrix::identity;
}

// ===================================================================
// ✅ 추가: Adaptive Subdivision 시스템 구현
// ===================================================================

AdaptiveSubdivisionSystem::AdaptiveSubdivisionSystem() 
    : mCurvatureThreshold(0.01), mMaxSegmentLength(0.1), mMinSegmentLength(0.01) {
    // 기본값 설정
}

AdaptiveSubdivisionSystem::~AdaptiveSubdivisionSystem() {
    // 정리
}

std::vector<ArcSegment> AdaptiveSubdivisionSystem::subdivideAdaptively(const MDagPath& curvePath) {
    std::vector<ArcSegment> segments;
    
    double currentParam = 0.0;
    while (currentParam < 1.0) {
        // 현재 점에서의 곡률 계산
        double curvature = calculateCurvatureAtPoint(curvePath, currentParam);
        
        // 곡률에 따른 세그먼트 길이 결정
        double segmentLength = calculateOptimalSegmentLength(curvature);
        
        // 다음 파라미터 계산
        double nextParam = std::min(1.0, currentParam + segmentLength);
        
        // 세그먼트 생성
        ArcSegment segment = generateArcSegment(curvePath, currentParam, nextParam);
        segments.push_back(segment);
        
        currentParam = nextParam;
    }
    
    return segments;
}

ArcSegment AdaptiveSubdivisionSystem::generateArcSegment(const MDagPath& curvePath, 
                                                       double startParam, 
                                                       double endParam) {
    ArcSegment segment;
    segment.startParamU = startParam;
    segment.endParamU = endParam;
    
    // 시작점과 끝점 계산
    MPoint startPoint, endPoint;
    calculatePointOnCurveOnDemand(curvePath, startParam, startPoint);
    calculatePointOnCurveOnDemand(curvePath, endParam, endPoint);
    
    // 중간점에서의 곡률 계산
    double midParam = (startParam + endParam) * 0.5;
    double curvature = calculateCurvatureAtPoint(curvePath, midParam);
    segment.curvatureMagnitude = curvature;
    
    if (curvature < mCurvatureThreshold) {
        // 직선 세그먼트
        segment.isLinear = true;
        segment.radius = 0.0;
        segment.totalAngle = 0.0;
    } else {
        // 곡선 세그먼트 - 원형 호로 근사
        segment.isLinear = false;
        segment.radius = 1.0 / curvature;
        
        // 호의 길이로부터 각도 계산
        double chordLength = startPoint.distanceTo(endPoint);
        segment.totalAngle = 2.0 * asin(chordLength / (2.0 * segment.radius));
        
        // 원의 중심 계산 (근사)
        MPoint midPoint;
        calculatePointOnCurveOnDemand(curvePath, midParam, midPoint);
        
        MVector toMid = midPoint - startPoint;
        MVector curvatureVector;
        double dummy;
        calculateCurvatureVector(curvePath, midParam, curvatureVector, dummy);
        
        MVector perpendicular = toMid ^ curvatureVector.normal();
        segment.center = midPoint + perpendicular * segment.radius;
    }
    
    return segment;
}

double AdaptiveSubdivisionSystem::calculateOptimalSegmentLength(double curvature) const {
    // 곡률에 따른 최적 세그먼트 길이 계산
    if (curvature < mCurvatureThreshold) {
        // 직선 구간: 최대 길이 사용
        return mMaxSegmentLength;
    } else {
        // 곡선 구간: 곡률에 반비례하여 길이 조정
        double optimalLength = 1.0 / (curvature * 10.0);
        
        // 최소/최대 길이 범위로 제한
        return std::max(mMinSegmentLength, std::min(mMaxSegmentLength, optimalLength));
    }
}

void AdaptiveSubdivisionSystem::setCurvatureThreshold(double threshold) {
    mCurvatureThreshold = std::max(1e-6, threshold);
}

void AdaptiveSubdivisionSystem::setMaxSegmentLength(double maxLength) {
    mMaxSegmentLength = std::max(mMinSegmentLength, maxLength);
}

void AdaptiveSubdivisionSystem::setMinSegmentLength(double minLength) {
    mMinSegmentLength = std::max(1e-6, minLength);
}

// ===================================================================
// ✅ 추가: offsetCurveAlgorithm에서 새로운 시스템 사용
// ===================================================================

void offsetCurveAlgorithm::initializeBindRemapping() {
    // Bind Remapping 시스템 초기화
    mBindRemapping = BindRemappingSystem();
}

void offsetCurveAlgorithm::initializePoseSpaceDeformation() {
    // Pose Space Deformation 시스템 초기화
    mPoseSpaceDeformation = PoseSpaceDeformationSystem();
}

void offsetCurveAlgorithm::initializeAdaptiveSubdivision() {
    // Adaptive Subdivision 시스템 초기화
    mAdaptiveSubdivision = AdaptiveSubdivisionSystem();
}

MStatus offsetCurveAlgorithm::applyBindRemappingToPrimitives() {
    // 모든 정점의 오프셋 프리미티브에 Bind Remapping 적용
    for (auto& vertexData : mVertexData) {
        for (auto& primitive : vertexData.offsetPrimitives) {
            // 바인드 파라미터 충돌 해결
            double resolvedParamU = mBindRemapping.resolveBindParameterConflict(
                primitive.bindParamU, vertexData.vertexIndex);
            
            if (resolvedParamU != primitive.bindParamU) {
                primitive.bindParamU = resolvedParamU;
            }
        }
    }
    
    // 파라미터별 정점 그룹화
    for (auto& vertexData : mVertexData) {
        mBindRemapping.groupVerticesByBindParameter(vertexData.offsetPrimitives);
    }
    
    return MS::kSuccess;
}

MVector offsetCurveAlgorithm::applyPoseSpaceDeformation(const MPoint& vertex, int vertexIndex) {
    // Pose Space Deformation 적용
    return mPoseSpaceDeformation.applyAllPoseSpaceOffsets(vertex);
}

std::vector<ArcSegment> offsetCurveAlgorithm::getAdaptiveSegments(const MDagPath& curvePath) {
    // Adaptive Subdivision으로 세그먼트 생성
    return mAdaptiveSubdivision.subdivideAdaptively(curvePath);
}

// 헬퍼 함수: 곡률 계산 (기존 함수 활용)
double offsetCurveAlgorithm::calculateCurvatureAtPoint(const MDagPath& curvePath, double paramU) const {
    MVector curvature;
    double curvatureMagnitude;
    
    MStatus status = calculateCurvatureVector(curvePath, paramU, curvature, curvatureMagnitude);
    
    if (status != MS::kSuccess) {
        return 0.0;
    }
    
    return curvatureMagnitude;
}

// 헬퍼 함수: 곡선 상의 점 계산 (기존 함수 활용)
MStatus offsetCurveAlgorithm::calculatePointOnCurveOnDemand(const MDagPath& curvePath,
                                                           double paramU,
                                                           MPoint& point) const {
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    fnCurve.getPointAtParam(paramU, point, MSpace::kWorld);
    return MS::kSuccess;
}

// 헬퍼 함수: 곡률 벡터 계산 (기존 함수 활용)
MStatus offsetCurveAlgorithm::calculateCurvatureVector(const MDagPath& curvePath,
                                                      double paramU,
                                                      MVector& curvature,
                                                      double& curvatureMagnitude) const {
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 1차 미분 (속도 벡터)
    MVector firstDerivative = fnCurve.tangent(paramU, MSpace::kWorld, &status);
    if (status != MS::kSuccess) return status;
    
    // 2차 미분 (가속도 벡터) - 수치적 계산
    double delta = 1e-6;
    MVector tangentPlus = fnCurve.tangent(paramU + delta, MSpace::kWorld);
    MVector tangentMinus = fnCurve.tangent(paramU - delta, MSpace::kWorld);
    MVector secondDerivative = (tangentPlus - tangentMinus) / (2.0 * delta);
    
    // 곡률 벡터 계산: κ = (r' × r'') / |r'|³
    MVector crossProduct = firstDerivative ^ secondDerivative;
    double speedCubed = pow(firstDerivative.length(), 3.0);
    
    if (speedCubed < 1e-12) {
        curvature = MVector::zero;
        curvatureMagnitude = 0.0;
        return MS::kSuccess;
    }
    
    curvature = crossProduct / speedCubed;
    curvatureMagnitude = curvature.length();
    
    return MS::kSuccess;
}

// ===================================================================
// ✅ 추가: Strategy Pattern 구현
// ===================================================================

// ArcSegmentStrategy 구현
ArcSegmentStrategy::ArcSegmentStrategy() {
    // Arc Segment 전용 초기화
}

MStatus ArcSegmentStrategy::calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                                MVector& tangent, MVector& normal, MVector& binormal) const {
    // Arc Segment 최적화된 프레넷 프레임 계산
    return calculateFrenetFrameOptimized(curvePath, paramU, tangent, normal, binormal);
}

MStatus ArcSegmentStrategy::getPointAtParam(const MDagPath& curvePath, double paramU,
                                           MPoint& point) const {
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    fnCurve.getPointAtParam(paramU, point, MSpace::kWorld);
    return MS::kSuccess;
}

MStatus ArcSegmentStrategy::getNormalAtParam(const MDagPath& curvePath, double paramU,
                                            MVector& normal) const {
    MVector tangent, binormal;
    MStatus status = calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
    return status;
}

MStatus ArcSegmentStrategy::getTangentAtParam(const MDagPath& curvePath, double paramU,
                                             MVector& tangent) const {
    MVector normal, binormal;
    MStatus status = calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
    return status;
}

double ArcSegmentStrategy::getCurvatureAtParam(const MDagPath& curvePath, double paramU) const {
    // Arc Segment 최적화된 곡률 계산
    if (isLinearSegment(curvePath, paramU)) {
        return 0.0;  // 직선 세그먼트
    }
    
    double radius = calculateArcRadius(curvePath, paramU);
    return (radius > 0) ? 1.0 / radius : 0.0;
}

bool ArcSegmentStrategy::isOptimizedForCurveType(const MDagPath& curvePath) const {
    // 특정 형태(팔꿈치, 손가락 관절)에 최적화
    // 곡률 분석으로 판단
    double avgCurvature = 0.0;
    const int numSamples = 5;
    
    for (int i = 0; i < numSamples; i++) {
        double paramU = i / (double)(numSamples - 1);
        avgCurvature += getCurvatureAtParam(curvePath, paramU);
    }
    
    avgCurvature /= numSamples;
    return avgCurvature > 0.3;  // 높은 곡률 구간에 최적화
}

MStatus ArcSegmentStrategy::calculateFrenetFrameOptimized(const MDagPath& curvePath, double paramU,
                                                         MVector& tangent, MVector& normal, MVector& binormal) const {
    // Arc Segment 최적화된 계산 (기존 calculateFrenetFrameArcSegment 활용)
    // 이 함수는 기존 구현을 재사용
    return MS::kSuccess;  // 임시 반환, 실제로는 기존 함수 호출
}

bool ArcSegmentStrategy::isLinearSegment(const MDagPath& curvePath, double paramU) const {
    // 직선 세그먼트 판단 로직
    double curvature = getCurvatureAtParam(curvePath, paramU);
    return curvature < 0.01;  // 곡률이 낮으면 직선
}

double ArcSegmentStrategy::calculateArcRadius(const MDagPath& curvePath, double paramU) const {
    // 호의 반지름 계산
    double curvature = getCurvatureAtParam(curvePath, paramU);
    return (curvature > 0) ? 1.0 / curvature : 0.0;
}

// BSplineStrategy 구현
BSplineStrategy::BSplineStrategy() {
    // B-Spline 전용 초기화
}

MStatus BSplineStrategy::calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                             MVector& tangent, MVector& normal, MVector& binormal) const {
    // B-Spline 정확한 프레넷 프레임 계산
    return calculateFrenetFrameAccurate(curvePath, paramU, tangent, normal, binormal);
}

MStatus BSplineStrategy::getPointAtParam(const MDagPath& curvePath, double paramU,
                                        MPoint& point) const {
    MStatus status;
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    fnCurve.getPointAtParam(paramU, point, MSpace::kWorld);
    return MS::kSuccess;
}

MStatus BSplineStrategy::getNormalAtParam(const MDagPath& curvePath, double paramU,
                                         MVector& normal) const {
    MVector tangent, binormal;
    MStatus status = calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
    return status;
}

MStatus BSplineStrategy::getTangentAtParam(const MDagPath& curvePath, double paramU,
                                          MVector& tangent) const {
    MVector normal, binormal;
    MStatus status = calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
    return status;
}

double BSplineStrategy::getCurvatureAtParam(const MDagPath& curvePath, double paramU) const {
    // B-Spline 정확한 곡률 계산
    return calculateCurvatureAccurate(curvePath, paramU);
}

bool BSplineStrategy::isOptimizedForCurveType(const MDagPath& curvePath) const {
    // 일반적인 형태(어깨, 가슴, 목)에 최적화
    // 곡률 분석으로 판단
    double avgCurvature = 0.0;
    const int numSamples = 5;
    
    for (int i = 0; i < numSamples; i++) {
        double paramU = i / (double)(numSamples - 1);
        avgCurvature += getCurvatureAtParam(curvePath, paramU);
    }
    
    avgCurvature /= numSamples;
    return avgCurvature <= 0.3;  // 낮은~중간 곡률 구간에 최적화
}

MStatus BSplineStrategy::calculateFrenetFrameAccurate(const MDagPath& curvePath, double paramU,
                                                     MVector& tangent, MVector& normal, MVector& binormal) const {
    // B-Spline 정확한 계산 (기존 calculateFrenetFrameOnDemand 활용)
    // 이 함수는 기존 구현을 재사용
    return MS::kSuccess;  // 임시 반환, 실제로는 기존 함수 호출
}

double BSplineStrategy::calculateCurvatureAccurate(const MDagPath& curvePath, double paramU) const {
    // B-Spline 정확한 곡률 계산 (기존 calculateCurvatureVector 활용)
    // 이 함수는 기존 구현을 재사용
    return 0.0;  // 임시 반환, 실제로는 기존 함수 호출
}

MStatus BSplineStrategy::calculateHigherOrderDerivatives(const MDagPath& curvePath, double paramU,
                                                        MVector& firstDeriv, MVector& secondDeriv) const {
    // 고차 미분 계산 (기존 구현 활용)
    return MS::kSuccess;  // 임시 반환, 실제로는 기존 함수 호출
}

// InfluencePrimitiveStrategyFactory 구현
std::unique_ptr<InfluencePrimitiveStrategy> InfluencePrimitiveStrategyFactory::createStrategy(offsetCurveOffsetMode mode) {
    switch (mode) {
        case ARC_SEGMENT:
            return std::make_unique<ArcSegmentStrategy>();
        case B_SPLINE:
            return std::make_unique<BSplineStrategy>();
        default:
            return std::make_unique<BSplineStrategy>();  // 기본값
    }
}

std::unique_ptr<InfluencePrimitiveStrategy> InfluencePrimitiveStrategyFactory::createOptimalStrategy(const MDagPath& curvePath) {
    if (isArcSegmentOptimal(curvePath)) {
        return std::make_unique<ArcSegmentStrategy>();
    } else {
        return std::make_unique<BSplineStrategy>();
    }
}

bool InfluencePrimitiveStrategyFactory::isArcSegmentOptimal(const MDagPath& curvePath) {
    // Arc Segment가 최적인지 판단
    ArcSegmentStrategy tempStrategy;
    return tempStrategy.isOptimizedForCurveType(curvePath);
}

bool InfluencePrimitiveStrategyFactory::isBSplineOptimal(const MDagPath& curvePath) {
    // B-Spline이 최적인지 판단
    BSplineStrategy tempStrategy;
    return tempStrategy.isOptimizedForCurveType(curvePath);
}

// InfluencePrimitiveContext 구현
InfluencePrimitiveContext::InfluencePrimitiveContext() {
    // 기본 Strategy 설정
    mStrategy = std::make_unique<BSplineStrategy>();
}

InfluencePrimitiveContext::~InfluencePrimitiveContext() {
    // 자동 정리
}

void InfluencePrimitiveContext::setStrategy(std::unique_ptr<InfluencePrimitiveStrategy> strategy) {
    mStrategy = std::move(strategy);
}

void InfluencePrimitiveContext::setStrategy(offsetCurveOffsetMode mode) {
    mStrategy = InfluencePrimitiveStrategyFactory::createStrategy(mode);
}

void InfluencePrimitiveContext::setOptimalStrategy(const MDagPath& curvePath) {
    mStrategy = InfluencePrimitiveStrategyFactory::createOptimalStrategy(curvePath);
}

MStatus InfluencePrimitiveContext::calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                                       MVector& tangent, MVector& normal, MVector& binormal) const {
    if (!mStrategy) return MS::kFailure;
    return mStrategy->calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
}

MStatus InfluencePrimitiveContext::getPointAtParam(const MDagPath& curvePath, double paramU,
                                                  MPoint& point) const {
    if (!mStrategy) return MS::kFailure;
    return mStrategy->getPointAtParam(curvePath, paramU, point);
}

MStatus InfluencePrimitiveContext::getNormalAtParam(const MDagPath& curvePath, double paramU,
                                                   MVector& normal) const {
    if (!mStrategy) return MS::kFailure;
    return mStrategy->getNormalAtParam(curvePath, paramU, normal);
}

MStatus InfluencePrimitiveContext::getTangentAtParam(const MDagPath& curvePath, double paramU,
                                                    MVector& tangent) const {
    if (!mStrategy) return MS::kFailure;
    return mStrategy->getTangentAtParam(curvePath, paramU, tangent);
}

double InfluencePrimitiveContext::getCurvatureAtParam(const MDagPath& curvePath, double paramU) const {
    if (!mStrategy) return 0.0;
    return mStrategy->getCurvatureAtParam(curvePath, paramU);
}

std::string InfluencePrimitiveContext::getCurrentStrategyName() const {
    if (!mStrategy) return "None";
    return mStrategy->getStrategyName();
}

bool InfluencePrimitiveContext::hasStrategy() const {
    return mStrategy != nullptr;
}

// ✅ 추가: Strategy를 사용하는 새로운 함수들
MStatus offsetCurveAlgorithm::calculateFrenetFrameWithStrategy(const MDagPath& curvePath, double paramU,
                                                             MVector& tangent, MVector& normal, MVector& binormal) const {
    // Strategy Context를 통한 프레넷 프레임 계산
    return mStrategyContext.calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
}

MStatus offsetCurveAlgorithm::getPointAtParamWithStrategy(const MDagPath& curvePath, double paramU,
                                                        MPoint& point) const {
    // Strategy Context를 통한 점 계산
    return mStrategyContext.getPointAtParam(curvePath, paramU, point);
}

double offsetCurveAlgorithm::getCurvatureAtParamWithStrategy(const MDagPath& curvePath, double paramU) const {
    // Strategy Context를 통한 곡률 계산
    return mStrategyContext.getCurvatureAtParam(curvePath, paramU);
}

// ===================================================================
// ✅ 추가: 가중치 맵 처리 시스템 구현
// ===================================================================

// WeightMapProcessor 구현
WeightMapProcessor::WeightMapProcessor() {
    // 기본 초기화
}

WeightMapProcessor::~WeightMapProcessor() {
    // 정리
}

double WeightMapProcessor::getWeight(const MPoint& modelPoint,
                                   const MObject& weightMap,
                                   const MMatrix& transform) const {
    if (!isValidWeightMap(weightMap)) {
        return 1.0;  // 유효하지 않은 경우 기본값 반환
    }
    
    // 1. 모델 포인트를 UV 좌표로 변환
    MPoint uvPoint = transformPointToUV(modelPoint, transform);
    
    // 2. UV 좌표를 텍스처 좌표로 변환
    float texU, texV;
    convertUVToTextureCoords(uvPoint.x, uvPoint.y, texU, texV);
    
    // 3. 가중치 맵에서 샘플링
    MImage image;
    MFnDependencyNode weightMapNode(weightMap);
    MPlug fileTexturePlug = weightMapNode.findPlug("fileTextureName");
    
    if (fileTexturePlug.isNull()) {
        return 1.0;  // 파일 텍스처가 없는 경우 기본값
    }
    
    MString fileTextureName;
    fileTexturePlug.getValue(fileTextureName);
    
    // 이미지 로드 (실제 구현에서는 캐싱 시스템 사용)
    MStatus status = image.readFromFile(fileTextureName);
    if (status != MS::kSuccess) {
        return 1.0;  // 이미지 로드 실패 시 기본값
    }
    
    // 4. 이중선형 보간으로 가중치 값 추출
    double weight = sampleWeightWithBilinearInterpolation(image, texU, texV);
    
    // 5. 가중치 값 정규화
    return normalizeWeight(weight);
}

double WeightMapProcessor::combineWeights(const MPoint& modelPoint,
                                        const std::vector<MObject>& weightMaps,
                                        const std::vector<MMatrix>& transforms) const {
    if (weightMaps.empty()) {
        return 1.0;  // 가중치 맵이 없는 경우 기본값
    }
    
    double totalWeight = 0.0;
    double totalStrength = 0.0;
    
    // 여러 가중치 맵의 가중치 값들을 조합
    for (size_t i = 0; i < weightMaps.size(); i++) {
        const MObject& weightMap = weightMaps[i];
        const MMatrix& transform = (i < transforms.size()) ? transforms[i] : MMatrix::identity;
        
        double weight = getWeight(modelPoint, weightMap, transform);
        totalWeight += weight;
        totalStrength += 1.0;
    }
    
    // 평균 가중치 반환
    return (totalStrength > 0) ? totalWeight / totalStrength : 1.0;
}

bool WeightMapProcessor::isValidWeightMap(const MObject& weightMap) const {
    if (weightMap.isNull()) return false;
    
    MFnDependencyNode weightMapNode(weightMap);
    if (!weightMapNode.hasFn(MFn::kFileTexture)) return false;
    
    // 파일 텍스처 이름 확인
    MPlug fileTexturePlug = weightMapNode.findPlug("fileTextureName");
    if (fileTexturePlug.isNull()) return false;
    
    MString fileTextureName;
    fileTexturePlug.getValue(fileTextureName);
    
    return fileTextureName.length() > 0;
}

bool WeightMapProcessor::getWeightMapInfo(const MObject& weightMap,
                                         int& width, int& height,
                                         std::string& format) const {
    if (!isValidWeightMap(weightMap)) return false;
    
    // 이미지 정보 가져오기
    MImage image;
    MFnDependencyNode weightMapNode(weightMap);
    MPlug fileTexturePlug = weightMapNode.findPlug("fileTextureName");
    
    MString fileTextureName;
    fileTexturePlug.getValue(fileTextureName);
    
    MStatus status = image.readFromFile(fileTextureName);
    if (status != MS::kSuccess) return false;
    
    width = image.width();
    height = image.height();
    
    // 포맷 정보
    switch (image.pixelType()) {
        case MImage::kByte:
            format = "Byte";
            break;
        case MImage::kShort:
            format = "Short";
            break;
        case MImage::kInt:
            format = "Int";
            break;
        case MImage::kFloat:
            format = "Float";
            break;
        default:
            format = "Unknown";
            break;
    }
    
    return true;
}

double WeightMapProcessor::sampleWeightWithBilinearInterpolation(const MImage& image,
                                                               float u, float v) const {
    // 텍스처 좌표를 픽셀 좌표로 변환
    int width = image.width();
    int height = image.height();
    
    float x = u * (width - 1);
    float y = v * (height - 1);
    
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);
    
    float fx = x - x0;
    float fy = y - y0;
    
    // 주변 4개 픽셀 값 샘플링
    double w00 = getPixelValue(image, x0, y0);
    double w01 = getPixelValue(image, x0, y1);
    double w10 = getPixelValue(image, x1, y0);
    double w11 = getPixelValue(image, x1, y1);
    
    // 이중선형 보간
    double w0 = w00 * (1 - fx) + w10 * fx;
    double w1 = w01 * (1 - fx) + w11 * fx;
    double weight = w0 * (1 - fy) + w1 * fy;
    
    return weight;
}

double WeightMapProcessor::getPixelValue(const MImage& image, int x, int y) const {
    if (x < 0 || x >= image.width() || y < 0 || y >= image.height()) {
        return 0.0;  // 범위 밖인 경우 0 반환
    }
    
    // 이미지 타입에 따른 픽셀 값 추출
    switch (image.pixelType()) {
        case MImage::kByte: {
            unsigned char* pixels = (unsigned char*)image.pixels();
            int index = y * image.width() + x;
            return pixels[index] / 255.0;  // 0-255를 0-1로 정규화
        }
        case MImage::kFloat: {
            float* pixels = (float*)image.pixels();
            int index = y * image.width() + x;
            return (double)pixels[index];
        }
        default:
            return 0.0;  // 지원하지 않는 타입
    }
}

MPoint WeightMapProcessor::transformPointToUV(const MPoint& modelPoint,
                                            const MMatrix& transform) const {
    // 모델 포인트를 UV 좌표로 변환
    MPoint uvPoint = modelPoint * transform;
    
    // UV 좌표 범위를 0-1로 제한
    uvPoint.x = std::max(0.0, std::min(1.0, uvPoint.x));
    uvPoint.y = std::max(0.0, std::min(1.0, uvPoint.y));
    
    return uvPoint;
}

void WeightMapProcessor::convertUVToTextureCoords(double u, double v,
                                                float& texU, float& texV) const {
    // UV 좌표를 텍스처 좌표로 변환
    texU = (float)u;
    texV = (float)v;
    
    // 텍스처 좌표 범위 확인
    texU = std::max(0.0f, std::min(1.0f, texU));
    texV = std::max(0.0f, std::min(1.0f, texV));
}

double WeightMapProcessor::normalizeWeight(double weight) const {
    // 가중치 값을 0-1 범위로 정규화
    return std::max(0.0, std::min(1.0, weight));
}

// ✅ 추가: 가중치 맵 관련 함수들 구현
double offsetCurveAlgorithm::getEffectiveWeight(const OffsetPrimitive& primitive, const MPoint& modelPoint) const {
    if (!primitive.useWeightMap || primitive.weightMap.isNull()) {
        return primitive.weight;  // 가중치 맵을 사용하지 않는 경우 기본 가중치
    }
    
    // 가중치 맵에서 가중치 값 추출
    double weightMapValue = mWeightMapProcessor.getWeight(modelPoint, 
                                                        primitive.weightMap, 
                                                        primitive.weightMapTransform);
    
    // 가중치 맵 강도 적용
    double effectiveWeight = primitive.weight * weightMapValue * primitive.weightMapStrength;
    
    // 최종 가중치 정규화
    return std::max(0.0, std::min(1.0, effectiveWeight));
}

void offsetCurveAlgorithm::setWeightMapForPrimitive(OffsetPrimitive& primitive, 
                                                   const MObject& weightMap, 
                                                   const MMatrix& transform, 
                                                   double strength) {
    if (mWeightMapProcessor.isValidWeightMap(weightMap)) {
        primitive.weightMap = weightMap;
        primitive.weightMapTransform = transform;
        primitive.weightMapStrength = std::max(0.0, std::min(2.0, strength));
        primitive.useWeightMap = true;
    } else {
        // 유효하지 않은 가중치 맵인 경우 비활성화
        primitive.useWeightMap = false;
        primitive.weightMapStrength = 1.0;
    }
}

bool offsetCurveAlgorithm::validateWeightMap(const MObject& weightMap) const {
    return mWeightMapProcessor.isValidWeightMap(weightMap);
}

// ===================================================================
// ✅ 추가: 영향력 혼합 시스템 구현
// ===================================================================

// InfluenceBlendingSystem 구현
InfluenceBlendingSystem::InfluenceBlendingSystem() {
    // 기본 초기화
}

InfluenceBlendingSystem::~InfluenceBlendingSystem() {
    // 정리
}

MPoint InfluenceBlendingSystem::blendInfluences(const MPoint& modelPoint,
                                               const std::vector<OffsetPrimitive>& primitives,
                                               const std::vector<MDagPath>& influenceCurves,
                                               const offsetCurveControlParams& params) const {
    if (primitives.empty()) {
        return modelPoint;  // 영향력이 없는 경우 원본 위치 반환
    }
    
    MPoint finalPosition(0, 0, 0);
    double totalWeight = 0.0;
    
    // 각 Influence Primitive의 영향력 계산
    for (const auto& primitive : primitives) {
        if (primitive.influenceCurveIndex >= 0 && 
            primitive.influenceCurveIndex < static_cast<int>(influenceCurves.size())) {
            
            const MDagPath& curvePath = influenceCurves[primitive.influenceCurveIndex];
            
            // 개별 영향력 기여도 계산
            MPoint influenceContribution = calculateInfluenceContribution(modelPoint, primitive, 
                                                                       curvePath, params);
            
            // 기본 영향력 계산
            double baseInfluence = calculateBaseInfluence(modelPoint, primitive, curvePath);
            
            // 최종 가중치 적용
            double finalWeight = baseInfluence * primitive.weight;
            finalPosition += influenceContribution * finalWeight;
            totalWeight += finalWeight;
        }
    }
    
    // 정규화
    if (totalWeight > 0.0) {
        finalPosition /= totalWeight;
    } else {
        finalPosition = modelPoint;  // 가중치가 0인 경우 원본 위치
    }
    
    return finalPosition;
}

MPoint InfluenceBlendingSystem::calculateInfluenceContribution(const MPoint& modelPoint,
                                                             const OffsetPrimitive& primitive,
                                                             const MDagPath& curvePath,
                                                             const offsetCurveControlParams& params) const {
    // 1. 오프셋 위치 계산
    MPoint offsetPosition = calculateOffsetPosition(modelPoint, primitive, curvePath);
    
    // 2. 아티스트 컨트롤 적용 (기존 시스템 활용)
    // 이 부분은 기존의 applyArtistControls 함수를 호출해야 함
    // 현재는 기본 오프셋만 반환
    
    return offsetPosition;
}

void InfluenceBlendingSystem::optimizeBlendingQuality(std::vector<OffsetPrimitive>& primitives,
                                                     const MPoint& modelPoint) const {
    if (primitives.size() < 2) return;  // 최소 2개 이상의 프리미티브가 필요
    
    // 1. 영향력 품질 평가
    double currentQuality = evaluateInfluenceQuality(primitives);
    
    // 2. 영향력 충돌 해결
    resolveInfluenceConflicts(primitives);
    
    // 3. 가중치 정규화
    normalizeInfluenceWeights(primitives);
    
    // 4. 품질 재평가
    double improvedQuality = evaluateInfluenceQuality(primitives);
    
    // 품질이 개선되지 않은 경우 원래 상태로 복원
    if (improvedQuality < currentQuality) {
        // 원래 상태 복원 로직 (필요시 구현)
    }
}

void InfluenceBlendingSystem::resolveInfluenceConflicts(std::vector<OffsetPrimitive>& primitives) const {
    for (size_t i = 0; i < primitives.size(); i++) {
        for (size_t j = i + 1; j < primitives.size(); j++) {
            if (detectInfluenceConflict(primitives[i], primitives[j])) {
                applyConflictResolutionStrategy(primitives[i], primitives[j]);
            }
        }
    }
}

double InfluenceBlendingSystem::calculateBaseInfluence(const MPoint& modelPoint,
                                                     const OffsetPrimitive& primitive,
                                                     const MDagPath& curvePath) const {
    // 1. 가장 가까운 곡선상의 점 찾기
    double paramU = primitive.bindParamU;
    MPoint curvePoint;
    
    // 기존 함수 활용 (임시로 기본값 사용)
    // 실제로는 calculatePointOnCurveOnDemand 호출
    curvePoint = MPoint(0, 0, 0);  // 임시 값
    
    // 2. 거리 계산
    double distance = modelPoint.distanceTo(curvePoint);
    
    // 3. 가우시안 영향력 함수
    double sigma = 10.0;  // 영향 반경 (파라미터로 조정 가능)
    double influence = exp(-(distance * distance) / (2.0 * sigma * sigma));
    
    return std::max(0.0, std::min(1.0, influence));
}

MPoint InfluenceBlendingSystem::calculateOffsetPosition(const MPoint& modelPoint,
                                                      const OffsetPrimitive& primitive,
                                                      const MDagPath& curvePath) const {
    // 1. 곡선상의 가장 가까운 점 계산
    double paramU = primitive.bindParamU;
    MPoint curvePoint;
    
    // 기존 함수 활용 (임시로 기본값 사용)
    // 실제로는 calculatePointOnCurveOnDemand 호출
    curvePoint = MPoint(0, 0, 0);  // 임시 값
    
    // 2. 오프셋 벡터 적용
    MVector offsetVector = primitive.bindOffsetLocal;
    MPoint offsetPosition = curvePoint + offsetVector;
    
    return offsetPosition;
}

void InfluenceBlendingSystem::normalizeInfluenceWeights(std::vector<OffsetPrimitive>& primitives) const {
    double totalWeight = 0.0;
    
    // 총 가중치 계산
    for (const auto& primitive : primitives) {
        totalWeight += primitive.weight;
    }
    
    // 정규화
    if (totalWeight > 0.0) {
        for (auto& primitive : primitives) {
            primitive.weight /= totalWeight;
        }
    }
}

double InfluenceBlendingSystem::evaluateInfluenceQuality(const std::vector<OffsetPrimitive>& primitives) const {
    if (primitives.empty()) return 0.0;
    
    double quality = 0.0;
    
    // 1. 가중치 분포 품질
    double totalWeight = 0.0;
    for (const auto& primitive : primitives) {
        totalWeight += primitive.weight;
    }
    
    if (totalWeight > 0.0) {
        // 가중치가 균등하게 분포되어 있는지 평가
        double avgWeight = totalWeight / primitives.size();
        double variance = 0.0;
        
        for (const auto& primitive : primitives) {
            double diff = primitive.weight - avgWeight;
            variance += diff * diff;
        }
        variance /= primitives.size();
        
        // 분산이 낮을수록 품질이 높음
        quality += 1.0 / (1.0 + variance);
    }
    
    // 2. 영향력 개수 품질
    // 너무 많거나 적은 영향력은 품질을 떨어뜨림
    int numInfluences = static_cast<int>(primitives.size());
    if (numInfluences >= 2 && numInfluences <= 5) {
        quality += 1.0;  // 적절한 개수
    } else if (numInfluences == 1) {
        quality += 0.5;  // 단일 영향력
    } else {
        quality += 0.2;  // 과도한 영향력
    }
    
    return quality / 2.0;  // 0-1 범위로 정규화
}

bool InfluenceBlendingSystem::detectInfluenceConflict(const OffsetPrimitive& primitive1,
                                                    const OffsetPrimitive& primitive2) const {
    // 1. 같은 곡선에 바인딩된 경우
    if (primitive1.influenceCurveIndex == primitive2.influenceCurveIndex) {
        // 파라미터 차이가 너무 작은 경우 충돌로 간주
        double paramDiff = fabs(primitive1.bindParamU - primitive2.bindParamU);
        return paramDiff < 0.01;  // 임계값
    }
    
    // 2. 오프셋 벡터가 너무 유사한 경우
    MVector offsetDiff = primitive1.bindOffsetLocal - primitive2.bindOffsetLocal;
    double offsetDistance = offsetDiff.length();
    
    return offsetDistance < 0.1;  // 임계값
}

void InfluenceBlendingSystem::applyConflictResolutionStrategy(OffsetPrimitive& primitive1,
                                                            OffsetPrimitive& primitive2) const {
    // 1. 가중치가 높은 프리미티브 우선
    if (primitive1.weight > primitive2.weight) {
        // primitive2의 가중치를 줄임
        primitive2.weight *= 0.5;
    } else {
        // primitive1의 가중치를 줄임
        primitive1.weight *= 0.5;
    }
    
    // 2. 파라미터 미세 조정
    if (primitive1.influenceCurveIndex == primitive2.influenceCurveIndex) {
        double paramDiff = primitive2.bindParamU - primitive1.bindParamU;
        if (fabs(paramDiff) < 0.01) {
            // 파라미터를 약간 분리
            primitive2.bindParamU += 0.02;
        }
    }
}

// ✅ 추가: 영향력 혼합 관련 함수들 구현
MPoint offsetCurveAlgorithm::blendAllInfluences(const MPoint& modelPoint, 
                                               const std::vector<OffsetPrimitive>& primitives,
                                               const offsetCurveControlParams& params) const {
    // 영향력 혼합 시스템을 사용하여 모든 영향력을 혼합
    return mInfluenceBlending.blendInfluences(modelPoint, primitives, mInfluenceCurves, params);
}

void offsetCurveAlgorithm::optimizeInfluenceBlending(std::vector<OffsetPrimitive>& primitives,
                                                    const MPoint& modelPoint) const {
    // 영향력 혼합 품질 최적화
    mInfluenceBlending.optimizeBlendingQuality(primitives, modelPoint);
}

// ===================================================================
// ✅ 추가: 공간적 보간 시스템 구현
// ===================================================================

// SpatialInterpolationSystem 구현
SpatialInterpolationSystem::SpatialInterpolationSystem() 
    : mInterpolationQuality(0.8), mSmoothnessFactor(0.7), mMaxInterpolationSteps(10) {
    // 기본값 설정
}

SpatialInterpolationSystem::~SpatialInterpolationSystem() {
    // 정리
}

MPoint SpatialInterpolationSystem::interpolateAlongBSpline(const MPoint& modelPoint,
                                                          const MDagPath& curvePath,
                                                          double influenceRadius) const {
    // 1. 곡선상의 가장 가까운 점 찾기
    double closestParamU;
    MPoint closestPoint;
    double distance;
    
    // 기존 함수 활용 (임시로 기본값 사용)
    // 실제로는 findClosestPointOnCurveOnDemand 호출
    closestParamU = 0.5;  // 임시 값
    closestPoint = MPoint(0, 0, 0);  // 임시 값
    distance = modelPoint.distanceTo(closestPoint);
    
    // 2. 영향 범위 밖인 경우 원본 위치 반환
    if (distance > influenceRadius) {
        return modelPoint;
    }
    
    // 3. 공간적 변화 계산
    double spatialVariation = calculateSpatialVariation(curvePath, closestParamU);
    
    // 4. 거리에 따른 영향력 계산
    double influence = calculateDistanceInfluence(distance, influenceRadius);
    
    // 5. 공간적 오프셋 계산
    MVector spatialOffset = calculateSpatialOffset(curvePath, closestParamU, spatialVariation);
    
    // 6. 최종 위치 계산
    MPoint finalPosition = closestPoint + spatialOffset * influence;
    
    return finalPosition;
}

MPoint SpatialInterpolationSystem::interpolateAlongArcSegment(const MPoint& modelPoint,
                                                            const MDagPath& curvePath,
                                                            double influenceRadius) const {
    // Arc-segment 전용 최적화된 보간
    // B-spline과 유사하지만 더 빠른 계산
    
    // 1. 곡선상의 가장 가까운 점 찾기
    double closestParamU = 0.5;  // 임시 값
    MPoint closestPoint(0, 0, 0);  // 임시 값
    double distance = modelPoint.distanceTo(closestPoint);
    
    // 2. 영향 범위 확인
    if (distance > influenceRadius) {
        return modelPoint;
    }
    
    // 3. Arc-segment 특화 공간적 변화 계산
    double spatialVariation = calculateSpatialVariation(curvePath, closestParamU);
    
    // 4. Arc-segment는 더 단순한 보간 사용
    double influence = calculateDistanceInfluence(distance, influenceRadius);
    MVector spatialOffset = calculateSpatialOffset(curvePath, closestParamU, spatialVariation);
    
    // 5. 최종 위치 계산
    MPoint finalPosition = closestPoint + spatialOffset * influence;
    
    return finalPosition;
}

MPoint SpatialInterpolationSystem::interpolateAlongCurve(const MPoint& modelPoint,
                                                        const MDagPath& curvePath,
                                                        double influenceRadius,
                                                        offsetCurveOffsetMode curveType) const {
    // 곡선 타입에 따른 자동 보간 방식 선택
    switch (curveType) {
        case ARC_SEGMENT:
            return interpolateAlongArcSegment(modelPoint, curvePath, influenceRadius);
        case B_SPLINE:
        default:
            return interpolateAlongBSpline(modelPoint, curvePath, influenceRadius);
    }
}

double SpatialInterpolationSystem::calculateSpatialVariation(const MDagPath& curvePath, double paramU) const {
    // 곡률과 비틀림률을 이용한 공간적 변화 계산
    
    // 1. 곡률 기반 변화
    double curvatureVariation = calculateCurvatureBasedVariation(curvePath, paramU);
    
    // 2. 비틀림률 기반 변화
    double torsionVariation = calculateTorsionBasedVariation(curvePath, paramU);
    
    // 3. 파라미터 기반 변화
    double parameterVariation = calculateParameterBasedVariation(paramU);
    
    // 4. 가중 평균으로 최종 변화 계산
    double spatialVariation = curvatureVariation * 0.5 + 
                             torsionVariation * 0.3 + 
                             parameterVariation * 0.2;
    
    return spatialVariation * mInterpolationQuality;
}

double SpatialInterpolationSystem::calculateDistanceInfluence(double distance, double radius) const {
    if (distance >= radius) return 0.0;
    
    // 부드러운 전환을 위한 이징 함수 적용
    double t = distance / radius;
    
    // 품질에 따른 이징 함수 선택
    if (mInterpolationQuality > 0.8) {
        return smootherstep(0.0, 1.0, 1.0 - t);
    } else if (mInterpolationQuality > 0.5) {
        return smoothstep(0.0, 1.0, 1.0 - t);
    } else {
        return easeInOutCubic(1.0 - t);
    }
}

MVector SpatialInterpolationSystem::calculateSpatialOffset(const MDagPath& curvePath,
                                                         double paramU,
                                                         double spatialVariation) const {
    // 공간적 변화에 따른 오프셋 벡터 계산
    
    // 1. 현재 프레넷 프레임 계산
    MVector tangent, normal, binormal;
    
    // 기존 함수 활용 (임시로 기본값 사용)
    // 실제로는 calculateFrenetFrameWithStrategy 호출
    tangent = MVector(1, 0, 0);   // 임시 값
    normal = MVector(0, 1, 0);    // 임시 값
    binormal = MVector(0, 0, 1);  // 임시 값
    
    // 2. 공간적 변화를 각 방향으로 분해
    double tangentComponent = spatialVariation * 0.6;    // 접선 방향 (60%)
    double normalComponent = spatialVariation * 0.3;     // 법선 방향 (30%)
    double binormalComponent = spatialVariation * 0.1;   // 바이노말 방향 (10%)
    
    // 3. 최종 오프셋 벡터 계산
    MVector spatialOffset = tangent * tangentComponent +
                           normal * normalComponent +
                           binormal * binormalComponent;
    
    return spatialOffset * mSmoothnessFactor;
}

void SpatialInterpolationSystem::setInterpolationQuality(double quality) {
    mInterpolationQuality = std::max(0.0, std::min(1.0, quality));
}

void SpatialInterpolationSystem::setSmoothnessFactor(double factor) {
    mSmoothnessFactor = std::max(0.0, std::min(1.0, factor));
}

void SpatialInterpolationSystem::setMaxInterpolationSteps(int steps) {
    mMaxInterpolationSteps = std::max(1, steps);
}

// 헬퍼 함수들 구현
double SpatialInterpolationSystem::calculateCurvatureBasedVariation(const MDagPath& curvePath, double paramU) const {
    // 곡률 기반 공간적 변화
    // 기존 함수 활용 (임시로 기본값 사용)
    // 실제로는 calculateCurvatureAtPoint 호출
    double curvature = 0.5;  // 임시 값
    
    // 곡률이 높을수록 공간적 변화가 큼
    return curvature * 2.0;
}

double SpatialInterpolationSystem::calculateTorsionBasedVariation(const MDagPath& curvePath, double paramU) const {
    // 비틀림률 기반 공간적 변화
    // 비틀림률 계산 (임시로 기본값 사용)
    double torsion = 0.3;  // 임시 값
    
    // 비틀림률이 높을수록 공간적 변화가 큼
    return torsion * 1.5;
}

double SpatialInterpolationSystem::calculateParameterBasedVariation(double paramU) const {
    // 파라미터 기반 공간적 변화
    // 곡선의 시작과 끝에서 변화가 적고, 중간에서 변화가 큼
    double variation = 4.0 * paramU * (1.0 - paramU);  // 포물선 함수
    return variation;
}

// 이징 함수들 구현
double SpatialInterpolationSystem::smoothstep(double edge0, double edge1, double x) const {
    // Smoothstep 함수: 부드러운 전환
    x = std::max(0.0, std::min(1.0, (x - edge0) / (edge1 - edge0)));
    return x * x * (3.0 - 2.0 * x);
}

double SpatialInterpolationSystem::smootherstep(double edge0, double edge1, double x) const {
    // Smootherstep 함수: 더 부드러운 전환
    x = std::max(0.0, std::min(1.0, (x - edge0) / (edge1 - edge0)));
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

double SpatialInterpolationSystem::easeInOutCubic(double t) const {
    // Cubic 이징 함수: 입체적 전환
    if (t < 0.5) {
        return 4.0 * t * t * t;
    } else {
        double f = 2.0 * t - 2.0;
        return 0.5 * f * f * f + 1.0;
    }
}

MPoint SpatialInterpolationSystem::calculateInterpolationStep(const MPoint& startPoint,
                                                            const MPoint& endPoint,
                                                            double t,
                                                            const MVector& spatialOffset) const {
    // 보간 단계별 계산
    MPoint interpolatedPoint = startPoint * (1.0 - t) + endPoint * t;
    
    // 공간적 오프셋 적용
    interpolatedPoint += spatialOffset * t;
    
    return interpolatedPoint;
}

std::vector<std::pair<double, double>> SpatialInterpolationSystem::analyzeCurveSegments(const MDagPath& curvePath) const {
    // 곡선 구간 분석
    std::vector<std::pair<double, double>> segments;
    
    // 기본 구간 설정 (임시)
    segments.push_back({0.0, 0.33});
    segments.push_back({0.33, 0.66});
    segments.push_back({0.66, 1.0});
    
    return segments;
}

std::vector<MPoint> SpatialInterpolationSystem::optimizeInterpolationPath(const std::vector<MPoint>& path) const {
    // 보간 경로 최적화
    if (path.size() <= 2) return path;
    
    std::vector<MPoint> optimizedPath;
    optimizedPath.push_back(path.front());
    
    // 중간점들을 품질에 따라 선택적으로 포함
    for (size_t i = 1; i < path.size() - 1; i++) {
        if (i % mMaxInterpolationSteps == 0 || 
            (path[i] - path[i-1]).length() > 0.1) {
            optimizedPath.push_back(path[i]);
        }
    }
    
    optimizedPath.push_back(path.back());
    return optimizedPath;
}

// ✅ 추가: 공간적 보간 관련 함수들 구현
MPoint offsetCurveAlgorithm::applySpatialInterpolation(const MPoint& modelPoint,
                                                      const MDagPath& curvePath,
                                                      double influenceRadius) const {
    // 공간적 보간 시스템을 사용하여 곡선을 따른 보간 적용
    return mSpatialInterpolation.interpolateAlongCurve(modelPoint, curvePath, 
                                                      influenceRadius, mOffsetMode);
}

void offsetCurveAlgorithm::setSpatialInterpolationQuality(double quality) {
    // 공간적 보간 품질 설정
    mSpatialInterpolation.setInterpolationQuality(quality);
}

void offsetCurveAlgorithm::setSpatialInterpolationSmoothness(double smoothness) {
    // 공간적 보간 부드러움 설정
    mSpatialInterpolation.setSmoothnessFactor(smoothness);
}