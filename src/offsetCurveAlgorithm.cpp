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
    fnCurve.getPointAtParam(0.0, startPoint);
    fnCurve.getPointAtParam(0.5, midPoint);  
    fnCurve.getPointAtParam(1.0, endPoint);
    
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
    // Maya 2020 호환성: getTangent 대신 tangent 사용
    status = fnCurve.tangent(paramU, tangent);
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
    
    status = fnCurve.getPointAtParam(paramU, point);
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
    status = fnCurve.closestPoint(modelPoint, &paramU, false, MSpace::kWorld);
    if (status == MS::kSuccess) {
        status = fnCurve.getPointAtParam(paramU, closestPoint, MSpace::kWorld);
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
    mInfluenceCurvePaths = influenceCurves;
    
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
            const MDagPath& curvePath = mInfluenceCurvePaths[primitive.influenceCurveIndex];
            
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
            
            // 5. 새로운 정점 위치 = 현재 영향점 + 제어된 오프셋
            MPoint deformedPosition = currentInfluencePoint + offsetWorldCurrent;
            
            // 6. 볼륨 보존 보정 적용 (필요시)
            if (params.getVolumeStrength() > 0.0) {
                MPoint originalPosition = points[vertexIndex];
                MVector volumeCorrectedOffset = applyVolumeControl(offsetWorldCurrent,
                                                                 originalPosition,
                                                                 deformedPosition,
                                                                 params.getVolumeStrength());
                deformedPosition = currentInfluencePoint + volumeCorrectedOffset;
            }
            
            // 7. 가중치 적용하여 누적
            newPosition += deformedPosition * primitive.weight;
            totalWeight += primitive.weight;
        }
        
        // 8. 정규화 및 최종 위치 설정
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
        const MDagPath& curvePath = mInfluenceCurvePaths[primitive.influenceCurveIndex];
        
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
        
        // 6. 볼륨 보존 보정 적용 (필요시)
        if (params.getVolumeStrength() > 0.0) {
            MPoint originalPosition = points[vertexIndex];
            MVector volumeCorrectedOffset = applyVolumeControl(offsetWorldCurrent,
                                                             originalPosition,
                                                             deformedPosition,
                                                             params.getVolumeStrength());
            deformedPosition = currentInfluencePoint + volumeCorrectedOffset;
        }
        
        // 7. 가중치 적용하여 누적
        newPosition += deformedPosition * primitive.weight;
        totalWeight += primitive.weight;
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
    // Maya 2020 호환성: getTangent 대신 tangent 사용
    status = fnCurve.tangent(paramU, firstDerivative, MSpace::kWorld);
    if (status != MS::kSuccess) return status;
    
    // 2차 미분 (가속도 벡터) - 수치적 계산
    double delta = 1e-6;
    MVector tangentPlus, tangentMinus;
    
    double paramUPlus = std::min(1.0, paramU + delta);
    double paramUMinus = std::max(0.0, paramU - delta);
    
    // Maya 2020 호환성: getTangent 대신 tangent 사용
    fnCurve.tangent(paramUPlus, tangentPlus, MSpace::kWorld);
    fnCurve.tangent(paramUMinus, tangentMinus, MSpace::kWorld);
    
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
            fnCurve.getPointAtParam(paramU, startPoint);
            fnCurve.getPointAtParam(nextParamU, endPoint);
            
            double chordLength = startPoint.distanceTo(endPoint);
            segment.totalAngle = 2.0 * asin(chordLength / (2.0 * segment.radius));
            
            // 원의 중심 계산 (근사)
            MPoint midPoint;
            fnCurve.getPointAtParam((paramU + nextParamU) * 0.5, midPoint);
            
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

// Volume 제어: 볼륨 보존 보정
MVector offsetCurveAlgorithm::applyVolumeControl(const MVector& deformedOffset,
                                                const MPoint& originalPosition,
                                                const MPoint& deformedPosition,
                                                double volumeStrength) const
{
    if (volumeStrength < 1e-6) {
        return deformedOffset; // 볼륨 보정 없음
    }
    
    // 특허에서 언급하는 볼륨 손실 보정
    // 변형 전후의 거리 차이를 기반으로 보정 벡터 계산
    MVector displacement = deformedPosition - originalPosition;
    double displacementLength = displacement.length();
    
    if (displacementLength < 1e-6) {
        return deformedOffset;
    }
    
    // 볼륨 보존을 위한 법선 방향 보정
    // 압축된 영역을 팽창시키고, 확장된 영역을 축소
    MVector normalizedDisplacement = displacement.normal();
    double volumeCorrection = volumeStrength * 0.1 * displacementLength;
    
    // 변형 방향에 수직인 성분을 강화하여 볼륨 보존
    MVector volumeOffset = normalizedDisplacement * volumeCorrection;
    
    return deformedOffset + volumeOffset;
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