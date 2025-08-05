/**
 * offsetCurveStrategy.cpp
 * 오프셋 곡선 계산 전략 구현 - 리팩토링 버전
 */

#include "offsetCurveStrategy.h"
#include <maya/MGlobal.h>
#include <algorithm>
#include <cmath>
#include <limits>

// BaseOffsetCurveStrategy 구현
BaseOffsetCurveStrategy::BaseOffsetCurveStrategy()
{
}

BaseOffsetCurveStrategy::~BaseOffsetCurveStrategy()
{
}

void BaseOffsetCurveStrategy::computeFrenetFrame(const offsetCurveData& curveData, 
                                               double paramU,
                                               MVector& tangent, 
                                               MVector& normal, 
                                               MVector& binormal)
{
    // 탄젠트 계산
    curveData.getTangent(paramU, tangent);
    tangent.normalize();
    
    // CV 인덱스 추정
    int numCVs = curveData.getNumCVs();
    int cvIndex = static_cast<int>(paramU * numCVs);
    if (cvIndex >= numCVs) cvIndex = numCVs - 1;
    
    // 방향 데이터 가져오기
    double orientation;
    curveData.getOrientation(cvIndex, orientation);
    
    // 초기 노멀 벡터 (방향 기반)
    normal = MVector(cos(orientation), sin(orientation), 0.0);
    
    // 탄젠트와 수직이 되도록 노멀 조정
    normal = normal - (normal * tangent) * tangent;
    normal.normalize();
    
    // 바이노멀 계산 (외적)
    binormal = tangent ^ normal;
    binormal.normalize();
}

double BaseOffsetCurveStrategy::findClosestPointOnCurve(const MPoint& point, 
                                                     const offsetCurveData& curveData,
                                                     double& paramU,
                                                     MPoint& closestPoint)
{
    // 기본 구현 - 대부분의 경우 각 전략에서 오버라이드됨
    double distance = curveData.findClosestPoint(point, paramU);
    curveData.getPoint(paramU, closestPoint);
    return distance;
}

MStatus BaseOffsetCurveStrategy::createInfluence(unsigned int vertexIndex,
                                              const MPoint& point,
                                              const MPoint& curvePoint,
                                              double paramU,
                                              const MVector& tangent,
                                              const MVector& normal,
                                              const MVector& binormal,
                                              double curvature,
                                              offsetCurveBinding& influence)
{
    // 로컬 좌표 계산
    MVector toPoint = point - curvePoint;
    double localX = toPoint * normal;
    double localY = toPoint * binormal;
    double localZ = toPoint * tangent;
    
    // 영향 객체 설정
    influence.setParamU(paramU);
    influence.setWeight(1.0); // 나중에 조정
    influence.setBindLocalPoint(MPoint(localX, localY, localZ));
    influence.setTangent(tangent);
    influence.setNormal(normal);
    influence.setBinormal(binormal);
    influence.setCurvature(curvature);
    
    // 바인드 행렬 설정
    MMatrix matrix = createFrenetFrame(tangent, normal, binormal, curvePoint);
    influence.setBindMatrix(matrix);
    
    return MS::kSuccess;
}

MMatrix BaseOffsetCurveStrategy::createFrenetFrame(const MVector& tangent, 
                                                const MVector& normal, 
                                                const MVector& binormal, 
                                                const MPoint& origin)
{
    // 프레넷 프레임 행렬 생성
    double matrixArray[4][4] = {
        {tangent.x, normal.x, binormal.x, origin.x},
        {tangent.y, normal.y, binormal.y, origin.y},
        {tangent.z, normal.z, binormal.z, origin.z},
        {0.0, 0.0, 0.0, 1.0}
    };
    return MMatrix(matrixArray);
}

double BaseOffsetCurveStrategy::normalizeAngle(double angle)
{
    // 각도를 0 ~ 2π 범위로 정규화
    const double TWO_PI = 2.0 * M_PI;
    while (angle < 0.0) angle += TWO_PI;
    while (angle >= TWO_PI) angle -= TWO_PI;
    return angle;
}

// 전략 팩토리 구현
BaseOffsetCurveStrategy* OffsetCurveStrategyFactory::createStrategy(int offsetMode) 
{
    switch (offsetMode) {
        case 0: // ARC_SEGMENT
            return new ArcSegmentStrategy();
        case 1: // B_SPLINE
            return new BSplineStrategy();
        default:
            return new ArcSegmentStrategy(); // 기본값
    }
}

// ArcSegmentStrategy 구현
ArcSegmentStrategy::ArcSegmentStrategy() : BaseOffsetCurveStrategy()
{
}

ArcSegmentStrategy::~ArcSegmentStrategy()
{
}

MStatus ArcSegmentStrategy::computeOffsets(unsigned int vertexIndex,
                                         const MPoint& point,
                                         const offsetCurveData& curveData,
                                         std::vector<offsetCurveBinding>& influences)
{
    MStatus status;
    
    // 1. 곡선에서 가장 가까운 점 찾기
    double paramU;
    MPoint closestPoint;
    double distance = findClosestPointOnCurve(point, curveData, paramU, closestPoint);
    
    // 영향 반경 검사 (임의 값, 실제로는 매개변수화 필요)
    if (distance > 10.0) {
        return MS::kSuccess;  // 영향 없음
    }
    
    // 2. 어떤 세그먼트에 있는지 결정
    int segmentIndex = curveData.getSegmentIndex(paramU);
    
    // 3. 세그먼트 또는 연결부(junction)에 따라 다른 처리
    MPoint segStart, segEnd;
    MVector startTangent, endTangent;
    
    if (curveData.getSegmentPoints(segmentIndex, segStart, segEnd) &&
        curveData.getSegmentVectors(segmentIndex, startTangent, endTangent)) {
        
        // 탄젠트가 평행한지 확인 (같은 방향 또는 반대 방향)
        double dotProduct = startTangent * endTangent;
        
        if (fabs(dotProduct - 1.0) < 0.001) {
            // 평행 세그먼트
            return handleParallelSegment(vertexIndex, point, curveData, paramU, segmentIndex, influences);
        }
        else {
            // 연결부(Junction)
            return handleJunctionSegment(vertexIndex, point, curveData, paramU, segmentIndex, influences);
        }
    }
    
    return MS::kSuccess;
}

double ArcSegmentStrategy::findClosestPointOnCurve(const MPoint& point, 
                                                 const offsetCurveData& curveData,
                                                 double& paramU,
                                                 MPoint& closestPoint)
{
    // 기본 곡선 검색 사용
    double distance = curveData.findClosestPoint(point, paramU);
    curveData.getPoint(paramU, closestPoint);
    return distance;
}

MStatus ArcSegmentStrategy::handleParallelSegment(unsigned int vertexIndex,
                                               const MPoint& point,
                                               const offsetCurveData& curveData,
                                               double paramU,
                                               int segmentIndex,
                                               std::vector<offsetCurveBinding>& influences)
{
    // 세그먼트 정보 가져오기
    MPoint segStart, segEnd;
    MVector startTangent, endTangent;
    
    curveData.getSegmentPoints(segmentIndex, segStart, segEnd);
    curveData.getSegmentVectors(segmentIndex, startTangent, endTangent);
    
    // 세그먼트 파라미터 정규화 (0~1)
    double localParamT = (paramU - (double)segmentIndex / curveData.getNumSegments()) * curveData.getNumSegments();
    
    // 세그먼트 상의 점 계산
    MPoint segmentPoint = segStart * (1.0 - localParamT) + segEnd * localParamT;
    
    // 프레넷 프레임 계산
    MVector tangent = endTangent;
    MVector normal, binormal;
    computeFrenetFrame(curveData, paramU, tangent, normal, binormal);
    
    // 영향 생성
    offsetCurveBinding influence;
    influence.setCurveIndex(-1);  // 나중에 설정
    
    // 공통 영향 정보 설정
    double curvature;
    curveData.getCurvature(paramU, curvature);
    
    createInfluence(vertexIndex, point, segmentPoint, paramU, tangent, normal, 
                   binormal, curvature, influence);
    
    // 세그먼트 고유 정보 설정
    influence.setSegmentIndex(segmentIndex);
    influence.setIsJunction(false);
    influence.setSegmentLength(segStart.distanceTo(segEnd));
    
    influences.push_back(influence);
    
    return MS::kSuccess;
}

MStatus ArcSegmentStrategy::handleJunctionSegment(unsigned int vertexIndex,
                                               const MPoint& point,
                                               const offsetCurveData& curveData,
                                               double paramU,
                                               int segmentIndex,
                                               std::vector<offsetCurveBinding>& influences)
{
    // 세그먼트 정보 가져오기
    MPoint segStart, segEnd;
    MVector startTangent, endTangent;
    
    curveData.getSegmentPoints(segmentIndex, segStart, segEnd);
    curveData.getSegmentVectors(segmentIndex, startTangent, endTangent);
    
    // 호의 중심과 반경 계산
    MPoint arcCenter;
    double radius;
    MStatus status = computeArcSegmentJunction(segStart, segEnd, startTangent, endTangent, arcCenter, radius);
    
    if (status != MS::kSuccess) {
        return status;
    }
    
    // 가장 가까운 점 계산
    MPoint closestPoint;
    curveData.getPoint(paramU, closestPoint);
    
    // 호 상의 벡터 계산
    MVector radialVector = closestPoint - arcCenter;
    radialVector.normalize();
    
    // 탄젠트는 호에 접선
    MVector tangent = (radialVector ^ MVector(0.0, 1.0, 0.0)).normal();
    
    // 노멀과 바이노멀 계산
    MVector normal = radialVector;
    MVector binormal = tangent ^ normal;
    
    // 영향 생성
    offsetCurveBinding influence;
    influence.setCurveIndex(-1);  // 나중에 설정
    
    // 공통 영향 정보 설정
    double curvature = 1.0 / radius;  // 곡률 = 1/반경
    
    createInfluence(vertexIndex, point, closestPoint, paramU, tangent, normal, 
                   binormal, curvature, influence);
    
    // 연결부 고유 정보 설정
    influence.setSegmentIndex(segmentIndex);
    influence.setIsJunction(true);
    influence.setJunctionRadius(radius);
    influence.setSegmentLength(segStart.distanceTo(segEnd));
    
    influences.push_back(influence);
    
    return MS::kSuccess;
}

MStatus ArcSegmentStrategy::computeArcSegmentJunction(const MPoint& p1, const MPoint& p2, 
                                                   const MVector& t1, const MVector& t2,
                                                   MPoint& center, double& radius)
{
    // 두 접선의 교차점 계산으로 호의 중심 찾기
    MVector l1 = t1;
    MVector l2 = t2;
    
    // 2D 평면 가정 (실제로는 3D 일반화 필요)
    MVector n1(-l1.y, l1.x, 0.0);
    MVector n2(-l2.y, l2.x, 0.0);
    
    // 직선 방정식 설정: p + t*n
    // p1 + t1*n1 = p2 + t2*n2 풀기
    
    // 행렬 방정식 설정
    double a[2][2] = {
        {n1.x, -n2.x},
        {n1.y, -n2.y}
    };
    
    double b[2] = {
        p2.x - p1.x,
        p2.y - p1.y
    };
    
    // 간단한 2x2 행렬 풀이
    double det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    
    if (fabs(det) < 1e-10) {
        // 병렬 또는 거의 병렬인 접선
        return MS::kFailure;
    }
    
    double t1Param = (b[0] * a[1][1] - b[1] * a[0][1]) / det;
    
    // 중심점 계산
    center.x = p1.x + n1.x * t1Param;
    center.y = p1.y + n1.y * t1Param;
    center.z = (p1.z + p2.z) * 0.5;  // Z는 평균 (더 정확한 계산 가능)
    
    // 반경 계산
    radius = (center - p1).length();
    
    return MS::kSuccess;
}

// BSplineStrategy 구현
BSplineStrategy::BSplineStrategy() : BaseOffsetCurveStrategy()
{
}

BSplineStrategy::~BSplineStrategy()
{
}

MStatus BSplineStrategy::computeOffsets(unsigned int vertexIndex,
                                      const MPoint& point,
                                      const offsetCurveData& curveData,
                                      std::vector<offsetCurveBinding>& influences)
{
    MStatus status;
    
    // B-스플라인 방식으로 가장 가까운 점 찾기
    double paramU;
    MPoint closestPoint;
    double distance = findClosestPointOnCurve(point, curveData, paramU, closestPoint);
    
    // 영향 범위 확인 (실제 구현에서는 매개변수화 필요)
    if (distance > 10.0) {
        return MS::kSuccess;  // 영향 없음
    }
    
    // 프레넷 프레임 계산
    MVector tangent, normal, binormal;
    computeFrenetFrame(curveData, paramU, tangent, normal, binormal);
    
    // 곡률 계산
    double curvature;
    curveData.getCurvature(paramU, curvature);
    
    // 영향 생성
    offsetCurveBinding influence;
    influence.setCurveIndex(-1);  // 나중에 설정
    
    createInfluence(vertexIndex, point, closestPoint, paramU, tangent, normal, 
                   binormal, curvature, influence);
    
    // B-스플라인 특화 속성 설정
    optimizeCurvatureEffect(influence, curvature);
    
    influences.push_back(influence);
    
    return MS::kSuccess;
}

double BSplineStrategy::findClosestPointOnCurve(const MPoint& point, 
                                              const offsetCurveData& curveData,
                                              double& paramU,
                                              MPoint& closestPoint)
{
    // B-스플라인 최적화 알고리즘을 사용한 가까운 점 찾기
    double distance = findBSplineMappingOptimized(point, curveData, paramU);
    
    // 해당 파라미터에서 곡선 위의 점 계산
    curveData.getPoint(paramU, closestPoint);
    
    return distance;
}

double BSplineStrategy::findBSplineMappingOptimized(const MPoint& point,
                                                  const offsetCurveData& curveData,
                                                  double& paramU)
{
    // 곡선 매듭 벡터 가져오기
    MDoubleArray knots;
    curveData.getKnotVector(knots);
    
    // 초기 추정 (선형 맵핑)
    int numSegments = curveData.getNumSegments();
    double stepSize = 1.0 / numSegments;
    double bestParam = 0.0;
    double minDistance = std::numeric_limits<double>::max();
    
    // 균일하게 분할된 포인트에서 첫 검색
    for (int i = 0; i <= numSegments; i++) {
        double testU = i * stepSize;
        MPoint curvePoint;
        curveData.getPoint(testU, curvePoint);
        
        double dist = point.distanceTo(curvePoint);
        if (dist < minDistance) {
            minDistance = dist;
            bestParam = testU;
        }
    }
    
    // 뉴턴-랩슨 최적화로 정확도 향상
    // 이 부분에는 실제로 더 정교한 최적화 알고리즘 구현 필요
    double epsilon = 1e-5;
    int maxIterations = 10;
    
    for (int iter = 0; iter < maxIterations; iter++) {
        MPoint curvePoint;
        MVector tangent;
        
        curveData.getPoint(bestParam, curvePoint);
        curveData.getTangent(bestParam, tangent);
        
        // 현재 점과 곡선 위 점 사이의 벡터
        MVector diff = point - curvePoint;
        
        // 접선 방향 변화량 계산
        double alpha = diff * tangent;
        
        // 다음 단계 추정
        double nextParam = bestParam + alpha * stepSize;
        
        // 파라미터 범위 제한
        if (nextParam < 0.0) nextParam = 0.0;
        if (nextParam > 1.0) nextParam = 1.0;
        
        // 이전 추정값과 새 추정값의 차이
        double change = fabs(nextParam - bestParam);
        bestParam = nextParam;
        
        if (change < epsilon) {
            break;
        }
    }
    
    // 최종 파라미터와 최소 거리 반환
    paramU = bestParam;
    
    // 파라미터에서 점 계산 후 거리 계산
    MPoint finalPoint;
    curveData.getPoint(paramU, finalPoint);
    return point.distanceTo(finalPoint);
}

MStatus BSplineStrategy::optimizeCurvatureEffect(offsetCurveBinding& influence, 
                                               double curvature)
{
    // B-스플라인 특화 곡률 최적화
    // 실제 구현에서는 더 정교한 알고리즘 필요
    
    // 곡률이 높으면 B-스플라인 영향을 강화
    double optimizedCurvature = curvature * 1.5;
    influence.setCurvature(optimizedCurvature);
    
    return MS::kSuccess;
}