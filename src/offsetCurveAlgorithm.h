#ifndef OFFSET_CURVE_ALGORITHM_H
#define OFFSET_CURVE_ALGORITHM_H

#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MVector.h>
#include <maya/MVectorArray.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MFloatArray.h>
#include <maya/MIntArray.h>
#include <vector>

// 곡선 타입 열거형
enum CurveType {
  kBSpline = 0,      // B-spline 곡선
  kArcSegment = 1    // Arc-segment (원호 + 선분)
};

// B-spline 곡선 구조체
struct BSplineCurve {
  MPointArray controlPoints;  // 제어점들
  MFloatArray knots;          // 노트 벡터
  int degree;                 // 차수
  
  BSplineCurve() : degree(3) {}
  
  // 곡선상의 점 계산
  MPoint evaluateAt(double t) const;
  
  // 곡선의 접선 벡터 계산
  MVector tangentAt(double t) const;
  
  // 곡선의 법선 벡터 계산
  MVector normalAt(double t) const;
};

// Arc-segment 곡선 구조체 (특허의 특수한 형태)
struct ArcSegmentCurve {
  MPoint startPoint;          // 시작점
  MPoint endPoint;            // 끝점
  MPoint centerPoint;         // 원호 중심점
  double radius;              // 반지름
  double startAngle;          // 시작 각도
  double endAngle;            // 끝 각도
  bool isArc;                 // 원호인지 선분인지
  
  ArcSegmentCurve() : radius(0.0), startAngle(0.0), endAngle(0.0), isArc(false) {}
  
  // 곡선상의 점 계산
  MPoint evaluateAt(double t) const;
  
  // 곡선의 접선 벡터 계산
  MVector tangentAt(double t) const;
  
  // 곡선의 법선 벡터 계산
  MVector normalAt(double t) const;
};

// 오프셋 곡선 데이터 구조체
struct OffsetCurveData {
  CurveType type;             // 곡선 타입
  union {
    BSplineCurve bspline;     // B-spline 데이터
    ArcSegmentCurve arc;      // Arc-segment 데이터
  };
  
  OffsetCurveData() : type(kBSpline) {}
  ~OffsetCurveData() {}
};

// OCD 알고리즘 클래스
class OffsetCurveAlgorithm {
public:
  OffsetCurveAlgorithm();
  ~OffsetCurveAlgorithm();
  
  // 바인딩 단계: 각 모델 포인트에 대해 오프셋 곡선 생성
  MStatus bindModelPoints(const MPointArray& modelPoints,
                         const std::vector<OffsetCurveData>& influenceCurves,
                         double offsetDistance,
                         double falloffRadius,
                         MMatrixArray& bindMatrices,
                         MPointArray& samplePoints,
                         MFloatArray& sampleWeights,
                         MVectorArray& offsetVectors);
  
  // 변형 단계: 스켈레톤 애니메이션에 따른 오프셋 곡선 변형
  MStatus deformModelPoints(const MPointArray& originalPoints,
                           const MMatrixArray& bindMatrices,
                           const MPointArray& samplePoints,
                           const MFloatArray& sampleWeights,
                           const MVectorArray& offsetVectors,
                           const std::vector<OffsetCurveData>& deformedInfluenceCurves,
                           MPointArray& deformedPoints);
  
  // 특허의 핵심: 각 포인트에 대한 개별 오프셋 곡선 생성
  MStatus createOffsetCurveForPoint(const MPoint& modelPoint,
                                   const std::vector<OffsetCurveData>& influenceCurves,
                                   double offsetDistance,
                                   OffsetCurveData& offsetCurve);
  
  // B-spline 오프셋 곡선 생성
  MStatus createBSplineOffsetCurve(const BSplineCurve& baseCurve,
                                  const MPoint& offsetPoint,
                                  double distance,
                                  BSplineCurve& offsetCurve);
  
  // Arc-segment 오프셋 곡선 생성
  MStatus createArcSegmentOffsetCurve(const ArcSegmentCurve& baseCurve,
                                     const MPoint& offsetPoint,
                                     double distance,
                                     ArcSegmentCurve& offsetCurve);
  
  // 곡선 간의 거리 계산
  double calculateDistanceToCurve(const MPoint& point, const OffsetCurveData& curve);
  
  // 곡선의 가장 가까운 점 찾기
  MStatus findClosestPointOnCurve(const MPoint& point, 
                                  const OffsetCurveData& curve,
                                  MPoint& closestPoint,
                                  double& parameter);
  
  // 가중치 계산 (특허의 영향 메커니즘)
  float calculateInfluenceWeight(const MPoint& point,
                                const OffsetCurveData& curve,
                                double falloffRadius);
  
  // 볼륨 보존을 위한 오프셋 벡터 조정
  MStatus adjustOffsetVectorForVolumePreservation(const MPoint& originalPoint,
                                                 const MPoint& deformedPoint,
                                                 const MVector& originalOffset,
                                                 MVector& adjustedOffset);

private:
  // B-spline 계산 헬퍼 메서드들
  MStatus calculateBSplineBasisFunctions(int i, int k, double u, 
                                        const MFloatArray& knots, 
                                        MFloatArray& basisFunctions);
  
  // Arc-segment 계산 헬퍼 메서드들
  MPoint interpolateArcSegment(double t, const ArcSegmentCurve& arc) const;
  MVector calculateArcTangent(double t, const ArcSegmentCurve& arc) const;
  
  // 수학적 유틸리티
  double clampParameter(double t) const;
  bool isPointInArcSegment(const MPoint& point, const ArcSegmentCurve& arc) const;
};

#endif
