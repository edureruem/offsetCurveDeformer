/**
 * offsetCurveStrategy.h
 * 오프셋 곡선 계산을 위한 전략 패턴 인터페이스 - 리팩토링 버전
 */

#ifndef OFFSETCURVESTRATEGY_H
#define OFFSETCURVESTRATEGY_H

#include <maya/MStatus.h>
#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <vector>

#include "offsetCurveData.h"
#include "offsetCurveBinding.h" // 바인딩 클래스 포함

// 전방 선언
class offsetCurveBinding;

// 기본 오프셋 전략 인터페이스
class BaseOffsetCurveStrategy {
public:
    BaseOffsetCurveStrategy();
    virtual ~BaseOffsetCurveStrategy();
    
    // 필수 구현 메소드
    virtual MStatus computeOffsets(unsigned int vertexIndex, 
                                  const MPoint& point,
                                  const offsetCurveData& curveData,
                                  std::vector<offsetCurveBinding>& influences) = 0;
    
    // 공통 구현 메소드 (서브클래스에서 오버라이드 가능)
    virtual void computeFrenetFrame(const offsetCurveData& curveData, 
                                   double paramU,
                                   MVector& tangent, 
                                   MVector& normal, 
                                   MVector& binormal);
                                   
    // 각 전략별로 다른 구현이 필요한 메소드
    virtual double findClosestPointOnCurve(const MPoint& point, 
                                         const offsetCurveData& curveData,
                                         double& paramU,
                                         MPoint& closestPoint);

protected:
    // 공통 유틸리티 메서드 (서브클래스에서 사용)
    MStatus createInfluence(unsigned int vertexIndex,
                           const MPoint& point,
                           const MPoint& curvePoint,
                           double paramU,
                           const MVector& tangent,
                           const MVector& normal,
                           const MVector& binormal,
                           double curvature,
                           offsetCurveBinding& influence);
                           
    // 행렬 생성 유틸리티
    MMatrix createFrenetFrame(const MVector& tangent, const MVector& normal, const MVector& binormal, const MPoint& origin);
    
    // 각도 정규화 유틸리티
    double normalizeAngle(double angle);
};

// 아크 세그먼트 전략 구현
class ArcSegmentStrategy : public BaseOffsetCurveStrategy {
public:
    ArcSegmentStrategy();
    virtual ~ArcSegmentStrategy();
    
    virtual MStatus computeOffsets(unsigned int vertexIndex, 
                                  const MPoint& point,
                                  const offsetCurveData& curveData,
                                  std::vector<offsetCurveBinding>& influences) override;
                                   
    virtual double findClosestPointOnCurve(const MPoint& point, 
                                         const offsetCurveData& curveData,
                                         double& paramU,
                                         MPoint& closestPoint) override;
                                         
private:
    // 아크 세그먼트 전략에 특화된 메소드
    MStatus computeArcSegmentJunction(const MPoint& p1, const MPoint& p2, 
                                     const MVector& t1, const MVector& t2,
                                     MPoint& center, double& radius);
    
    MStatus handleParallelSegment(unsigned int vertexIndex,
                                 const MPoint& point, 
                                 const offsetCurveData& curveData,
                                 double paramU,
                                 int segmentIndex,
                                 std::vector<offsetCurveBinding>& influences);
                                 
    MStatus handleJunctionSegment(unsigned int vertexIndex,
                                 const MPoint& point, 
                                 const offsetCurveData& curveData,
                                 double paramU,
                                 int segmentIndex,
                                 std::vector<offsetCurveBinding>& influences);
};

// B-스플라인 전략 구현
class BSplineStrategy : public BaseOffsetCurveStrategy {
public:
    BSplineStrategy();
    virtual ~BSplineStrategy();
    
    virtual MStatus computeOffsets(unsigned int vertexIndex, 
                                  const MPoint& point,
                                  const offsetCurveData& curveData,
                                  std::vector<offsetCurveBinding>& influences) override;
                                   
    virtual double findClosestPointOnCurve(const MPoint& point, 
                                         const offsetCurveData& curveData,
                                         double& paramU,
                                         MPoint& closestPoint) override;
                                         
private:
    // B-스플라인 전략에 특화된 메소드
    double findBSplineMappingOptimized(const MPoint& point,
                                      const offsetCurveData& curveData,
                                      double& paramU);
                                      
    MStatus optimizeCurvatureEffect(offsetCurveBinding& influence, 
                                   double curvature);
};

// 전략 팩토리
class OffsetCurveStrategyFactory {
public:
    static BaseOffsetCurveStrategy* createStrategy(int offsetMode);
};

#endif // OFFSETCURVESTRATEGY_H