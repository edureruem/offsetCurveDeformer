#ifndef OFFSET_CURVE_CONTROL_PARAMS_H
#define OFFSET_CURVE_CONTROL_PARAMS_H

#include <maya/MObject.h>
#include <maya/MString.h>

// OCD 제어 파라미터 클래스
class OffsetCurveControlParams {
public:
  OffsetCurveControlParams();
  ~OffsetCurveControlParams();
  
  // 파라미터 초기화
  static MStatus initialize();
  
  // 파라미터 정리
  static MStatus uninitialize();
  
  // 정적 파라미터 객체들
  static MObject aInfluenceCurves;      // 영향 곡선들
  static MObject aOffsetDistance;       // 오프셋 거리
  static MObject aFalloffRadius;        // 영향 감쇠 반경
  static MObject aCurveType;            // 곡선 타입
  static MObject aBindData;             // 바인딩 데이터
  static MObject aSamplePoints;         // 샘플링된 포인트들
  static MObject aSampleWeights;        // 샘플링 가중치들
  static MObject aOffsetVectors;        // 오프셋 벡터들
  static MObject aBindMatrices;         // 바인딩 매트릭스들
  static MObject aNumTasks;             // 태스크 수
  static MObject aEnvelope;             // 엔벨로프
  
  // 파라미터 기본값들
  static const double kDefaultOffsetDistance;
  static const double kDefaultFalloffRadius;
  static const int kDefaultCurveType;
  static const int kDefaultNumTasks;
  static const float kDefaultEnvelope;
  
  // 파라미터 범위들
  static const double kMinOffsetDistance;
  static const double kMaxOffsetDistance;
  static const double kMinFalloffRadius;
  static const double kMaxFalloffRadius;
  static const int kMinNumTasks;
  static const int kMaxNumTasks;
  static const float kMinEnvelope;
  static const float kMaxEnvelope;
};

#endif
