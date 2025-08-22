#ifndef OFFSET_CURVE_SYSTEMS_H
#define OFFSET_CURVE_SYSTEMS_H

#include <maya/MObject.h>
#include <maya/MString.h>
#include <maya/MStatus.h>

// OCD 시스템 관리 클래스
class OffsetCurveSystems {
public:
  OffsetCurveSystems();
  ~OffsetCurveSystems();
  
  // 시스템 초기화
  static MStatus initialize();
  
  // 시스템 정리
  static MStatus uninitialize();
  
  // 곡선 타입별 시스템 등록
  static MStatus registerCurveSystems();
  
  // 곡선 타입별 시스템 해제
  static MStatus unregisterCurveSystems();
  
  // B-spline 시스템 등록
  static MStatus registerBSplineSystem();
  
  // Arc-segment 시스템 등록
  static MStatus registerArcSegmentSystem();
  
  // 시스템 유효성 검사
  static bool validateSystem(const MObject& systemNode);
  
  // 시스템 타입 확인
  static int getSystemType(const MObject& systemNode);
  
  // 시스템 파라미터 추출
  static MStatus extractSystemParameters(const MObject& systemNode, 
                                       MString& systemType,
                                       MObject& controlObject);
};

#endif
