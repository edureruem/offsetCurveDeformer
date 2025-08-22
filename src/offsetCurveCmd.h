#ifndef OFFSET_CURVE_CMD_H
#define OFFSET_CURVE_CMD_H

#include <maya/MArgList.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MDGModifier.h>
#include <maya/MFloatArray.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MMatrixArray.h>
#include <maya/MObjectArray.h>
#include <maya/MPlug.h>
#include <maya/MPointArray.h>
#include <maya/MSelectionList.h>
#include <maya/MString.h>
#include <maya/MStringArray.h>
#include <maya/MThreadPool.h>
#include <maya/MPxCommand.h>

#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include "common.h"
#include "offsetCurveAlgorithm.h"

// OCD 바인딩 데이터 구조체
struct OCDBindData {
  MPointArray inputPoints;           // 변형될 지오메트리의 월드 공간 포인트들
  std::vector<OffsetCurveData> influenceCurves;  // 영향 곡선들
  MMatrixArray influenceMatrices;    // 영향 곡선들의 월드 매트릭스들
  double offsetDistance;             // 오프셋 거리
  double falloffRadius;              // 영향 감쇠 반경
  CurveType curveType;               // 곡선 타입
  
  // 스레드에서 계산되는 요소들
  std::vector<MIntArray> sampleCurveIds;    // 샘플링된 곡선 ID들
  std::vector<MFloatArray> sampleWeights;   // 샘플링 가중치들
  MMatrixArray bindMatrices;                // 바인딩 매트릭스들
  MVectorArray offsetVectors;               // 오프셋 벡터들
  MPointArray samplePoints;                 // 샘플링된 포인트들
};

// OCD 명령어 클래스
class OffsetCurveCmd : public MPxCommand {              
 public:
  enum CommandMode { 
    kCommandCreate,      // 새로운 OCD 디포머 생성
    kCommandExport,      // 바인딩 데이터 내보내기
    kCommandImport,      // 바인딩 데이터 가져오기
    kCommandHelp,        // 도움말 표시
    kCommandRebind       // 선택된 컴포넌트 재바인딩
  };
  
  OffsetCurveCmd();              
  virtual MStatus doIt(const MArgList&);
  virtual MStatus undoIt();
  virtual MStatus redoIt();
  virtual bool isUndoable() const;
  static void* creator();    
  static MSyntax newSyntax();

  // 멀티스레딩 태스크 생성
  static void CreateTasks(void *data, MThreadRootTask *pRoot);
  static MThreadRetVal CalculateBindingTask(void *pParam);

  const static char* kName;
  
  // 명령어 플래그들
  const static char* kNameFlagShort;
  const static char* kNameFlagLong;
  const static char* kOffsetDistanceFlagShort;
  const static char* kOffsetDistanceFlagLong;
  const static char* kFalloffRadiusFlagShort;
  const static char* kFalloffRadiusFlagLong;
  const static char* kCurveTypeFlagShort;
  const static char* kCurveTypeFlagLong;
  const static char* kNewBindMeshFlagShort;
  const static char* kNewBindMeshFlagLong;
  const static char* kExportFlagShort;
  const static char* kExportFlagLong;
  const static char* kImportFlagShort;
  const static char* kImportFlagLong;
  const static char* kBindingFlagShort;
  const static char* kBindingFlagLong;
  const static char* kRebindFlagShort;
  const static char* kRebindFlagLong;
  const static char* kHelpFlagShort;
  const static char* kHelpFlagLong;

 private:
  // 명령어 인수 수집
  MStatus GatherCommandArguments(const MArgList& args);

  // 지오메트리 경로 획득
  MStatus GetGeometryPaths();

  // OCD 디포머 생성
  MStatus CreateOffsetCurveDeformer();

  // 최신 OCD 노드 획득
  MStatus GetLatestOCDNode();

  // 바인딩 메시 생성
  MStatus CreateBindMesh(MDagPath& pathBindMesh);

  // 바인딩 메시 연결
  MStatus ConnectBindMesh(MDagPath& pathBindMesh);

  // 바인딩 데이터 계산
  MStatus CalculateBinding(MDagPath& pathBindMesh, OCDBindData& bindData, MDGModifier& dgMod);
    
  // 기존 바인딩 메시 획득
  MStatus GetExistingBindMesh(MDagPath &pathBindMesh);

  // 재바인딩 수행
  MStatus Rebind();

  // 바인딩 메시 획득
  MStatus GetBindMesh(MObject& oOCDNode, MDagPath& pathBindMesh);

  // 재바인딩용 서브셋 메시 생성
  MStatus CreateRebindSubsetMesh(MDagPath& pathDriverSubset);

  // 영향 곡선 데이터 추출
  MStatus ExtractInfluenceCurves(const MDagPathArray& curvePaths, 
                                std::vector<OffsetCurveData>& curves);

  // B-spline 곡선 데이터 추출
  MStatus ExtractBSplineCurve(const MDagPath& curvePath, BSplineCurve& curve);

  // Arc-segment 곡선 데이터 추출
  MStatus ExtractArcSegmentCurve(const MDagPath& curvePath, ArcSegmentCurve& curve);

  // 명령어 변수들
  MString name_;                    // OCD 노드 이름
  double offsetDistance_;           // 오프셋 거리
  double falloffRadius_;            // 영향 감쇠 반경
  CurveType curveType_;             // 곡선 타입
  CommandMode command_;             // 명령어 모드
  MString filePath_;                // 파일 경로
  bool useBinding_;                 // 바인딩 파일 사용 여부
  bool newBindMesh_;                // 새로운 바인딩 메시 생성 여부
  
  MSelectionList selectionList_;    // 선택된 명령어 입력 노드들
  MObject oOCDNode_;               // OCD 노드 객체
  MDagPathArray pathInfluenceCurves_; // 영향 곡선들의 경로
  MObjectArray influenceComponents_;   // 선택된 영향 컴포넌트들
  MDagPathArray pathDriven_;       // 변형될 지오메트리들의 경로
  MObjectArray drivenComponents_;  // 선택된 변형 컴포넌트들
  
  MDGModifier dgMod_;              // DG 수정자
  MStringArray bindMeshes_;        // 바인딩 메시들
};

#endif
