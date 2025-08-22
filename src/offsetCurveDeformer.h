#ifndef OFFSET_CURVE_DEFORMER_H
#define OFFSET_CURVE_DEFORMER_H

#include <maya/MDGModifier.h>
#include <maya/MFloatArray.h>
#include <maya/MIntArray.h>
#include <maya/MMatrix.h> 
#include <maya/MMatrixArray.h> 
#include <maya/MMessage.h>
#include <maya/MPoint.h> 
#include <maya/MPointArray.h>
#include <maya/MThreadPool.h>
#include <maya/MPxDeformerNode.h>

#if MAYA_API_VERSION >= 201600
#include <maya/MPxGPUDeformer.h>
#include <maya/MGPUDeformerRegistry.h>
#include <maya/MOpenCLInfo.h>
#include <clew/clew_cl.h>
#endif

#include <map>
#include <vector>
#include "common.h"
#include "offsetCurveAlgorithm.h"
#include "offsetCurveControlParams.h"
#include "offsetCurveSystems.h"

// OCD 디포머 노드
class OffsetCurveDeformer : public MPxDeformerNode {
 public:
  OffsetCurveDeformer();
  virtual ~OffsetCurveDeformer(); 
  
  virtual void postConstructor();
  virtual MStatus deform(MDataBlock& data, MItGeometry& iter, const MMatrix& mat,
                         unsigned int mIndex);
  virtual MStatus setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs);

  static void* creator();
  static MStatus initialize();

  // 멀티스레딩 태스크 생성
  static void CreateTasks(void *data, MThreadRootTask *pRoot);
  static MThreadRetVal EvaluateOffsetCurve(void *pParam);
    
  const static char* kName;
  static MObject aInfluenceCurves;      // 영향 곡선들
  static MObject aOffsetDistance;       // 오프셋 거리
  static MObject aFalloffRadius;        // 영향 감쇠 반경
  static MObject aCurveType;            // 곡선 타입 (B-spline 또는 Arc-segment)
  static MObject aBindData;             // 바인딩 데이터
  static MObject aSamplePoints;         // 샘플링된 포인트들
  static MObject aSampleWeights;        // 샘플링 가중치들
  static MObject aOffsetVectors;        // 오프셋 벡터들
  static MObject aBindMatrices;         // 바인딩 매트릭스들
  static MObject aNumTasks;             // 태스크 수
  static MObject aEnvelope;             // 엔벨로프
  static MTypeId id;

private:
  static void aboutToDeleteCB(MObject &node, MDGModifier &modifier, void *clientData);

  std::map<unsigned int, bool> dirty_;
  std::vector<OffsetCurveTaskData> taskData_;
  std::vector<ThreadData<OffsetCurveTaskData>*> threadData_;
  MCallbackId onDeleteCallbackId;
  
  // OCD 알고리즘 인스턴스
  OffsetCurveAlgorithm algorithm_;
};

#if MAYA_API_VERSION >= 201600
// GPU 오버라이드 구현
class OffsetCurveGPU : public MPxGPUDeformer {
 public:
  OffsetCurveGPU();
  virtual ~OffsetCurveGPU();

#if MAYA_API_VERSION <= 201700
  virtual MPxGPUDeformer::DeformerStatus evaluate(MDataBlock& block, const MEvaluationNode&,
                                                  const MPlug& plug, unsigned int numElements,
                                                  const MAutoCLMem, const MAutoCLEvent,
                                                  MAutoCLMem, MAutoCLEvent&);
#else
  virtual MPxGPUDeformer::DeformerStatus evaluate(MDataBlock& block, const MEvaluationNode& evaluationNode,
                                                  const MPlug& plug, const MGPUDeformerData& inputData,
                                                  MGPUDeformerData& outputData);
#endif
  virtual void terminate();

  static MGPUDeformerRegistrationInfo* GetGPUDeformerInfo();
  static bool ValidateNode(MDataBlock& block, const MEvaluationNode&, const MPlug& plug, MStringArray* messages);
  
  static MString pluginLoadPath;

private:
  // GPU 데이터 전송 헬퍼 메서드들
  MStatus EnqueueBindData(MDataBlock& data, const MEvaluationNode& evaluationNode, const MPlug& plug);
  MStatus EnqueueInfluenceCurves(MDataBlock& data, const MEvaluationNode& evaluationNode, const MPlug& plug);
  MStatus EnqueueOffsetData(MDataBlock& data, const MEvaluationNode& evaluationNode, const MPlug& plug);

  // GPU 메모리 저장소
  MAutoCLMem influenceCurves_;
  MAutoCLMem offsetVectors_;
  MAutoCLMem bindMatrices_;
  MAutoCLMem samplePoints_;
  MAutoCLMem sampleWeights_;
  MAutoCLMem offsetDistances_;
  MAutoCLMem falloffRadii_;

  unsigned int numElements_;
  MAutoCLKernel kernel_;
};

// GPU 디포머 등록 정보
class OffsetCurveGPUDeformerInfo : public MGPUDeformerRegistrationInfo {
 public:
  OffsetCurveGPUDeformerInfo(){}
  virtual ~OffsetCurveGPUDeformerInfo(){}

  virtual MPxGPUDeformer* createGPUDeformer() {
    return new OffsetCurveGPU();
  }

#if MAYA_API_VERSION >= 201650
  virtual bool validateNodeInGraph(MDataBlock& block, const MEvaluationNode& evaluationNode,
                                   const MPlug& plug, MStringArray* messages) {
    return true;
  }

  virtual bool validateNodeValues(MDataBlock& block, const MEvaluationNode& evaluationNode,
                                  const MPlug& plug, MStringArray* messages) {
    return true;
  }
#else
  virtual bool validateNode(MDataBlock& block, const MEvaluationNode& evaluationNode,
                            const MPlug& plug, MStringArray* messages) {
    return true;
  }
#endif
};

#endif // End Maya 2016

#endif
