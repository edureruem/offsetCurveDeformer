/**
 * offsetCurveDeformerNode.h
 * Maya 2020용 Offset Curve Deformer 노드
 * 소니 특허(US8400455) 기반 구현
 * cvWrap의 Maya API 사용법 완전 준수
 */

#pragma once

// Maya 헤더들
#include <maya/MPxDeformerNode.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>

// Maya API 버전별 멀티스레딩 및 GPU 지원 (cvWrap 방식)
#if MAYA_API_VERSION >= 201600
#include <maya/MThreadPool.h>
#include <maya/MPxGPUDeformer.h>
#include <maya/MGPUDeformerRegistry.h>
#include <maya/MOpenCLInfo.h>
#include <clew/clew_cl.h>
#endif

#include <maya/MDGModifier.h>
#include <maya/MMessage.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MFloatArray.h>
#include <maya/MIntArray.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MFnMesh.h>
#include <maya/MItGeometry.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMessageAttribute.h>

// C++ 표준 라이브러리
#include <vector>
#include <map>

#include "offsetCurveAlgorithm.h"
#include "offsetCurveControlParams.h"

// 전방 선언 (Forward Declaration)
class OffsetCurveGPUDeformer;

// cvWrap 방식의 TaskData 구조체 (완전 동일)
struct TaskData {
    MMatrix drivenMatrix;
    MMatrix drivenInverseMatrix;
    float envelope;
    float scale;

    MIntArray membership;
    MFloatArray paintWeights;
    MPointArray points;

    // cvWrap 방식의 드라이버 데이터
    MPointArray driverPoints;
    MFloatVectorArray driverNormals;
    MMatrixArray bindMatrices;
    std::vector<MIntArray> sampleIds;
    std::vector<MDoubleArray> sampleWeights;
    std::vector<MIntArray> triangleVerts;
    std::vector<MFloatArray> baryCoords;

    // OCD 특화 데이터 (특허 알고리즘 유지)
    std::vector<MDagPath> influenceCurves;
    std::vector<OffsetPrimitive> offsetPrimitives;
    offsetCurveControlParams controlParams;
};

// cvWrap 방식의 ThreadData 템플릿 (바인딩 계산용)
template<typename T>
struct ThreadData {
    T* pData;
    unsigned int start;
    unsigned int end;
    unsigned int threadId;
    int numTasks;
};

class OffsetCurveDeformerNode : public MPxDeformerNode {
public:
    OffsetCurveDeformerNode();
    virtual ~OffsetCurveDeformerNode();
    virtual void postConstructor();

    // 필수 오버라이드 메서드
    static void* creator();
    static MStatus initialize();
    
    // 변형 메서드 (cvWrap 방식)
    virtual MStatus deform(MDataBlock& data, MItGeometry& iter, 
                          const MMatrix& mat, unsigned int mIndex);
    
    // Maya 의존성 관리 (cvWrap 방식)
    virtual MStatus setDependentsDirty(const MPlug& plugBeingDirtied, 
                                      MPlugArray& affectedPlugs);
    
    // cvWrap 방식의 멀티스레딩 태스크 생성
    static void CreateTasks(void *data, MThreadRootTask *pRoot);
    static MThreadRetVal EvaluateOffsetCurve(void *pParam);
    
    // cvWrap 방식의 바인딩 정보 가져오기
    MStatus GetBindInfo(MDataBlock& data, unsigned int geomIndex, TaskData& taskData);
    MStatus GetDriverData(MDataBlock& data, TaskData& taskData);
    
    // 고급 OCD 알고리즘을 위한 컨트롤 파라미터 로딩
    MStatus getControlParamsFromData(MDataBlock& data, offsetCurveControlParams& controlParams);
    
    // 기본 동작 복구 함수들 (특허 알고리즘 유지)
    MStatus applyBasicDeformation(MPointArray& points, 
                                 const std::vector<MDagPath>& curves);
    double calculateDistanceToCurve(const MPoint& point, const MDagPath& curve);
    MVector calculateBasicOffset(const MPoint& point, const MDagPath& curve);
    
    // influenceCurve에서 데이터 가져오기
    MStatus getInfluenceCurve(MDataBlock& dataBlock, MDagPath& influenceCurve);
    
    // Maya 권장 방식: compute() 오버라이드하지 않음
    MStatus updateParameters(MDataBlock& dataBlock);
    MStatus rebindDeformer(MDataBlock& dataBlock, MItGeometry& iter);
    MStatus getCurvesFromInputs(MDataBlock& dataBlock, std::vector<MDagPath>& curves);
    MStatus getPoseTargetMesh(MDataBlock& dataBlock, MPointArray& points);
    MStatus initializeBinding(MDataBlock& dataBlock, MItGeometry& iter);
    
    // 변형 적용 함수 (cvWrap 방식: GPU 우선, CPU 폴백)
    MStatus applyDeformation(MPointArray& points, 
                            const std::vector<MDagPath>& curves,
                            MDataBlock& data, unsigned int mIndex);
    
    // GPU 가속 변형 적용
    MStatus applyGPUDeformation(MPointArray& points, 
                               const std::vector<MDagPath>& curves,
                               MDataBlock& data, unsigned int mIndex);
    
    // CPU 폴백 변형 적용
    MStatus applyCPUDeformation(MPointArray& points, 
                               const std::vector<MDagPath>& curves,
                               MDataBlock& data, unsigned int mIndex);
    
    // Maya 콜백 시스템
    static void aboutToDeleteCB(MObject &node, MDGModifier &modifier, void *clientData);
    
    // 다른 메서드
    virtual MStatus connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc);
    virtual MStatus connectionBroken(const MPlug& plug, const MPlug& otherPlug, bool asSrc);
    
    // 스킨 바인딩
    MStatus bindSkin();
    
    // 특허 기술 관련 메서드 (유지)
    MStatus applyVolumePreservationCorrection(MPointArray& points, 
                                            const offsetCurveControlParams& params);
    
    // 에러 처리 및 검증 메서드들
    bool validateInputData(MDataBlock& dataBlock);
    bool checkMemoryStatus();
    bool checkGPUStatus();
    bool validateOutputData(MItGeometry& iter);
    
    // 안전한 메모리 관리
    void cleanupResources();
    bool initializeResources();

private:
    // cvWrap 방식의 멤버 변수들
    std::map<unsigned int, bool> dirty_;
    std::vector<TaskData> taskData_;
    std::vector<ThreadData<TaskData>*> threadData_;
    MCallbackId onDeleteCallbackId;
    
    // OCD 특화 멤버 변수들 (특허 알고리즘 유지)
    offsetCurveAlgorithm* mAlgorithm;
    offsetCurveControlParams mControlParams;
    
    // GPU 가속 관련 멤버 변수들
    OffsetCurveGPUDeformer* mGPUDeformer;
    bool mGPUAvailable;
    bool mUseGPU;
    
    // 기존 OCD 멤버 변수들 (특허 알고리즘 유지)
    bool mNeedsRebind;
    bool mBindingInitialized;
    MPointArray mOriginalPoints;
    MPointArray mPoseTargetPoints;

public:
    // 노드 속성
    static MTypeId id;
    static const MString nodeName;

    // 기본 속성 (cvWrap 방식으로 단순화)
    static MObject aOffsetCurves;        // 오프셋 곡선들
    static MObject aCurvesData;          // 곡선 데이터
    static MObject aBindPose;            // 바인드 포즈
    
    // 바인딩 데이터 속성 (cvWrap 방식)
    static MObject aBindData;            // 바인딩 데이터 복합 속성
    static MObject aSampleComponents;    // 샘플 컴포넌트
    static MObject aSampleWeights;       // 샘플 가중치
    static MObject aTriangleVerts;       // 삼각형 버텍스
    static MObject aBarycentricWeights;  // Barycentric 가중치
    static MObject aBindMatrix;          // 바인딩 매트릭스
    static MObject aDriverGeo;           // 드라이버 지오메트리
    static MObject aNumTasks;            // 태스크 수
    static MObject aScale;               // 스케일
    
    // 리바인딩 속성
    static MObject aRebindMesh;          // 메시 리바인드
    static MObject aRebindCurves;        // 곡선 리바인드
    static MObject aUseParallel;         // 병렬 처리
    static MObject aDebugDisplay;        // 디버그 표시
    
    // influenceCurve 관련 속성들
    static MObject aInfluenceCurve;      // 영향 곡선 (복합 속성, 배열)
    static MObject aInfluenceCurveData;  // 영향 곡선 데이터 (NURBS 곡선)
    static MObject aInfluenceCurveGroupId; // 영향 곡선 그룹 ID
    
    // 아티스트 제어 속성 (특허 알고리즘 유지)
    static MObject aRotationDistribution;   // 회전 분포
    static MObject aScaleDistribution;      // 스케일 분포
    static MObject aTwistDistribution;      // 꼬임 분포
    static MObject aAxialSliding;           // 축 방향 슬라이딩
    
    // 포즈 타겟 속성
    static MObject aPoseTarget;             // 포즈 타겟
    static MObject aPoseWeight;             // 포즈 가중치

    // GPU 가속 관련 속성들
    static MObject aUseGPU;               // GPU 가속 사용 여부
    static MObject aGPUDevice;            // GPU 디바이스 선택
    static MObject aGPUMemoryLimit;       // GPU 메모리 제한
    static MObject aGPUBatchSize;         // GPU 배치 크기

    // 레거시 호환성을 위한 캐시
    std::vector<MDagPath> mCurvePaths;
};

// GPU 오버라이드 구현 (cvWrap과 정확히 동일한 패턴)
class OffsetCurveGPUDeformer : public MPxGPUDeformer {
public:
    // Virtual methods from MPxGPUDeformer
    OffsetCurveGPUDeformer();
    virtual ~OffsetCurveGPUDeformer();

    virtual MPxGPUDeformer::DeformerStatus evaluate(
        MDataBlock& block, 
        const MEvaluationNode& evaluationNode,
        const MPlug& plug, 
        const MGPUDeformerData& inputData,
        MGPUDeformerData& outputData);
    
    virtual void terminate();

    // cvWrap과 동일한 정적 메서드들
    static MGPUDeformerRegistrationInfo* GetGPUDeformerInfo();
    static bool ValidateNode(MDataBlock& block, const MEvaluationNode& evaluationNode, 
                           const MPlug& plug, MStringArray* messages);

    /** 플러그인 로드 경로 (OpenCL 커널 찾기용) */
    static MString pluginLoadPath;

private:
    // cvWrap과 동일한 헬퍼 메서드들
    MStatus EnqueueBindData(MDataBlock& data, const MEvaluationNode& evaluationNode, const MPlug& plug);
    MStatus EnqueueCurveData(MDataBlock& data, const MEvaluationNode& evaluationNode, const MPlug& plug);
    MStatus EnqueuePaintMapData(MDataBlock& data, const MEvaluationNode& evaluationNode, 
                               unsigned int numElements, const MPlug& plug);

    // GPU 메모리 저장소 (cvWrap과 동일)
    MAutoCLMem curvePoints_;
    MAutoCLMem curveTangents_;
    MAutoCLMem paintWeights_;
    MAutoCLMem bindMatrices_;
    MAutoCLMem sampleCounts_;
    MAutoCLMem sampleOffsets_;
    MAutoCLMem sampleIds_;
    MAutoCLMem sampleWeights_;
    MAutoCLMem triangleVerts_;
    MAutoCLMem baryCoords_;
    MAutoCLMem drivenMatrices_;

    unsigned int numElements_;

    // OpenCL 커널
    MAutoCLKernel kernel_;
};

/**
  GPU 디포머 등록 정보 (cvWrap 방식)
*/
class OffsetCurveGPUDeformerInfo : public MGPUDeformerRegistrationInfo {
public:
    OffsetCurveGPUDeformerInfo(){}
    virtual ~OffsetCurveGPUDeformerInfo(){}

    virtual MPxGPUDeformer* createGPUDeformer() {
        return new OffsetCurveGPUDeformer();
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
