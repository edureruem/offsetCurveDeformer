#include "offsetCurveDeformer.h"
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnMesh.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MGlobal.h>
#include <maya/MItGeometry.h>
#include <maya/MNodeMessage.h>
#include <maya/MPlugArray.h>
#include <cassert>

MTypeId OffsetCurveDeformer::id(0x0011580C);

const char* OffsetCurveDeformer::kName = "offsetCurveDeformer";
MObject OffsetCurveDeformer::aInfluenceCurves;
MObject OffsetCurveDeformer::aOffsetDistance;
MObject OffsetCurveDeformer::aFalloffRadius;
MObject OffsetCurveDeformer::aCurveType;
MObject OffsetCurveDeformer::aBindData;
MObject OffsetCurveDeformer::aSamplePoints;
MObject OffsetCurveDeformer::aSampleWeights;
MObject OffsetCurveDeformer::aOffsetVectors;
MObject OffsetCurveDeformer::aBindMatrices;
MObject OffsetCurveDeformer::aNumTasks;
MObject OffsetCurveDeformer::aEnvelope;

MStatus OffsetCurveDeformer::initialize() {
    MFnCompoundAttribute cAttr;
    MFnMatrixAttribute mAttr;
    MFnMessageAttribute meAttr;
    MFnTypedAttribute tAttr;
    MFnNumericAttribute nAttr;
    MStatus status;

    // 영향 곡선들
    aInfluenceCurves = tAttr.create("influenceCurves", "influenceCurves", MFnData::kNurbsCurve);
    tAttr.setArray(true);
    status = addAttribute(aInfluenceCurves);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aInfluenceCurves, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 오프셋 거리
    aOffsetDistance = nAttr.create("offsetDistance", "offsetDistance", MFnNumericData::kDouble, 1.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.001);
    nAttr.setMax(100.0);
    status = addAttribute(aOffsetDistance);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aOffsetDistance, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 영향 감쇠 반경
    aFalloffRadius = nAttr.create("falloffRadius", "falloffRadius", MFnNumericData::kDouble, 2.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.1);
    nAttr.setMax(100.0);
    status = addAttribute(aFalloffRadius);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aFalloffRadius, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 곡선 타입
    aCurveType = nAttr.create("curveType", "curveType", MFnNumericData::kInt, 0);
    nAttr.setKeyable(true);
    nAttr.setMin(0);
    nAttr.setMax(1);
    status = addAttribute(aCurveType);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aCurveType, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 바인딩 데이터 (복합 속성)
    aSamplePoints = tAttr.create("samplePoints", "samplePoints", MFnData::kPointArray);
    tAttr.setArray(true);

    aSampleWeights = tAttr.create("sampleWeights", "sampleWeights", MFnData::kDoubleArray);
    tAttr.setArray(true);

    aOffsetVectors = tAttr.create("offsetVectors", "offsetVectors", MFnData::kVectorArray);
    tAttr.setArray(true);

    aBindMatrices = mAttr.create("bindMatrices", "bindMatrices");
    mAttr.setArray(true);

    aBindData = cAttr.create("bindData", "bindData");
    cAttr.setArray(true);
    cAttr.addChild(aSamplePoints);
    cAttr.addChild(aSampleWeights);
    cAttr.addChild(aOffsetVectors);
    cAttr.addChild(aBindMatrices);
    status = addAttribute(aBindData);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 태스크 수
    aNumTasks = nAttr.create("numTasks", "numTasks", MFnNumericData::kInt, 32);
    nAttr.setMin(1);
    nAttr.setMax(64);
    status = addAttribute(aNumTasks);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aNumTasks, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 엔벨로프
    aEnvelope = nAttr.create("envelope", "envelope", MFnNumericData::kFloat, 1.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    status = addAttribute(aEnvelope);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aEnvelope, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 페인트 가능한 가중치 만들기
    status = MGlobal::executeCommandOnIdle("makePaintable -attrType multiFloat -sm deformer offsetCurveDeformer weights");
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return MS::kSuccess;
}

OffsetCurveDeformer::OffsetCurveDeformer() {
    MThreadPool::init();
    onDeleteCallbackId = 0;
}

OffsetCurveDeformer::~OffsetCurveDeformer() {
    if (onDeleteCallbackId != 0)
        MMessage::removeCallback(onDeleteCallbackId);
    
    MThreadPool::release();
    std::vector<ThreadData<OffsetCurveTaskData>*>::iterator iter;
    for (iter = threadData_.begin(); iter != threadData_.end(); ++iter) {
        delete [] *iter;
    }
    threadData_.clear();
}

void* OffsetCurveDeformer::creator() {
    return new OffsetCurveDeformer();
}

void OffsetCurveDeformer::postConstructor() {
    MPxDeformerNode::postConstructor();

    MStatus status = MS::kSuccess;
    MObject obj = thisMObject();
    onDeleteCallbackId = MNodeMessage::addNodeAboutToDeleteCallback(obj, aboutToDeleteCB, NULL, &status);
}

void OffsetCurveDeformer::aboutToDeleteCB(MObject &node, MDGModifier &modifier, void *clientData) {
    // 노드 삭제 시 정리 작업
}

MStatus OffsetCurveDeformer::setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs) {
    // 바인딩 데이터가 더티해지면 플래그 설정
    if (plugBeingDirtied.isElement()) {
        MPlug parent = plugBeingDirtied.array().parent();
        if (parent == aBindData) {
            unsigned int geomIndex = parent.logicalIndex();
            dirty_[geomIndex] = true;
        }
    }
    return MS::kSuccess;
}

MStatus OffsetCurveDeformer::deform(MDataBlock& data, MItGeometry& itGeo, const MMatrix& localToWorldMatrix,
                                   unsigned int geomIndex) {
    MStatus status;
    
    if (geomIndex >= taskData_.size()) {
        taskData_.resize(geomIndex + 1);
    }
    OffsetCurveTaskData& taskData = taskData_[geomIndex];
    
    // 영향 곡선 확인
    MDataHandle hInfluenceCurves = data.inputValue(aInfluenceCurves, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    MObject oInfluenceCurves = hInfluenceCurves.data();
    if (oInfluenceCurves.isNull()) {
        return MS::kSuccess; // 영향 곡선이 없으면 변형하지 않음
    }

    // 바인딩 데이터가 더티하거나 없으면 재계산
    if (dirty_[geomIndex] || taskData.samplePoints.length() == 0) {
        dirty_[geomIndex] = false;
        status = GetBindInfo(data, geomIndex, taskData);
        if (status == MS::kNotImplemented) {
            return MS::kSuccess; // 바인딩 정보가 아직 없음
        } else if (MFAIL(status)) {
            CHECK_MSTATUS_AND_RETURN_IT(status);
        }
    }

    // 현재 영향 곡선 상태 가져오기
    status = GetInfluenceCurveData(data, taskData);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 디포머 멤버십과 페인트 가중치 가져오기
    unsigned int membershipCount = itGeo.count();
    taskData.membership.setLength(membershipCount);
    taskData.paintWeights.setLength(membershipCount);
    status = itGeo.allPositions(taskData.points);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    for (int i = 0; !itGeo.isDone(); itGeo.next(), i++) {
        taskData.membership[i] = itGeo.index();
        taskData.paintWeights[i] = weightValue(data, geomIndex, itGeo.index());
    }
    
    taskData.drivenMatrix = localToWorldMatrix;
    taskData.drivenInverseMatrix = localToWorldMatrix.inverse();
    
    // 파라미터 가져오기
    taskData.offsetDistance = (float)data.inputValue(aOffsetDistance).asDouble();
    taskData.falloffRadius = (float)data.inputValue(aFalloffRadius).asDouble();
    taskData.envelope = data.inputValue(aEnvelope).asFloat();
    int taskCount = data.inputValue(aNumTasks).asInt();
    
    if (taskData.envelope == 0.0f || taskCount <= 0) {
        return MS::kSuccess;
    }

    // 스레드 데이터 준비
    if (geomIndex >= threadData_.size()) {
        size_t currentSize = threadData_.size();
        threadData_.resize(geomIndex + 1);
        for (size_t i = currentSize; i < geomIndex + 1; ++i) {
            threadData_[i] = new ThreadData<OffsetCurveTaskData>[taskCount];
        }
    } else {
        if (threadData_[geomIndex][0].numTasks != taskCount) {
            delete [] threadData_[geomIndex];
            threadData_[geomIndex] = new ThreadData<OffsetCurveTaskData>[taskCount];
        }
    }

    CreateThreadData<OffsetCurveTaskData>(taskCount, taskData_[geomIndex].points.length(),
                                         &taskData_[geomIndex], threadData_[geomIndex]);
    
    // 멀티스레딩으로 변형 계산
    MThreadPool::newParallelRegion(CreateTasks, (void *)threadData_[geomIndex]);

    // 변형된 포인트들을 지오메트리에 적용
    status = itGeo.setAllPositions(taskData.points);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return MS::kSuccess;
}

MStatus OffsetCurveDeformer::GetBindInfo(MDataBlock& data, unsigned int geomIndex, OffsetCurveTaskData& taskData) {
    MStatus status;
    
    MArrayDataHandle hBindDataArray = data.inputArrayValue(aBindData);
    status = hBindDataArray.jumpToElement(geomIndex);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    MDataHandle hBindData = hBindDataArray.inputValue();
    MArrayDataHandle hSamplePoints = hBindData.child(aSamplePoints);
    unsigned int numVerts = hSamplePoints.elementCount();
    
    if (numVerts == 0) {
        return MS::kNotImplemented; // 바인딩 정보가 아직 없음
    }
    
    MArrayDataHandle hSampleWeights = hBindData.child(aSampleWeights);
    MArrayDataHandle hOffsetVectors = hBindData.child(aOffsetVectors);
    MArrayDataHandle hBindMatrices = hBindData.child(aBindMatrices);

    // 배열 초기화
    taskData.samplePoints.setLength(numVerts);
    taskData.sampleWeights.setLength(numVerts);
    taskData.offsetVectors.setLength(numVerts);
    taskData.bindMatrices.setLength(numVerts);

    // 바인딩 데이터 읽기
    for (unsigned int i = 0; i < numVerts; ++i) {
        int logicalIndex = hSamplePoints.elementIndex();
        
        // 샘플 포인트
        MObject oPointData = hSamplePoints.inputValue().data();
        MFnPointArrayData fnPointData(oPointData, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        MPointArray points = fnPointData.array();
        if (points.length() > 0) {
            taskData.samplePoints[logicalIndex] = points[0];
        }
        
        // 샘플 가중치
        MObject oWeightData = hSampleWeights.inputValue().data();
        MFnDoubleArrayData fnWeightData(oWeightData, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        MDoubleArray weights = fnWeightData.array();
        if (weights.length() > 0) {
            taskData.sampleWeights[logicalIndex] = (float)weights[0];
        }
        
        // 오프셋 벡터
        MObject oVectorData = hOffsetVectors.inputValue().data();
        MFnVectorArrayData fnVectorData(oVectorData, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        MVectorArray vectors = fnVectorData.array();
        if (vectors.length() > 0) {
            taskData.offsetVectors[logicalIndex] = vectors[0];
        }
        
        // 바인딩 매트릭스
        taskData.bindMatrices[logicalIndex] = hBindMatrices.inputValue().asMatrix();

        // 다음 요소로 이동
        hSamplePoints.next();
        hSampleWeights.next();
        hOffsetVectors.next();
        hBindMatrices.next();
    }
    
    return MS::kSuccess;
}

MStatus OffsetCurveDeformer::GetInfluenceCurveData(MDataBlock& data, OffsetCurveTaskData& taskData) {
    MStatus status;
    
    // 영향 곡선 데이터 가져오기 (구현 예정)
    // 현재는 기본값 사용
    
    return MS::kSuccess;
}

void OffsetCurveDeformer::CreateTasks(void *data, MThreadRootTask *pRoot) {
    ThreadData<OffsetCurveTaskData>* threadData = static_cast<ThreadData<OffsetCurveTaskData>*>(data);

    if (threadData) {
        int numTasks = threadData[0].numTasks;
        for(int i = 0; i < numTasks; i++) {
            MThreadPool::createTask(EvaluateOffsetCurve, (void *)&threadData[i], pRoot);
        }
        MThreadPool::executeAndJoin(pRoot);
    }
}

MThreadRetVal OffsetCurveDeformer::EvaluateOffsetCurve(void *pParam) {
    ThreadData<OffsetCurveTaskData>* pThreadData = static_cast<ThreadData<OffsetCurveTaskData>*>(pParam);
    OffsetCurveTaskData* pData = pThreadData->pData;
    
    // 스레드 데이터 추출
    MMatrix& drivenMatrix = pData->drivenMatrix;
    MMatrix& drivenInverseMatrix = pData->drivenInverseMatrix;
    float env = pData->envelope;
    float offsetDistance = pData->offsetDistance;
    float falloffRadius = pData->falloffRadius;
    MIntArray& membership = pData->membership;
    MFloatArray& paintWeights = pData->paintWeights;
    MPointArray& points = pData->points;
    MPointArray& samplePoints = pData->samplePoints;
    MFloatArray& sampleWeights = pData->sampleWeights;
    MVectorArray& offsetVectors = pData->offsetVectors;
    MMatrixArray& bindMatrices = pData->bindMatrices;

    unsigned int taskStart = pThreadData->start;
    unsigned int taskEnd = pThreadData->end;

    // 각 포인트에 대해 변형 계산
    for (unsigned int i = taskStart; i < taskEnd; ++i) {
        if (i >= points.length()) {
            break;
        }
        
        int index = membership[i];
        if (index >= samplePoints.length()) {
            continue;
        }

        // 바인딩된 샘플 포인트와 가중치
        MPoint samplePoint = samplePoints[index];
        float sampleWeight = sampleWeights[index];
        MVector offsetVector = offsetVectors[index];
        MMatrix bindMatrix = bindMatrices[index];

        // 오프셋 거리에 따른 변형 계산
        MPoint originalPoint = points[i];
        MPoint deformedPoint = originalPoint;
        
        if (sampleWeight > 0.0f) {
            // 바인딩 매트릭스를 사용한 변형
            MPoint transformedPoint = originalPoint * bindMatrix;
            
            // 오프셋 벡터 적용
            MVector currentOffset = offsetVector * offsetDistance;
            deformedPoint = transformedPoint + currentOffset;
            
            // 엔벨로프와 페인트 가중치 적용
            float finalWeight = sampleWeight * paintWeights[i] * env;
            points[i] = originalPoint + (deformedPoint - originalPoint) * finalWeight;
        }
    }
    
    return 0;
}

#if MAYA_API_VERSION >= 201600

MString OffsetCurveGPU::pluginLoadPath;

#if MAYA_API_VERSION >= 201650
cl_command_queue (*getMayaDefaultOpenCLCommandQueue)() = MOpenCLInfo::getMayaDefaultOpenCLCommandQueue;
#else
cl_command_queue (*getMayaDefaultOpenCLCommandQueue)() = MOpenCLInfo::getOpenCLCommandQueue;
#endif

MGPUDeformerRegistrationInfo* OffsetCurveGPU::GetGPUDeformerInfo() {
    static OffsetCurveGPUDeformerInfo ocdInfo;
    return &ocdInfo;
}

OffsetCurveGPU::OffsetCurveGPU() {
    // 생성자는 빠르게 유지
}

OffsetCurveGPU::~OffsetCurveGPU() {
    terminate();
}

#if MAYA_API_VERSION <= 201700
MPxGPUDeformer::DeformerStatus OffsetCurveGPU::evaluate(MDataBlock& block,
                                                       const MEvaluationNode& evaluationNode,
                                                       const MPlug& plug,
                                                       unsigned int numElements,
                                                       const MAutoCLMem inputBuffer,
                                                       const MAutoCLEvent inputEvent,
                                                       MAutoCLMem outputBuffer,
                                                       MAutoCLEvent& outputEvent) {
#else
MPxGPUDeformer::DeformerStatus OffsetCurveGPU::evaluate(MDataBlock& block,
                                                        const MEvaluationNode& evaluationNode,
                                                        const MPlug& plug,
                                                        const MGPUDeformerData& inputData,
                                                        MGPUDeformerData& outputData) {
    // GPU 데이터 가져오기
    MGPUDeformerBuffer inputDeformerBuffer = inputData.getBuffer(sPositionsName());
    const MAutoCLMem inputBuffer = inputDeformerBuffer.buffer();
    unsigned int numElements = inputDeformerBuffer.elementCount();
    const MAutoCLEvent inputEvent = inputDeformerBuffer.bufferReadyEvent();

    // 출력 버퍼 생성
    MGPUDeformerBuffer outputDeformerBuffer = createOutputBuffer(inputDeformerBuffer);
    MAutoCLEvent outputEvent;
    MAutoCLMem outputBuffer = outputDeformerBuffer.buffer();
#endif

    MStatus status;
    numElements_ = numElements;
    
    // GPU에 필요한 데이터 전송
    status = EnqueueBindData(block, evaluationNode, plug);
    CHECK_MSTATUS(status);
    status = EnqueueInfluenceCurves(block, evaluationNode, plug);
    CHECK_MSTATUS(status);
    status = EnqueueOffsetData(block, evaluationNode, plug);
    CHECK_MSTATUS(status);

    // OpenCL 커널 로드 (구현 예정)
    if (!kernel_.get()) {
        // 커널 로드 로직
        MGlobal::displayWarning("OpenCL kernel not yet implemented for OCD");
        return MPxGPUDeformer::kDeformerFailure;
    }

    // GPU 변형 계산 (구현 예정)
    // 현재는 CPU 폴백 사용
    
    return MPxGPUDeformer::kDeformerSuccess;
}

MStatus OffsetCurveGPU::EnqueueBindData(MDataBlock& data, const MEvaluationNode& evaluationNode, const MPlug& plug) {
    // 바인딩 데이터를 GPU에 전송 (구현 예정)
    return MS::kSuccess;
}

MStatus OffsetCurveGPU::EnqueueInfluenceCurves(MDataBlock& data, const MEvaluationNode& evaluationNode, const MPlug& plug) {
    // 영향 곡선 데이터를 GPU에 전송 (구현 예정)
    return MS::kSuccess;
}

MStatus OffsetCurveGPU::EnqueueOffsetData(MDataBlock& data, const MEvaluationNode& evaluationNode, const MPlug& plug) {
    // 오프셋 데이터를 GPU에 전송 (구현 예정)
    return MS::kSuccess;
}

void OffsetCurveGPU::terminate() {
    // GPU 리소스 정리
    influenceCurves_.reset();
    offsetVectors_.reset();
    bindMatrices_.reset();
    samplePoints_.reset();
    sampleWeights_.reset();
    offsetDistances_.reset();
    falloffRadii_.reset();
    
    MOpenCLInfo::releaseOpenCLKernel(kernel_);
    kernel_.reset();
}

#endif // End Maya 2016

// GPU 디포머 등록 정보
#if MAYA_API_VERSION >= 201600
OffsetCurveGPUDeformerInfo::OffsetCurveGPUDeformerInfo() {
}

OffsetCurveGPUDeformerInfo::~OffsetCurveGPUDeformerInfo() {
}

MPxGPUDeformer* OffsetCurveGPUDeformerInfo::createGPUDeformer() {
    return new OffsetCurveGPU();
}

#if MAYA_API_VERSION >= 201650
bool OffsetCurveGPUDeformerInfo::validateNodeInGraph(MDataBlock& block, const MEvaluationNode& evaluationNode,
                                                     const MPlug& plug, MStringArray* messages) {
    return true;
}

bool OffsetCurveGPUDeformerInfo::validateNodeValues(MDataBlock& block, const MEvaluationNode& evaluationNode,
                                                    const MPlug& plug, MStringArray* messages) {
    return true;
}
#else
bool OffsetCurveGPUDeformerInfo::validateNode(MDataBlock& block, const MEvaluationNode& evaluationNode,
                                              const MPlug& plug, MStringArray* messages) {
    return true;
}
#endif

#endif // End Maya 2016
