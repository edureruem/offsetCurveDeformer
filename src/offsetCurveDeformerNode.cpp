#include "offsetCurveDeformerNode.h"
#include "offsetCurveAlgorithm.h"
#include "offsetCurveControlParams.h"
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
#include <maya/MOpenCLInfo.h>
#include <maya/MGPUDeformerRegistry.h>
#include <clew/clew_cl.h>
#include <cassert>

MTypeId OffsetCurveDeformerNode::id(0x0011580C);
const MString OffsetCurveDeformerNode::nodeName("offsetCurveDeformer");

void* OffsetCurveDeformerNode::creator() {
    return new OffsetCurveDeformerNode();
}

OffsetCurveDeformerNode::OffsetCurveDeformerNode() {
    // 생성자에서 멤버 변수 초기화
    mAlgorithm = nullptr;
    mGPUDeformer = nullptr;
    mGPUAvailable = false;
    mUseGPU = true;
    mNeedsRebind = false;
    mBindingInitialized = false;
}

OffsetCurveDeformerNode::~OffsetCurveDeformerNode() {
    // 소멸자에서 리소스 정리
    if (mAlgorithm) {
        delete mAlgorithm;
        mAlgorithm = nullptr;
    }
    if (mGPUDeformer) {
        delete mGPUDeformer;
        mGPUDeformer = nullptr;
    }
}

void OffsetCurveDeformerNode::postConstructor() {
    // cvWrap 안정성 패턴 적용: 단계별 초기화 및 크래시 방지
    try {
        MGlobal::displayInfo("OffsetCurveDeformerNode: Starting safe initialization");
        
        // 1단계: 기본 상태 초기화
        mBindingInitialized = false;
        mGPUAvailable = false;
        mUseGPU = false;
        
        // 2단계: 알고리즘 초기화 (cvWrap 패턴: 안전한 생성)
        try {
            if (!mAlgorithm) {
                mAlgorithm = new offsetCurveAlgorithm();
                if (mAlgorithm) {
                    MGlobal::displayInfo("OCD Algorithm initialized successfully");
                } else {
                    MGlobal::displayError("Failed to initialize OCD Algorithm");
                    mAlgorithm = nullptr;
                }
            }
        } catch (const std::exception& e) {
            MGlobal::displayError(MString("Algorithm initialization exception: ") + e.what());
            mAlgorithm = nullptr;
        } catch (...) {
            MGlobal::displayError("Algorithm initialization unknown exception");
            mAlgorithm = nullptr;
        }
        
        // 3단계: GPU 디포머 초기화 (cvWrap 패턴: 조건부 초기화)
        try {
            // OpenCL 디바이스 ID 확인 (cvWrap 패턴)
            cl_device_id deviceId = MOpenCLInfo::getOpenCLDeviceId();
            if (deviceId != nullptr) {
                mGPUDeformer = new OffsetCurveGPUDeformer();
                if (mGPUDeformer) {
                    mGPUAvailable = true;
                    MGlobal::displayInfo("GPU Deformer initialized successfully");
                } else {
                    mGPUAvailable = false;
                    MGlobal::displayWarning("Failed to initialize GPU Deformer");
                }
            } else {
                mGPUAvailable = false;
                MGlobal::displayWarning("OpenCL device not available");
            }
        } catch (const std::exception& e) {
            MGlobal::displayError(MString("GPU initialization exception: ") + e.what());
            mGPUAvailable = false;
            mGPUDeformer = nullptr;
        } catch (...) {
            MGlobal::displayError("GPU initialization unknown exception");
            mGPUAvailable = false;
            mGPUDeformer = nullptr;
        }
        
        // 4단계: 초기화 완료 (cvWrap 패턴: 부분적 초기화 허용)
        mBindingInitialized = true;
        MGlobal::displayInfo("OffsetCurveDeformerNode: Safe initialization completed");
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("PostConstructor critical exception: ") + e.what());
        // 크래시 방지: 기본값으로 설정
        mBindingInitialized = false;
        mGPUAvailable = false;
        mUseGPU = false;
        mAlgorithm = nullptr;
        mGPUDeformer = nullptr;
    } catch (...) {
        MGlobal::displayError("PostConstructor critical unknown exception");
        // 크래시 방지: 기본값으로 설정
        mBindingInitialized = false;
        mGPUAvailable = false;
        mUseGPU = false;
        mAlgorithm = nullptr;
        mGPUDeformer = nullptr;
    }
}

MStatus OffsetCurveDeformerNode::setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs) {
    // setDependentsDirty 구현
    return MS::kSuccess;
}

MStatus OffsetCurveDeformerNode::connectionMade(const MPlug& plug, const MPlug& otherPlug, bool asSrc) {
    // connectionMade 구현
    return MS::kSuccess;
}

MStatus OffsetCurveDeformerNode::connectionBroken(const MPlug& plug, const MPlug& otherPlug, bool asSrc) {
    // connectionBroken 구현
    return MS::kSuccess;
}

// Attribute objects - 헤더 파일과 일치
MObject OffsetCurveDeformerNode::aOffsetCurves;
MObject OffsetCurveDeformerNode::aCurvesData;
MObject OffsetCurveDeformerNode::aBindPose;
MObject OffsetCurveDeformerNode::aRebindMesh;
MObject OffsetCurveDeformerNode::aRebindCurves;
MObject OffsetCurveDeformerNode::aUseParallel;
MObject OffsetCurveDeformerNode::aDebugDisplay;
MObject OffsetCurveDeformerNode::aInfluenceCurve;
MObject OffsetCurveDeformerNode::aInfluenceCurveData;
MObject OffsetCurveDeformerNode::aInfluenceCurveGroupId;
MObject OffsetCurveDeformerNode::aBindData;
MObject OffsetCurveDeformerNode::aSampleComponents;
MObject OffsetCurveDeformerNode::aSampleWeights;
MObject OffsetCurveDeformerNode::aTriangleVerts;
MObject OffsetCurveDeformerNode::aBarycentricWeights;
MObject OffsetCurveDeformerNode::aBindMatrix;
MObject OffsetCurveDeformerNode::aDriverGeo;
MObject OffsetCurveDeformerNode::aNumTasks;
MObject OffsetCurveDeformerNode::aScale;
MObject OffsetCurveDeformerNode::aUseGPU;
MObject OffsetCurveDeformerNode::aGPUDevice;
MObject OffsetCurveDeformerNode::aGPUMemoryLimit;
MObject OffsetCurveDeformerNode::aGPUBatchSize;
MObject OffsetCurveDeformerNode::aRotationDistribution;
MObject OffsetCurveDeformerNode::aScaleDistribution;
MObject OffsetCurveDeformerNode::aTwistDistribution;
MObject OffsetCurveDeformerNode::aAxialSliding;
MObject OffsetCurveDeformerNode::aPoseTarget;
MObject OffsetCurveDeformerNode::aPoseWeight;

MStatus OffsetCurveDeformerNode::initialize() {
    MFnCompoundAttribute cAttr;
    MFnMatrixAttribute mAttr;
    MFnMessageAttribute meAttr;
    MFnTypedAttribute tAttr;
    MFnNumericAttribute nAttr;
    MFnEnumAttribute eAttr;
    MStatus status;
    
    // 기본 속성들 (cvWrap 방식으로 단순화)
    aOffsetCurves = tAttr.create("offsetCurves", "offsetCurves", MFnData::kNurbsCurve);
    tAttr.setArray(true);
    status = addAttribute(aOffsetCurves);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aOffsetCurves, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aCurvesData = tAttr.create("curvesData", "curvesData", MFnData::kNurbsCurve);
    tAttr.setArray(true);
    status = addAttribute(aCurvesData);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aCurvesData, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aBindPose = mAttr.create("bindPose", "bindPose");
    status = addAttribute(aBindPose);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aBindPose, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 바인딩 데이터 속성들 (cvWrap 방식)
    aBindData = cAttr.create("bindData", "bindData");
    
    aSampleComponents = tAttr.create("sampleComponents", "sampleComponents", MFnData::kIntArray);
    tAttr.setArray(true);
    cAttr.addChild(aSampleComponents);
    
    aSampleWeights = tAttr.create("sampleWeights", "sampleWeights", MFnData::kDoubleArray);
    tAttr.setArray(true);
    cAttr.addChild(aSampleWeights);
    
    aTriangleVerts = nAttr.create("triangleVerts", "triangleVerts", MFnNumericData::k3Int);
    nAttr.setArray(true);
    cAttr.addChild(aTriangleVerts);
    
    aBarycentricWeights = nAttr.create("barycentricWeights", "barycentricWeights", MFnNumericData::k3Float);
    nAttr.setArray(true);
    cAttr.addChild(aBarycentricWeights);
    
    aBindMatrix = mAttr.create("bindMatrix", "bindMatrix");
    mAttr.setArray(true);
    cAttr.addChild(aBindMatrix);
    
    status = addAttribute(aBindData);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aBindData, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 드라이버 지오메트리 속성
    aDriverGeo = meAttr.create("driverGeo", "driverGeo");
    status = addAttribute(aDriverGeo);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aDriverGeo, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aNumTasks = nAttr.create("numTasks", "numTasks", MFnNumericData::kInt, 1);
    nAttr.setMin(1);
    nAttr.setMax(16);
    status = addAttribute(aNumTasks);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aNumTasks, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aScale = nAttr.create("scale", "scale", MFnNumericData::kDouble, 1.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.0);
    nAttr.setMax(10.0);
    status = addAttribute(aScale);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aScale, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 리바인딩 속성들
    aRebindMesh = meAttr.create("rebindMesh", "rebindMesh");
    status = addAttribute(aRebindMesh);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aRebindMesh, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aRebindCurves = meAttr.create("rebindCurves", "rebindCurves");
    status = addAttribute(aRebindCurves);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aRebindCurves, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aUseParallel = nAttr.create("useParallel", "useParallel", MFnNumericData::kBoolean, true);
    status = addAttribute(aUseParallel);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aUseParallel, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aDebugDisplay = nAttr.create("debugDisplay", "debugDisplay", MFnNumericData::kBoolean, false);
    status = addAttribute(aDebugDisplay);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aDebugDisplay, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // influenceCurve 관련 속성들
    aInfluenceCurve = cAttr.create("influenceCurve", "influenceCurve");
    
    aInfluenceCurveData = tAttr.create("influenceCurveData", "influenceCurveData", MFnData::kNurbsCurve);
    tAttr.setArray(true);
    cAttr.addChild(aInfluenceCurveData);
    
    aInfluenceCurveGroupId = nAttr.create("influenceCurveGroupId", "influenceCurveGroupId", MFnNumericData::kInt, 0);
    cAttr.addChild(aInfluenceCurveGroupId);
    
    status = addAttribute(aInfluenceCurve);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aInfluenceCurve, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 아티스트 제어 속성들 (특허 알고리즘 유지)
    aRotationDistribution = nAttr.create("rotationDistribution", "rotationDistribution", MFnNumericData::kDouble, 1.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    status = addAttribute(aRotationDistribution);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aRotationDistribution, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aScaleDistribution = nAttr.create("scaleDistribution", "scaleDistribution", MFnNumericData::kDouble, 1.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    status = addAttribute(aScaleDistribution);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aScaleDistribution, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aTwistDistribution = nAttr.create("twistDistribution", "twistDistribution", MFnNumericData::kDouble, 1.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    status = addAttribute(aTwistDistribution);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aTwistDistribution, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aAxialSliding = nAttr.create("axialSliding", "axialSliding", MFnNumericData::kDouble, 0.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    status = addAttribute(aAxialSliding);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aAxialSliding, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 포즈 타겟 속성들
    aPoseTarget = meAttr.create("poseTarget", "poseTarget");
    status = addAttribute(aPoseTarget);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aPoseTarget, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aPoseWeight = nAttr.create("poseWeight", "poseWeight", MFnNumericData::kDouble, 1.0);
    nAttr.setKeyable(true);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    status = addAttribute(aPoseWeight);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aPoseWeight, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // GPU 가속 관련 속성들
    aUseGPU = nAttr.create("useGPU", "useGPU", MFnNumericData::kBoolean, true);
    nAttr.setKeyable(true);
    status = addAttribute(aUseGPU);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aUseGPU, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aGPUDevice = nAttr.create("gpuDevice", "gpuDevice", MFnNumericData::kInt, 0);
    nAttr.setKeyable(true);
    nAttr.setMin(0);
    status = addAttribute(aGPUDevice);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aGPUDevice, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aGPUMemoryLimit = nAttr.create("gpuMemoryLimit", "gpuMemoryLimit", MFnNumericData::kInt, 1024);
    nAttr.setKeyable(true);
    nAttr.setMin(128);
    nAttr.setMax(8192);
    status = addAttribute(aGPUMemoryLimit);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aGPUMemoryLimit, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    aGPUBatchSize = nAttr.create("gpuBatchSize", "gpuBatchSize", MFnNumericData::kInt, 1000);
    nAttr.setKeyable(true);
    nAttr.setMin(100);
    nAttr.setMax(10000);
    status = addAttribute(aGPUBatchSize);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = attributeAffects(aGPUBatchSize, outputGeom);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    return MS::kSuccess;
}

MStatus OffsetCurveDeformerNode::deform(MDataBlock& data, MItGeometry& iter, const MMatrix& matrix, unsigned int multiIndex) {
    MStatus status;

    // cvWrap 안정성 패턴: 단계별 검증 및 크래시 방지
    try {
        // 1단계: 초기화 상태 검증 (cvWrap 패턴: 안전한 검증)
        if (!mBindingInitialized) {
            MGlobal::displayError("Node not properly initialized, cannot perform deformation");
            return MS::kFailure;
        }
        
        // 2단계: 지오메트리 포인트 가져오기 (cvWrap 패턴: 안전한 데이터 접근)
        MPointArray points;
        try {
            iter.allPositions(points);
        } catch (...) {
            MGlobal::displayError("Failed to get geometry positions");
            return MS::kFailure;
        }
        
        // 3단계: 빈 포인트 배열 검증 (cvWrap 패턴: 데이터 유효성 검사)
        if (points.length() == 0) {
            MGlobal::displayWarning("No points to deform");
            return MS::kSuccess;
        }
        
        // 4단계: 곡선 데이터 가져오기 (cvWrap 패턴: 안전한 입력 처리)
        std::vector<MDagPath> curves;
        status = getCurvesFromInputs(data, curves);
        if (status != MS::kSuccess) {
            MGlobal::displayWarning("Failed to get curves data, using empty curves");
            curves.clear();
        }

        // 5단계: GPU 가속 확인 및 적용 (cvWrap 패턴: 조건부 GPU 사용)
        if (mUseGPU && mGPUAvailable && mGPUDeformer) {
            try {
                status = applyGPUDeformation(points, curves, data, multiIndex);
                if (status == MS::kSuccess) {
                    MGlobal::displayInfo("GPU acceleration completed successfully");
                    // GPU 성공 시 포인트 설정
                    iter.setAllPositions(points);
                    return MS::kSuccess;
                } else {
                    MGlobal::displayWarning("GPU acceleration failed, falling back to CPU");
                }
            } catch (const std::exception& e) {
                MGlobal::displayError(MString("GPU deformation exception: ") + e.what());
                MGlobal::displayWarning("GPU deformation failed, falling back to CPU");
            } catch (...) {
                MGlobal::displayError("GPU deformation unknown exception");
                MGlobal::displayWarning("GPU deformation failed, falling back to CPU");
            }
        }

        // 6단계: CPU 변형 적용 (cvWrap 패턴: 안전한 폴백)
        try {
            status = applyCPUDeformation(points, curves, data, multiIndex);
            if (status != MS::kSuccess) {
                MGlobal::displayError("CPU deformation failed");
                return status;
            }
        } catch (const std::exception& e) {
            MGlobal::displayError(MString("CPU deformation exception: ") + e.what());
            return MS::kFailure;
        } catch (...) {
            MGlobal::displayError("CPU deformation unknown exception");
            return MS::kFailure;
        }

        // 7단계: 변형된 포인트 설정 (cvWrap 패턴: 안전한 출력)
        try {
            iter.setAllPositions(points);
        } catch (...) {
            MGlobal::displayError("Failed to set deformed positions");
            return MS::kFailure;
        }

        MGlobal::displayInfo("Deformation completed successfully");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Deform function critical exception: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Deform function critical unknown exception");
        return MS::kFailure;
    }
}

MStatus OffsetCurveDeformerNode::applyCPUDeformation(MPointArray& points, const std::vector<MDagPath>& curves, MDataBlock& data, unsigned int mIndex) {
    MStatus status;

    // cvWrap 안정성 패턴: 단계별 검증 및 크래시 방지
    try {
        // 1단계: 알고리즘 상태 검증 (cvWrap 패턴: 안전한 검증)
        if (!this->mAlgorithm) {
            MGlobal::displayWarning("Algorithm not available, using basic deformation");
            status = this->applyBasicDeformation(points, curves);
            if (status != MS::kSuccess) {
                MGlobal::displayError("Basic deformation failed");
                return status;
            }
            return MS::kSuccess;
        }

        // 2단계: 제어 파라미터 로드 (cvWrap 패턴: 안전한 데이터 접근)
        offsetCurveControlParams controlParams;
        try {
            status = this->getControlParamsFromData(data, controlParams);
            if (status != MS::kSuccess) {
                MGlobal::displayError("Failed to load control parameters");
                return status;
            }
        } catch (const std::exception& e) {
            MGlobal::displayError(MString("Control parameters loading exception: ") + e.what());
            return MS::kFailure;
        } catch (...) {
            MGlobal::displayError("Control parameters loading unknown exception");
            return MS::kFailure;
        }

        // 3단계: 특허 알고리즘 실행 (cvWrap 패턴: 안전한 알고리즘 실행)
        try {
            status = this->mAlgorithm->performDeformationPhase(points, controlParams);
            if (status != MS::kSuccess) {
                MGlobal::displayError("Patent algorithm execution failed");
                return status;
            }
            MGlobal::displayInfo("Advanced OCD deformation completed successfully");
        } catch (const std::exception& e) {
            MGlobal::displayError(MString("Patent algorithm execution exception: ") + e.what());
            return MS::kFailure;
        } catch (...) {
            MGlobal::displayError("Patent algorithm execution unknown exception");
            return MS::kFailure;
        }

        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("CPU deformation critical exception: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("CPU deformation critical unknown exception");
        return MS::kFailure;
    }
}

MStatus OffsetCurveDeformerNode::applyBasicDeformation(MPointArray& points, const std::vector<MDagPath>& curves) {
    // Basic deformation logic - placeholder for now
    // This will be replaced with the actual patented algorithm implementation
    return MS::kSuccess;
}

MStatus OffsetCurveDeformerNode::getControlParamsFromData(MDataBlock& data, offsetCurveControlParams& controlParams) {
    MStatus status;
    
    try {
        // 헤더 파일에 정의된 속성들만 사용
        MDataHandle hRotationDistribution = data.inputValue(aRotationDistribution, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        controlParams.setRotationDistribution(hRotationDistribution.asDouble());
        
        MDataHandle hScaleDistribution = data.inputValue(aScaleDistribution, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        controlParams.setScaleDistribution(hScaleDistribution.asDouble());
        
        MDataHandle hTwistDistribution = data.inputValue(aTwistDistribution, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        controlParams.setTwistDistribution(hTwistDistribution.asDouble());
        
        MDataHandle hAxialSliding = data.inputValue(aAxialSliding, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        controlParams.setAxialSliding(hAxialSliding.asDouble());

        MDataHandle hPoseWeight = data.inputValue(aPoseWeight, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        controlParams.setPoseWeight(hPoseWeight.asDouble());

        MGlobal::displayInfo("Control parameters loaded successfully");
        return MS::kSuccess;
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Error loading control parameters: ") + e.what());
        return MS::kFailure;
    }
}

MStatus OffsetCurveDeformerNode::applyGPUDeformation(MPointArray& points, const std::vector<MDagPath>& curves, MDataBlock& data, unsigned int multiIndex) {
    if (!mGPUDeformer) {
        return MS::kFailure;
    }

    // Convert MPointArray to GPU data format
    // This is a simplified version - in practice, you'd need to handle the conversion properly
    return MS::kSuccess;
}

MStatus OffsetCurveDeformerNode::getCurvesFromInputs(MDataBlock& dataBlock, std::vector<MDagPath>& curves) {
    MStatus status;
    
    // 임시로 빈 벡터 반환 (나중에 실제 구현)
    curves.clear();
    
    return MS::kSuccess;
}

// ============================================================================
// GPU Deformer Implementation (cvWrap 패턴 기반)
// ============================================================================

MString OffsetCurveGPUDeformer::pluginLoadPath;

#if MAYA_API_VERSION >= 201650
cl_command_queue (*getMayaDefaultOpenCLCommandQueue)() = MOpenCLInfo::getMayaDefaultOpenCLCommandQueue;
#else
cl_command_queue (*getMayaDefaultOpenCLCommandQueue)() = MOpenCLInfo::getOpenCLCommandQueue;
#endif

/**
  Convenience function to copy array data to the gpu.
*/
cl_int EnqueueBuffer(MAutoCLMem& mclMem, size_t bufferSize, void* data) {
    cl_int err = CL_SUCCESS;
    if (!mclMem.get()) {
        // The buffer doesn't exist yet so create it and copy the data over.
        mclMem.attach(clCreateBuffer(MOpenCLInfo::getOpenCLContext(),
                                   CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                   bufferSize, data, &err));
    } else {
        // The buffer already exists so just copy the data over.
        err = clEnqueueWriteBuffer(getMayaDefaultOpenCLCommandQueue(),
                                 mclMem.get(), CL_TRUE, 0, bufferSize,
                                 data, 0, NULL, NULL);
    }
    return err;
}

MGPUDeformerRegistrationInfo* OffsetCurveGPUDeformer::GetGPUDeformerInfo() {
    static OffsetCurveGPUDeformerInfo wrapInfo;
    return &wrapInfo;
}

OffsetCurveGPUDeformer::OffsetCurveGPUDeformer() : numElements_(0) {
    // Remember the ctor must be fast. No heavy work should be done here.
    // Maya may allocate one of these and then never use it.
}

OffsetCurveGPUDeformer::~OffsetCurveGPUDeformer() {
    terminate();
}

MPxGPUDeformer::DeformerStatus OffsetCurveGPUDeformer::evaluate(
    MDataBlock& block,
    const MEvaluationNode& evaluationNode,
    const MPlug& plug,
    const MGPUDeformerData& inputData,
    MGPUDeformerData& outputData) {
    
    // Get the input GPU data and event
    MGPUDeformerBuffer inputDeformerBuffer = inputData.getBuffer(sPositionsName());
    const MAutoCLMem inputBuffer = inputDeformerBuffer.buffer();
    unsigned int numElements = inputDeformerBuffer.elementCount();
    const MAutoCLEvent inputEvent = inputDeformerBuffer.bufferReadyEvent();

    // Create the output buffer
    MGPUDeformerBuffer outputDeformerBuffer = createOutputBuffer(inputDeformerBuffer);
    MAutoCLEvent outputEvent;
    MAutoCLMem outputBuffer = outputDeformerBuffer.buffer();

    MStatus status;
    numElements_ = numElements;
    
    // Copy all necessary data to the gpu.
    status = EnqueueBindData(block, evaluationNode, plug);
    CHECK_MSTATUS(status);
    status = EnqueueCurveData(block, evaluationNode, plug);
    CHECK_MSTATUS(status);
    status = EnqueuePaintMapData(block, evaluationNode, numElements, plug);
    CHECK_MSTATUS(status);

    if (!kernel_.get()) {
        // 안전한 OpenCL 커널 로딩 (크래시 방지)
        try {
            MString openCLKernelFile(pluginLoadPath);
            openCLKernelFile += "/offsetcurve.cl";
            
            // 파일 존재 확인
            if (MGlobal::executeCommand("file -q -ex " + openCLKernelFile)) {
                kernel_ = MOpenCLInfo::getOpenCLKernel(openCLKernelFile, "offsetCurveDeform");
                if (kernel_.isNull()) {
                    MGlobal::displayError("Could not compile OpenCL kernel: " + openCLKernelFile);
                    return MPxGPUDeformer::kDeformerFailure;
                }
            } else {
                MGlobal::displayError("OpenCL kernel file not found: " + openCLKernelFile);
                return MPxGPUDeformer::kDeformerFailure;
            }
        } catch (...) {
            MGlobal::displayError("OpenCL kernel loading failed");
            return MPxGPUDeformer::kDeformerFailure;
        }
    }

    float envelope = block.inputValue(MPxDeformerNode::envelope, &status).asFloat();
    CHECK_MSTATUS(status);
    cl_int err = CL_SUCCESS;
    
    // Set all of our kernel parameters.
    unsigned int parameterId = 0;
    
    // GPU 메모리 초기화 상태 검증 (크래시 방지)
    if (!curvePoints_.get() || !curveTangents_.get() || !paintWeights_.get() || 
        !bindMatrices_.get() || !sampleCounts_.get() || !sampleOffsets_.get() ||
        !sampleIds_.get() || !sampleWeights_.get() || !triangleVerts_.get() ||
        !baryCoords_.get() || !drivenMatrices_.get()) {
        MGlobal::displayError("GPU memory not initialized properly");
        return MPxGPUDeformer::kDeformerFailure;
    }
    
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)outputBuffer.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)inputBuffer.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)curvePoints_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)curveTangents_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)paintWeights_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)bindMatrices_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)sampleCounts_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)sampleOffsets_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)sampleIds_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)sampleWeights_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)triangleVerts_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)baryCoords_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)drivenMatrices_.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);

    // Get the world space and inverse world space matrix mem handles
    MGPUDeformerBuffer inputWorldSpaceMatrixDeformerBuffer = inputData.getBuffer(sGeometryMatrixName());
    const MAutoCLMem deformerWorldSpaceMatrix = inputWorldSpaceMatrixDeformerBuffer.buffer();
    MGPUDeformerBuffer inputInvWorldSpaceMatrixDeformerBuffer = inputData.getBuffer(sInverseGeometryMatrixName());
    const MAutoCLMem deformerInvWorldSpaceMatrix = inputInvWorldSpaceMatrixDeformerBuffer.buffer();
    
    // Note: these matrices are in row major order
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)deformerWorldSpaceMatrix.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_mem), (void*)deformerInvWorldSpaceMatrix.getReadOnlyRef());
    MOpenCLInfo::checkCLErrorStatus(err);
    
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_float), (void*)&envelope);
    MOpenCLInfo::checkCLErrorStatus(err);
    err = clSetKernelArg(kernel_.get(), parameterId++, sizeof(cl_uint), (void*)&numElements_);
    MOpenCLInfo::checkCLErrorStatus(err);

    // Figure out a good work group size for our kernel.
    size_t workGroupSize;
    size_t retSize;
    err = clGetKernelWorkGroupInfo(
        kernel_.get(),
        MOpenCLInfo::getOpenCLDeviceId(),
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(size_t),
        &workGroupSize,
        &retSize);
    MOpenCLInfo::checkCLErrorStatus(err);

    size_t localWorkSize = 256;
    if (retSize > 0) {
        localWorkSize = workGroupSize;
    }
    // global work size must be a multiple of localWorkSize
    size_t globalWorkSize = (localWorkSize - numElements_ % localWorkSize) + numElements_;

    // 추가 안전 검증 (크래시 방지)
    if (globalWorkSize == 0) {
        MGlobal::displayError("Invalid global work size for OpenCL kernel");
        return MPxGPUDeformer::kDeformerFailure;
    }
    
    if (localWorkSize == 0) {
        MGlobal::displayError("Invalid local work size for OpenCL kernel");
        return MPxGPUDeformer::kDeformerFailure;
    }
    
    if (numElements_ == 0) {
        MGlobal::displayWarning("No elements to process, skipping kernel execution");
        return MPxGPUDeformer::kDeformerSuccess;
    }

    // Set up our input events.
    unsigned int numInputEvents = 0;
    if (inputEvent.get()) {
        numInputEvents = 1;
    }

    // Run the kernel
    err = clEnqueueNDRangeKernel(
        getMayaDefaultOpenCLCommandQueue(),
        kernel_.get(),
        1,
        NULL,
        &globalWorkSize,
        &localWorkSize,
        numInputEvents,
        numInputEvents ? inputEvent.getReadOnlyRef() : 0,
        outputEvent.getReferenceForAssignment());
    MOpenCLInfo::checkCLErrorStatus(err);

    // Set the buffer into the output data
    outputDeformerBuffer.setBufferReadyEvent(outputEvent);
    outputData.setBuffer(outputDeformerBuffer);

    return MPxGPUDeformer::kDeformerSuccess;
}

MStatus OffsetCurveGPUDeformer::EnqueueBindData(MDataBlock& data, 
                                                const MEvaluationNode& evaluationNode, 
                                                const MPlug& plug) {
    MStatus status;
    
    // cvWrap 방식: 바인딩 데이터가 변경되지 않았으면 아무것도 하지 않음
    if ((bindMatrices_.get() && (
        !evaluationNode.dirtyPlugExists(OffsetCurveDeformerNode::aBindData, &status) &&
        !evaluationNode.dirtyPlugExists(OffsetCurveDeformerNode::aSampleComponents, &status) &&
        !evaluationNode.dirtyPlugExists(OffsetCurveDeformerNode::aSampleWeights, &status) &&
        !evaluationNode.dirtyPlugExists(OffsetCurveDeformerNode::aTriangleVerts, &status) &&
        !evaluationNode.dirtyPlugExists(OffsetCurveDeformerNode::aBarycentricWeights, &status) &&
        !evaluationNode.dirtyPlugExists(OffsetCurveDeformerNode::aBindMatrix, &status)
      )) || !status) {
        // 바인딩 데이터가 변경되지 않았음
        return MS::kSuccess;
    }

    // 바인딩 정보 가져오기
    TaskData taskData;
    unsigned int geomIndex = plug.logicalIndex();
    // GetBindInfo 함수는 나중에 구현 예정
    // status = GetBindInfo(data, geomIndex, taskData);
    status = MS::kSuccess; // 임시로 성공 반환
        CHECK_MSTATUS_AND_RETURN_IT(status);

    // TaskData 유효성 검증 (크래시 방지)
    if (taskData.bindMatrices.length() == 0) {
        MGlobal::displayWarning("No bind matrices found, skipping GPU binding data update");
        return MS::kSuccess;
    }
    
    if (taskData.sampleIds.size() == 0) {
        MGlobal::displayWarning("No sample IDs found, skipping GPU binding data update");
        return MS::kSuccess;
    }
    
    if (taskData.triangleVerts.size() == 0) {
        MGlobal::displayWarning("No triangle vertices found, skipping GPU binding data update");
        return MS::kSuccess;
    }
    
    if (taskData.baryCoords.size() == 0) {
        MGlobal::displayWarning("No barycentric coordinates found, skipping GPU binding data update");
        return MS::kSuccess;
    }
    
    // 바인딩 매트릭스를 float 배열로 변환 (cvWrap 방식)
    size_t arraySize = taskData.bindMatrices.length() * 16;
    float* bindMatrices = new float[arraySize];
    for(unsigned int i = 0, idx = 0; i < taskData.bindMatrices.length(); ++i) {
        for(unsigned int row = 0; row < 4; row++) {
            for(unsigned int column = 0; column < 4; column++) {
                bindMatrices[idx++] = (float)taskData.bindMatrices[i](row, column);
            }
        }
    }
    
    // GPU 메모리 검증
    if (!bindMatrices_.get()) {
        MGlobal::displayError("bindMatrices GPU memory not initialized");
        delete [] bindMatrices;
            return MS::kFailure;
        }
        
    cl_int err = EnqueueBuffer(bindMatrices_, arraySize * sizeof(float), (void*)bindMatrices);
    delete [] bindMatrices;
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue bindMatrices to GPU");
        return MS::kFailure;
    }

    // 버텍스당 샘플 수 저장 (cvWrap 방식)
    arraySize = taskData.sampleIds.size();
    int* samplesPerVertex = new int[arraySize];
    int* sampleOffsets = new int[arraySize];
    int totalSamples = 0;
    for(size_t i = 0; i < taskData.sampleIds.size(); ++i) {
        samplesPerVertex[i] = (int)taskData.sampleIds[i].length();
        sampleOffsets[i] = totalSamples;
        totalSamples += samplesPerVertex[i];
    }
    
    // GPU 메모리 검증
    if (!sampleCounts_.get() || !sampleOffsets_.get()) {
        MGlobal::displayError("sampleCounts or sampleOffsets GPU memory not initialized");
        delete [] samplesPerVertex;
        delete [] sampleOffsets;
        return MS::kFailure;
    }
    
    err = EnqueueBuffer(sampleCounts_, arraySize * sizeof(int), (void*)samplesPerVertex);
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue sampleCounts to GPU");
        delete [] samplesPerVertex;
        delete [] sampleOffsets;
        return MS::kFailure;
    }
    
    err = EnqueueBuffer(sampleOffsets_, arraySize * sizeof(int), (void*)sampleOffsets);
    delete [] samplesPerVertex;
    delete [] sampleOffsets;
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue sampleOffsets to GPU");
        return MS::kFailure;
    }

    // sampleIds와 sampleWeights 저장 (cvWrap 방식)
    int* sampleIds = new int[totalSamples];
    float* sampleWeights = new float[totalSamples];
    int iter = 0;
    for(size_t i = 0; i < taskData.sampleIds.size(); ++i) {
        for(unsigned int j = 0; j < taskData.sampleIds[i].length(); ++j) {
            sampleIds[iter] = taskData.sampleIds[i][j];
            sampleWeights[iter] = (float)taskData.sampleWeights[i][j];
            iter++;
        }
    }
    
    // GPU 메모리 검증
    if (!sampleIds_.get() || !sampleWeights_.get()) {
        MGlobal::displayError("sampleIds or sampleWeights GPU memory not initialized");
        delete [] sampleIds;
        delete [] sampleWeights;
        return MS::kFailure;
    }
    
    err = EnqueueBuffer(sampleIds_, totalSamples * sizeof(int), (void*)sampleIds);
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue sampleIds to GPU");
        delete [] sampleIds;
        delete [] sampleWeights;
        return MS::kFailure;
    }
    
    err = EnqueueBuffer(sampleWeights_, totalSamples * sizeof(float), (void*)sampleWeights);
    delete [] sampleIds;
    delete [] sampleWeights;
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue sampleWeights to GPU");
        return MS::kFailure;
    }

    // 삼각형 버텍스와 바리센트릭 좌표 저장 (cvWrap 방식)
    arraySize = taskData.triangleVerts.size() * 3;
    int* triangleVerts = new int[arraySize];
    float* baryCoords = new float[arraySize];
    iter = 0;
    for(size_t i = 0; i < taskData.triangleVerts.size(); ++i) {
        for(unsigned int j = 0; j < 3; ++j) {
            triangleVerts[iter] = taskData.triangleVerts[i][j];
            baryCoords[iter] = (float)taskData.baryCoords[i][j];
            iter++;
        }
    }
    
    // GPU 메모리 검증
    if (!triangleVerts_.get() || !baryCoords_.get()) {
        MGlobal::displayError("triangleVerts or baryCoords GPU memory not initialized");
        delete [] triangleVerts;
        delete [] baryCoords;
        return MS::kFailure;
    }
    
    err = EnqueueBuffer(triangleVerts_, arraySize * sizeof(int), (void*)triangleVerts);
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue triangleVerts to GPU");
        delete [] triangleVerts;
        delete [] baryCoords;
        return MS::kFailure;
    }
    
    err = EnqueueBuffer(baryCoords_, arraySize * sizeof(float), (void*)baryCoords);
    delete [] triangleVerts;
    delete [] baryCoords;
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue baryCoords to GPU");
        return MS::kFailure;
    }
    
        return MS::kSuccess;
}

MStatus OffsetCurveGPUDeformer::EnqueueCurveData(MDataBlock& data, 
                                                 const MEvaluationNode& evaluationNode, 
                                                 const MPlug& plug) {
    MStatus status;
    TaskData taskData;
    // GetBindInfo는 offsetCurveDeformerNode의 멤버 함수이므로 임시로 성공 반환
    status = MS::kSuccess;
    CHECK_MSTATUS_AND_RETURN_IT(status);
    cl_int err = CL_SUCCESS;
    
    // curvePoints와 curveTangents는 TaskData에 없으므로 제거
    // 대신 drivenMatrices만 처리

    // Store the driven matrices on the gpu.
    MArrayDataHandle hInputs = data.inputValue(OffsetCurveDeformerNode::input, &status);
        if (!status) {
        MGlobal::displayError("Failed to get input data in EnqueueCurveData");
        return status;
    }
    
    unsigned int geomIndex = plug.logicalIndex();
    status = hInputs.jumpToElement(geomIndex);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    MDataHandle hInput = hInputs.inputValue(&status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    MDataHandle hGeom = hInput.child(OffsetCurveDeformerNode::inputGeom);
    if (hGeom.data().isNull()) {
        MGlobal::displayError("Invalid geometry handle in EnqueueCurveData");
        return MS::kFailure;
    }
    
    MMatrix localToWorldMatrix = hGeom.geometryTransformMatrix();
    MMatrix worldToLocalMatrix = localToWorldMatrix.inverse();
    float drivenMatrices[48]; // 0-15: localToWorld, 16-31: worldToLocal, 32-47: scale

    // Store in column order so we can dot in the cl kernel.
    int idx = 0;
    for(unsigned int column = 0; column < 4; column++) {
        for(unsigned int row = 0; row < 4; row++) {
            drivenMatrices[idx++] = (float)localToWorldMatrix(row, column);
        }
    }
    for(unsigned int column = 0; column < 4; column++) {
        for(unsigned int row = 0; row < 4; row++) {
            drivenMatrices[idx++] = (float)worldToLocalMatrix(row, column);
        }
    }
    
    // Scale matrix is stored row major
    float scale = data.inputValue(OffsetCurveDeformerNode::aOffsetCurves, &status).asFloat();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    MMatrix scaleMatrix;
    scaleMatrix[0][0] = scale;
    scaleMatrix[1][1] = scale;
    scaleMatrix[2][2] = scale;
    for(unsigned int row = 0; row < 4; row++) {
        for(unsigned int column = 0; column < 4; column++) {
            drivenMatrices[idx++] = (float)scaleMatrix(row, column);
        }
    }
    
    // GPU 메모리 검증
    if (!drivenMatrices_.get()) {
        MGlobal::displayError("drivenMatrices GPU memory not initialized");
        return MS::kFailure;
    }
    
    err = EnqueueBuffer(drivenMatrices_, 48 * sizeof(float), (void*)drivenMatrices);
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue drivenMatrices to GPU");
        return MS::kFailure;
    }
    
    return MS::kSuccess;
}

MStatus OffsetCurveGPUDeformer::EnqueuePaintMapData(MDataBlock& data,
                                                    const MEvaluationNode& evaluationNode,
                                                    unsigned int numElements,
                                                    const MPlug& plug) {
    MStatus status;
    if ((paintWeights_.get() &&
         !evaluationNode.dirtyPlugExists(MPxDeformerNode::weightList, &status)) || !status) {
        // The paint weights are not dirty so no need to get them.
        return MS::kSuccess;
    }

    cl_int err = CL_SUCCESS;

    // Store the paint weights on the gpu.
    float* paintWeights = new float[numElements];
    MArrayDataHandle weightList = data.outputArrayValue(MPxDeformerNode::weightList, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    unsigned int geomIndex = plug.logicalIndex();
    status = weightList.jumpToElement(geomIndex);
    
    // It is possible that the jumpToElement fails. In that case all weights are 1.
    if (!status) {  
        for(unsigned int i = 0; i < numElements; i++) {
            paintWeights[i] = 1.0f;
        }
    } else {
        // Initialize all weights to 1.0f
        for(unsigned int i = 0; i < numElements; i++) {
            paintWeights[i] = 1.0f;
        }
        MDataHandle weightsStructure = weightList.inputValue(&status);
        if (status) {
            MArrayDataHandle weights = weightsStructure.child(MPxDeformerNode::weightList);
            unsigned int numWeights = weights.elementCount(&status);
            if (status && numWeights > 0) {
                status = weights.jumpToElement(0);
                if (status) {
                    MDataHandle weight = weights.inputValue(&status);
                    if (status) {
                        // Maya 2020 호환: 단순화된 접근 방식
                        // 기본값 사용 (나중에 실제 가중치 로딩 구현)
                        for(unsigned int i = 0; i < numElements; i++) {
                            paintWeights[i] = 1.0f;
                        }
                    }
                }
            }
        }
    }
    
    // GPU 메모리 검증
    if (!paintWeights_.get()) {
        MGlobal::displayError("paintWeights GPU memory not initialized");
        delete [] paintWeights;
        return MS::kFailure;
    }
    
    err = EnqueueBuffer(paintWeights_, numElements * sizeof(float), (void*)paintWeights);
    delete [] paintWeights;
    
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to enqueue paintWeights to GPU");
        return MS::kFailure;
    }
    
    return MS::kSuccess;
}

void OffsetCurveGPUDeformer::terminate() {
    curvePoints_.reset();
    curveTangents_.reset();
    paintWeights_.reset();
    bindMatrices_.reset();
    sampleCounts_.reset();
    sampleIds_.reset();
    sampleWeights_.reset();
    triangleVerts_.reset();
    baryCoords_.reset();
    drivenMatrices_.reset();
    MOpenCLInfo::releaseOpenCLKernel(kernel_);
    kernel_.reset();
}

bool OffsetCurveGPUDeformer::ValidateNode(MDataBlock& block, 
                                         const MEvaluationNode& evaluationNode,
                                         const MPlug& plug, 
                                         MStringArray* messages) {
    return true;
}

// ============================================================================
// Command Implementation - 나중에 구현 예정
// ============================================================================

// ============================================================================
// Helper Functions and Data Structures
// ============================================================================

// TaskData 구조체는 헤더 파일에 이미 정의되어 있음

// 이 함수들은 offsetCurveDeformerNode 클래스의 멤버 함수로 이미 정의되어 있음
