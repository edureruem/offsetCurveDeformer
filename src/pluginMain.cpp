/**
 * pluginMain.cpp
 * Maya 플러그인 등록 및 초기화
 */

// 프로젝트 헤더
#include "offsetCurveDeformerNode.h"
#include "offsetCurveCmd.h"

// Maya 헤더들
#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>

// GPU 가속 등록을 위한 헤더 (Maya 2020 전용)
#include <maya/MGPUDeformerRegistry.h>
#include <maya/MOpenCLInfo.h>

// cvWrap에서 사용하는 상태 체크 매크로 (Maya에서 이미 정의된 경우 제외)
#ifndef CHECK_MSTATUS_AND_RETURN_IT
#define CHECK_MSTATUS_AND_RETURN_IT(_stat) \
    if (MStatus::kSuccess != (_stat)) { \
        return (_stat); \
    }
#endif

MStatus initializePlugin(MObject obj) { 
    MStatus status;
    MFnPlugin plugin(obj, "Offset Curve Deformer", "2.0", "Any");
    
    // 노드 등록
    status = plugin.registerNode(OffsetCurveDeformerNode::nodeName, OffsetCurveDeformerNode::id, 
                                OffsetCurveDeformerNode::creator, OffsetCurveDeformerNode::initialize,
                                MPxNode::kDeformerNode);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 명령어 등록
    status = plugin.registerCommand(OffsetCurveCmd::kName, OffsetCurveCmd::creator, OffsetCurveCmd::newSyntax);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // GPU 시스템 검증 (cvWrap 안정성 기법 3번) - Maya 2020 호환
    bool gpuRegistrationSuccess = false;
    try {
        // OpenCL 디바이스 ID 확인
        cl_device_id deviceId = MOpenCLInfo::getOpenCLDeviceId();
        if (deviceId != nullptr) {
            // GPU 가속 등록 (Maya 2020 전용) - cvWrap 패턴
            status = MGPUDeformerRegistry::registerGPUDeformerCreator(OffsetCurveDeformerNode::nodeName, "offsetCurveOverride",
                                                                    OffsetCurveGPUDeformer::GetGPUDeformerInfo());
            if (status == MS::kSuccess) {
                // Set the load path so we can find the cl kernel.
                OffsetCurveGPUDeformer::pluginLoadPath = plugin.loadPath();
                gpuRegistrationSuccess = true;
                MGlobal::displayInfo("GPU acceleration enabled for Offset Curve Deformer");
            } else {
                MGlobal::displayWarning("GPU registration failed, continuing with CPU fallback");
            }
        } else {
            MGlobal::displayWarning("OpenCL device not available, Offset Curve Deformer will use CPU fallback");
        }
    } catch (...) {
        MGlobal::displayWarning("GPU initialization failed with exception, continuing with CPU fallback");
    }
    
    if (!gpuRegistrationSuccess) {
        MGlobal::displayInfo("Offset Curve Deformer initialized with CPU fallback mode");
    }

    // Maya 상태 확인 (cvWrap 안정성 기법 2번)
    if (MGlobal::mayaState() == MGlobal::kInteractive) {
        // 안전한 Python 메뉴 생성 (크래시 방지)
        try {
            // scripts 폴더 존재 확인
            MString scriptsPath = plugin.loadPath() + "/scripts";
            if (MGlobal::executeCommand("file -q -ex " + scriptsPath)) {
                MGlobal::executeCommand("python(\"import sys; sys.path.append('" + scriptsPath + "'); import offsetCurveDeformer.menu; offsetCurveDeformer.menu.createMenu()\")");
            } else {
                MGlobal::displayWarning("Scripts directory not found, menu creation skipped");
            }
        } catch (...) {
            MGlobal::displayWarning("Python menu creation failed, continuing without menu");
        }
    }

    return status;
}

MStatus uninitializePlugin(MObject obj) {
    MStatus status;
    MFnPlugin plugin(obj);

    // Maya 상태 확인 (cvWrap 안정성 기법 2번)
    if (MGlobal::mayaState() == MGlobal::kInteractive) {
        // 안전한 Python 메뉴 제거 (크래시 방지)
        try {
            MString scriptsPath = plugin.loadPath() + "/scripts";
            if (MGlobal::executeCommand("file -q -ex " + scriptsPath)) {
                MGlobal::executeCommand("python(\"import sys; sys.path.append('" + scriptsPath + "'); import offsetCurveDeformer.menu; offsetCurveDeformer.menu.removeMenu()\")");
            }
        } catch (...) {
            MGlobal::displayWarning("Python menu removal failed");
        }
    }

    // GPU 시스템 검증 후 등록 해제 - Maya 2020 호환
    try {
        cl_device_id deviceId = MOpenCLInfo::getOpenCLDeviceId();
        if (deviceId != nullptr) {
            // GPU 가속 등록 해제 (Maya 2020 전용) - cvWrap 패턴
            status = MGPUDeformerRegistry::deregisterGPUDeformerCreator(OffsetCurveDeformerNode::nodeName, "offsetCurveOverride");
            if (status != MS::kSuccess) {
                MGlobal::displayWarning("GPU deregistration failed");
            }
        }
    } catch (...) {
        MGlobal::displayWarning("GPU deregistration failed with exception");
    }
    
    // 명령어 등록 해제
    status = plugin.deregisterCommand(OffsetCurveCmd::kName);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 노드 등록 해제
    status = plugin.deregisterNode(OffsetCurveDeformerNode::id);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    return status;
}