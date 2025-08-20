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

MStatus initializePlugin(MObject obj)
{
    MStatus status;
    MFnPlugin plugin(obj, "Offset Curve Deformer", "1.0", "Any");
    
        // 추가: Maya 출력 인코딩을 UTF-8로 설정 (한글 출력 문제 해결)
    MGlobal::executeCommand("optionVar -stringValue \"scriptEditorEncoding\" \"utf-8\"");
    MGlobal::executeCommand("optionVar -stringValue \"scriptEditorFileEncoding\" \"utf-8\"");

    // 추가: 더 확실한 인코딩 설정
    MGlobal::executeCommand("optionVar -stringValue \"scriptEditorEncoding\" \"utf-8\"");
    MGlobal::executeCommand("optionVar -stringValue \"scriptEditorFileEncoding\" \"utf-8\"");
    MGlobal::executeCommand("optionVar -stringValue \"scriptEditorOutputEncoding\" \"utf-8\"");

    // 추가: Maya 시스템 인코딩 설정
    MGlobal::executeCommand("optionVar -stringValue \"scriptEditorSystemEncoding\" \"utf-8\"");
    
    // 노드 등록
    status = plugin.registerNode(
        offsetCurveDeformerNode::nodeName,
        offsetCurveDeformerNode::id,
        offsetCurveDeformerNode::creator,
        offsetCurveDeformerNode::initialize,
        MPxNode::kDeformerNode
    );
    
    if (!status) {
        status.perror("Failed to register Offset Curve Deformer node");
        return status;
    }
    
    // 명령어 등록 (cvwrap 방식)
    status = plugin.registerCommand(
        OffsetCurveCmd::kName,
        OffsetCurveCmd::creator,
        OffsetCurveCmd::newSyntax
    );
    
    if (!status) {
        status.perror("Failed to register OffsetCurve command");
        return status;
    }
    
    // 플러그인 등록 성공 메시지
    MGlobal::displayInfo("Offset Curve Deformer plugin loaded successfully.");
    
    // 사용법 설명
    MGlobal::displayInfo("Usage: Select a mesh and run: deformer -type offsetCurveDeformer");
    MGlobal::displayInfo("Then connect nurbs curve(s) to the offsetCurves attribute.");
    
    return status;
}

MStatus uninitializePlugin(MObject obj)
{
    MStatus status;
    MFnPlugin plugin(obj);
    
    // 명령어 등록 해제
    status = plugin.deregisterCommand(OffsetCurveCmd::kName);
    
    if (!status) {
        status.perror("Failed to deregister OffsetCurve command");
        return status;
    }
    
    // 노드 등록 해제
    status = plugin.deregisterNode(offsetCurveDeformerNode::id);
    
    if (!status) {
        status.perror("Failed to deregister Offset Curve Deformer node");
        return status;
    }
    
    MGlobal::displayInfo("Offset Curve Deformer plugin unloaded.");
    
    return status;
}