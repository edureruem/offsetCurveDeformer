/**
 * pluginMain.cpp
 * Maya 플러그인 등록 및 초기화
 */

#include "offsetCurveDeformerNode.h"

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>

MStatus initializePlugin(MObject obj)
{
    MStatus status;
    MFnPlugin plugin(obj, "Offset Curve Deformer", "1.0", "Any");
    
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
    
    // 노드 등록 해제
    status = plugin.deregisterNode(offsetCurveDeformerNode::id);
    
    if (!status) {
        status.perror("Failed to deregister Offset Curve Deformer node");
        return status;
    }
    
    MGlobal::displayInfo("Offset Curve Deformer plugin unloaded.");
    
    return status;
}