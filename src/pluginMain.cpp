#include "offsetCurveDeformer.h"
#include "offsetCurveCmd.h"

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>

MStatus initializePlugin(MObject obj) { 
  MStatus status;
  MFnPlugin plugin(obj, "AI Assistant", "1.0", "Any");
  
  // OCD 디포머 노드 등록
  status = plugin.registerNode(OffsetCurveDeformer::kName, 
                              OffsetCurveDeformer::id, 
                              OffsetCurveDeformer::creator, 
                              OffsetCurveDeformer::initialize,
                              MPxNode::kDeformerNode);
  CHECK_MSTATUS_AND_RETURN_IT(status);
  
  // OCD 명령어 등록
  status = plugin.registerCommand(OffsetCurveCmd::kName, 
                                 OffsetCurveCmd::creator, 
                                 OffsetCurveCmd::newSyntax);
  CHECK_MSTATUS_AND_RETURN_IT(status);

  // GPU 디포머 등록 (Maya 2016+)
#if MAYA_API_VERSION >= 201600
  status = MGPUDeformerRegistry::registerGPUDeformerCreator(
    OffsetCurveDeformer::kName, "offsetCurveOverride",
    OffsetCurveGPU::GetGPUDeformerInfo());
  CHECK_MSTATUS_AND_RETURN_IT(status);
  
  // 플러그인 로드 경로 설정
  OffsetCurveGPU::pluginLoadPath = plugin.loadPath();
#endif

  if (MGlobal::mayaState() == MGlobal::kInteractive) {
    MGlobal::executePythonCommandOnIdle("import offsetCurveDeformer.menu");
    MGlobal::executePythonCommandOnIdle("offsetCurveDeformer.menu.create_menuitems()");
  }

  return MS::kSuccess;
}

MStatus uninitializePlugin(MObject obj) {
  MStatus status;
  MFnPlugin plugin(obj);

#if MAYA_API_VERSION >= 201600
  status = MGPUDeformerRegistry::deregisterGPUDeformerCreator(
    OffsetCurveDeformer::kName, "offsetCurveOverride");
  CHECK_MSTATUS_AND_RETURN_IT(status);
#endif

  status = plugin.deregisterCommand(OffsetCurveCmd::kName);
  CHECK_MSTATUS_AND_RETURN_IT(status);
  
  status = plugin.deregisterNode(OffsetCurveDeformer::id);
  CHECK_MSTATUS_AND_RETURN_IT(status);

  if (MGlobal::mayaState() == MGlobal::kInteractive) {
    MGlobal::executePythonCommandOnIdle("import offsetCurveDeformer.menu");
    MGlobal::executePythonCommandOnIdle("offsetCurveDeformer.menu.destroy_menuitems()");
  }
  
  return MS::kSuccess;
}
