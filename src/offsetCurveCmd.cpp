#include "offsetCurveCmd.h"
#include "offsetCurveDeformerNode.h"

#include <maya/MArgDatabase.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
#include <maya/MItDependencyGraph.h>
#include <maya/MItGeometry.h>
#include <maya/MItSelectionList.h>
#include <maya/MMeshIntersector.h>
#include <maya/MFnSingleIndexedComponent.h>
#include <maya/MFnWeightGeometryFilter.h>
#include <maya/MSyntax.h>
#include <algorithm>
#include <cassert>
#include <utility>

#define PROGRESS_STEP 100
#define TASK_COUNT 32

/**
  A version number used to support future updates to the binary binding file.
*/
const float kBindingFileVersion = 1.0f;

const char* OffsetCurveCmd::kName = "offsetCurve";
const char* OffsetCurveCmd::kNameFlagShort = "-n";
const char* OffsetCurveCmd::kNameFlagLong = "-name";
const char* OffsetCurveCmd::kRadiusFlagShort = "-r";
const char* OffsetCurveCmd::kRadiusFlagLong = "-radius";
const char* OffsetCurveCmd::kNewBindMeshFlagShort = "-nbm";
const char* OffsetCurveCmd::kNewBindMeshFlagLong = "-newBindMesh";
const char* OffsetCurveCmd::kExportFlagShort = "-ex";
const char* OffsetCurveCmd::kExportFlagLong = "-export";
const char* OffsetCurveCmd::kImportFlagShort = "-im";
const char* OffsetCurveCmd::kImportFlagLong = "-import";
const char* OffsetCurveCmd::kBindingFlagShort = "-b";
const char* OffsetCurveCmd::kBindingFlagLong = "-binding";
const char* OffsetCurveCmd::kRebindFlagShort = "-rb";
const char* OffsetCurveCmd::kRebindFlagLong = "-rebind";
const char* OffsetCurveCmd::kHelpFlagShort = "-h";
const char* OffsetCurveCmd::kHelpFlagLong = "-help";

/**
  Displays command instructions.
*/
void DisplayHelp() {
    MString help;
    help += "Flags:\n"; 
    help += "-name (-n):          String     Name of the offsetCurve node to create.\n"; 
    help += "-radius (-r):        Double     Sample radius.  Default is 0.1.  The greater the radius,\n"; 
    help += "                                the smoother the deformation but slower performance.\n";
    help += "-newBindMesh (-nbm)  N/A        Creates a new bind mesh, otherwise the existing bind mesh will be used.\n";
    help += "-export (-ex):       String     Path to a file to export the binding to.\n"; 
    help += "-import (-im):       String     Path to a file to import the binding from.\n"; 
    help += "-binding (-b):       String     Path to a file to import the binding from on creation.\n"; 
    help += "-rebind (-rb):       String     The name of the offsetCurve node we are rebinding.\n"; 
    help += "-help (-h)           N/A        Display this text.\n";
    MGlobal::displayInfo(help);
}

OffsetCurveCmd::OffsetCurveCmd()
    : radius_(0.1),
      name_("offsetCurve#"),
      command_(kCommandCreate),
      useBinding_(false),
      newBindMesh_(false) {
}

MSyntax OffsetCurveCmd::newSyntax() {
    MSyntax syntax;
    syntax.addFlag(kNameFlagShort, kNameFlagLong, MSyntax::kString);
    syntax.addFlag(kRadiusFlagShort, kRadiusFlagLong, MSyntax::kDouble);
    syntax.addFlag(kNewBindMeshFlagShort, kNewBindMeshFlagLong);
    syntax.addFlag(kExportFlagShort, kExportFlagLong, MSyntax::kString);
    syntax.addFlag(kImportFlagShort, kImportFlagLong, MSyntax::kString);
    syntax.addFlag(kBindingFlagShort, kBindingFlagLong, MSyntax::kString);
    syntax.addFlag(kRebindFlagShort, kRebindFlagLong, MSyntax::kString);
    syntax.addFlag(kHelpFlagShort, kHelpFlagLong);
    syntax.setObjectType(MSyntax::kSelectionList, 2, 2);
    syntax.useSelectionAsDefault(true);
    return syntax;
}

void* OffsetCurveCmd::creator() {
    return new OffsetCurveCmd();
}

bool OffsetCurveCmd::isUndoable() const {
    return true;
}

MStatus OffsetCurveCmd::doIt(const MArgList& args) {
    MStatus status;
    
    // 명령어 인자 파싱
    status = GatherCommandArguments(args);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 도움말 표시
    if (command_ == kCommandHelp) {
        DisplayHelp();
        return MS::kSuccess;
    }
    
    // 선택된 지오메트리 경로 가져오기
    status = GetGeometryPaths();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 명령어 모드별 실행
    switch (command_) {
        case kCommandCreate:
            status = CreateOffsetCurveDeformer();
            break;
        case kCommandExport:
            // TODO: 바인딩 익스포트 구현
            MGlobal::displayInfo("Export functionality not yet implemented");
            break;
        case kCommandImport:
            // TODO: 바인딩 임포트 구현
            MGlobal::displayInfo("Import functionality not yet implemented");
            break;
        case kCommandRebind:
            status = Rebind();
            break;
        default:
            MGlobal::displayError("Unknown command mode");
            return MS::kFailure;
    }
    
    return status;
}

MStatus OffsetCurveCmd::undoIt() {
    // TODO: Undo 기능 구현
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::redoIt() {
    // TODO: Redo 기능 구현
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::GatherCommandArguments(const MArgList& args) {
    MStatus status;
    MArgDatabase argData(syntax(), args, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 이름 플래그
    if (argData.isFlagSet(kNameFlagShort)) {
        status = argData.getFlagArgument(kNameFlagShort, 0, name_);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    // 반지름 플래그
    if (argData.isFlagSet(kRadiusFlagShort)) {
        status = argData.getFlagArgument(kRadiusFlagShort, 0, radius_);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    // 새 바인딩 메시 플래그
    if (argData.isFlagSet(kNewBindMeshFlagShort)) {
        newBindMesh_ = true;
    }
    
    // 익스포트 플래그
    if (argData.isFlagSet(kExportFlagShort)) {
        command_ = kCommandExport;
        status = argData.getFlagArgument(kExportFlagShort, 0, filePath_);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        return MS::kSuccess;
    }
    
    // 임포트 플래그
    if (argData.isFlagSet(kImportFlagShort)) {
        command_ = kCommandImport;
        status = argData.getFlagArgument(kImportFlagShort, 0, filePath_);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        return MS::kSuccess;
    }
    
    // 바인딩 플래그
    if (argData.isFlagSet(kBindingFlagShort)) {
        useBinding_ = true;
        status = argData.getFlagArgument(kBindingFlagShort, 0, filePath_);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    // 리바인드 플래그
    if (argData.isFlagSet(kRebindFlagShort)) {
        command_ = kCommandRebind;
        status = argData.getFlagArgument(kRebindFlagShort, 0, name_);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        return MS::kSuccess;
    }
    
    // 도움말 플래그
    if (argData.isFlagSet(kHelpFlagShort)) {
        command_ = kCommandHelp;
        return MS::kSuccess;
    }
    
    // 선택 리스트 가져오기
    status = argData.getObjects(selectionList_);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::GetGeometryPaths() {
    MStatus status;
    
    if (selectionList_.length() < 2) {
        MGlobal::displayError("Please select at least 2 objects: driven geometry and influence curves");
        return MS::kFailure;
    }
    
    // 첫 번째 선택: 변형될 지오메트리
    MDagPath drivenPath;
    status = selectionList_.getDagPath(0, drivenPath);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    if (!drivenPath.isValid()) {
        MGlobal::displayError("Invalid driven geometry path");
        return MS::kFailure;
    }
    
    pathDriven_.append(drivenPath);
    
    // 두 번째 선택: 영향 곡선들
    MDagPath curvePath;
    status = selectionList_.getDagPath(1, curvePath);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    if (!curvePath.isValid()) {
        MGlobal::displayError("Invalid influence curve path");
        return MS::kFailure;
    }
    
    influenceCurves_.push_back(curvePath);
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::CreateOffsetCurveDeformer() {
    MStatus status;
    
    // offsetCurve 디포머 노드 생성
    MString command = "deformer -type offsetCurveDeformer";
    if (name_ != "offsetCurve#") {
        command += " -name " + name_;
    }
    command += " " + pathDriven_[0].partialPathName();
    
    status = MGlobal::executeCommand(command);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 생성된 디포머 노드 찾기
    status = GetLatestDeformerNode();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 영향 곡선 연결
    MString connectCommand = "connectAttr " + influenceCurves_[0].partialPathName() + ".worldSpace[0] " + 
                            name_ + ".offsetCurves[0]";
    status = MGlobal::executeCommand(connectCommand);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 반지름 설정
    if (radius_ != 0.1) {
        MString radiusCommand = "setAttr " + name_ + ".falloffRadius " + radius_;
        status = MGlobal::executeCommand(radiusCommand);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    MGlobal::displayInfo("OffsetCurve deformer created successfully: " + name_);
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::GetLatestDeformerNode() {
    MStatus status;
    
    // 히스토리에서 최신 offsetCurve 노드 찾기  
    MItDependencyGraph itDG(pathDriven_[0].node(), MFn::kInvalid, MItDependencyGraph::kUpstream);
    
    for (; !itDG.isDone(); itDG.next()) {
        MObject node = itDG.thisNode();
        MFnDependencyNode fnNode(node);
        
        if (fnNode.typeName() == "offsetCurveDeformer") {
            oDeformerNode_ = node;
            return MS::kSuccess;
        }
    }
    
    MGlobal::displayError("Could not find offsetCurve deformer node");
    return MS::kFailure;
}

MStatus OffsetCurveCmd::Rebind() {
    MStatus status;
    
    // 리바인딩 구현 (TODO)
    MGlobal::displayInfo("Rebind functionality not yet implemented");
    
    return MS::kSuccess;
}

// 멀티스레딩 태스크 생성 (cvwrap 방식)
void OffsetCurveCmd::CreateTasks(void *data, MThreadRootTask *pRoot) {
    // TODO: 멀티스레딩 바인딩 계산 구현
}

// 멀티스레딩 바인딩 태스크 (cvwrap 방식)
MThreadRetVal OffsetCurveCmd::CalculateBindingTask(void *pParam) {
    // TODO: 개별 바인딩 태스크 구현
    return (MThreadRetVal)0; // Maya 2020 호환
}
