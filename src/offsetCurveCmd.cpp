#include "offsetCurveCmd.h"
#include "offsetCurveDeformer.h"
#include "offsetCurveAlgorithm.h"

#include <maya/MArgDatabase.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnMesh.h>
#include <maya/MFnNurbsCurve.h>
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

const char* OffsetCurveCmd::kName = "offsetCurve";
const char* OffsetCurveCmd::kNameFlagShort = "-n";
const char* OffsetCurveCmd::kNameFlagLong = "-name";
const char* OffsetCurveCmd::kOffsetDistanceFlagShort = "-od";
const char* OffsetCurveCmd::kOffsetDistanceFlagLong = "-offsetDistance";
const char* OffsetCurveCmd::kFalloffRadiusFlagShort = "-fr";
const char* OffsetCurveCmd::kFalloffRadiusFlagLong = "-falloffRadius";
const char* OffsetCurveCmd::kCurveTypeFlagShort = "-ct";
const char* OffsetCurveCmd::kCurveTypeFlagLong = "-curveType";
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

void DisplayHelp() {
    MString help;
    help += "Flags:\n";
    help += "-name (-n):          String     Name of the OCD node to create.\n";
    help += "-offsetDistance (-od): Double   Offset distance from influence curves.\n";
    help += "-falloffRadius (-fr):  Double   Influence falloff radius.\n";
    help += "-curveType (-ct):      Int      Curve type (0: B-spline, 1: Arc-segment).\n";
    help += "-newBindMesh (-nbm)    N/A      Creates a new bind mesh.\n";
    help += "-export (-ex):       String     Path to export binding data.\n";
    help += "-import (-im):       String     Path to import binding data.\n";
    help += "-binding (-b):       String     Path to binding file for creation.\n";
    help += "-rebind (-rb):       String     Name of OCD node to rebind.\n";
    help += "-help (-h)           N/A        Display this text.\n";
    MGlobal::displayInfo(help);
}

OffsetCurveCmd::OffsetCurveCmd()
    : offsetDistance_(1.0),
      falloffRadius_(2.0),
      curveType_(kBSpline),
      name_("offsetCurve#"),
      command_(kCommandCreate),
      useBinding_(false),
      newBindMesh_(false) {
}

MSyntax OffsetCurveCmd::newSyntax() {
    MSyntax syntax;
    syntax.addFlag(kNameFlagShort, kNameFlagLong, MSyntax::kString);
    syntax.addFlag(kOffsetDistanceFlagShort, kOffsetDistanceFlagLong, MSyntax::kDouble);
    syntax.addFlag(kFalloffRadiusFlagShort, kFalloffRadiusFlagLong, MSyntax::kDouble);
    syntax.addFlag(kCurveTypeFlagShort, kCurveTypeFlagLong, MSyntax::kLong);
    syntax.addFlag(kNewBindMeshFlagShort, kNewBindMeshFlagLong);
    syntax.addFlag(kExportFlagShort, kExportFlagLong, MSyntax::kString);
    syntax.addFlag(kImportFlagShort, kImportFlagLong, MSyntax::kString);
    syntax.addFlag(kBindingFlagShort, kBindingFlagLong, MSyntax::kString);
    syntax.addFlag(kRebindFlagShort, kRebindFlagLong, MSyntax::kString);
    syntax.addFlag(kHelpFlagShort, kHelpFlagLong);
    syntax.setObjectType(MSyntax::kSelectionList, 0, 255);
    syntax.useSelectionAsDefault(true);
    return syntax;
}

void* OffsetCurveCmd::creator() {
    return new OffsetCurveCmd;
}

bool OffsetCurveCmd::isUndoable() const {
    return command_ == kCommandCreate;
}

MStatus OffsetCurveCmd::doIt(const MArgList& args) {
    MStatus status;
    
    status = GatherCommandArguments(args);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    if (command_ == kCommandImport || command_ == kCommandExport) {
        status = selectionList_.getDependNode(0, oOCDNode_);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        MFnDependencyNode fnNode(oOCDNode_);
        if (fnNode.typeId() != OffsetCurveDeformer::id) {
            MGlobal::displayError("No OCD node specified.");
            return MS::kFailure;
        }
    } else if (command_ == kCommandRebind) {
        status = GetGeometryPaths();
        CHECK_MSTATUS_AND_RETURN_IT(status);
        status = Rebind();
        CHECK_MSTATUS_AND_RETURN_IT(status);
    } else {
        status = GetGeometryPaths();
        CHECK_MSTATUS_AND_RETURN_IT(status);

        MString command = "deformer -type offsetCurveDeformer -n \"" + name_ + "\"";
        for (unsigned int i = 0; i < pathDriven_.length(); ++i) {
            MFnDagNode fnDriven(pathDriven_[i]);
            command += " " + fnDriven.partialPathName();
        }
        status = dgMod_.commandToExecute(command);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    return redoIt();
}

MStatus OffsetCurveCmd::GatherCommandArguments(const MArgList& args) {
    MStatus status;
    MArgDatabase argData(syntax(), args);
    argData.getObjects(selectionList_);
    
    if (argData.isFlagSet(kHelpFlagShort)) {
        command_ = kCommandHelp;
        DisplayHelp();
        return MS::kSuccess;
    } else if (argData.isFlagSet(kExportFlagShort)) {
        command_ = kCommandExport;
        filePath_ = argData.flagArgumentString(kExportFlagShort, 0, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    } else if (argData.isFlagSet(kImportFlagShort)) {
        command_ = kCommandImport;
        filePath_ = argData.flagArgumentString(kImportFlagShort, 0, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    newBindMesh_ = argData.isFlagSet(kNewBindMeshFlagShort);
    
    if (argData.isFlagSet(kOffsetDistanceFlagShort)) {
        offsetDistance_ = argData.flagArgumentDouble(kOffsetDistanceFlagShort, 0, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        if (offsetDistance_ <= 0.0) {
            offsetDistance_ = 0.001;
        }
    }
    
    if (argData.isFlagSet(kFalloffRadiusFlagShort)) {
        falloffRadius_ = argData.flagArgumentDouble(kFalloffRadiusFlagShort, 0, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        if (falloffRadius_ <= 0.0) {
            falloffRadius_ = 0.1;
        }
    }
    
    if (argData.isFlagSet(kCurveTypeFlagShort)) {
        curveType_ = (CurveType)argData.flagArgumentLong(kCurveTypeFlagShort, 0, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        if (curveType_ < 0 || curveType_ > 1) {
            curveType_ = kBSpline;
        }
    }
    
    if (argData.isFlagSet(kNameFlagShort)) {
        name_ = argData.flagArgumentString(kNameFlagShort, 0, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    if (argData.isFlagSet(kBindingFlagShort)) {
        useBinding_ = true;
        filePath_ = argData.flagArgumentString(kBindingFlagShort, 0, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    if (argData.isFlagSet(kRebindFlagShort)) {
        command_ = kCommandRebind;
        MString ocdNode = argData.flagArgumentString(kRebindFlagShort, 0, &status);
        MSelectionList slist;
        status = slist.add(ocdNode);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        status = slist.getDependNode(0, oOCDNode_);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        MFnDependencyNode fnNode(oOCDNode_, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        if (fnNode.typeId() != OffsetCurveDeformer::id) {
            MGlobal::displayError(fnNode.name() + " is not an offsetCurveDeformer node.");
            return MS::kFailure;
        }
    }
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::GetGeometryPaths() {
    MStatus status;
    
    // 마지막 선택된 것이 변형될 지오메트리
    status = selectionList_.getDagPath(selectionList_.length() - 1, pathDriven_, drivenComponents_);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = GetShapeNode(pathDriven_);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MItSelectionList iter(selectionList_);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    pathInfluenceCurves_.clear();
    influenceComponents_.clear();
    
    for (unsigned int i = 0; i < selectionList_.length() - 1; ++i, iter.next()) {
        MDagPath path;
        MObject component;
        iter.getDagPath(path, component);
        status = GetShapeNode(path);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        // 곡선인지 확인
        if (path.hasFn(MFn::kNurbsCurve)) {
            pathInfluenceCurves_.append(path);
            influenceComponents_.append(component);
        } else {
            MGlobal::displayWarning(path.partialPathName() + " is not a curve. Skipping.");
        }
    }
    
    if (pathInfluenceCurves_.length() == 0) {
        MGlobal::displayError("No influence curves found in selection.");
        return MS::kFailure;
    }
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::redoIt() {
    MStatus status;
    
    if (command_ == kCommandImport) {
        // 바인딩 데이터 가져오기 (구현 예정)
        MGlobal::displayInfo("Import functionality not yet implemented.");
        return MS::kSuccess;
    } else if (command_ == kCommandExport) {
        // 바인딩 데이터 내보내기 (구현 예정)
        MGlobal::displayInfo("Export functionality not yet implemented.");
        return MS::kSuccess;
    } else if (command_ == kCommandRebind) {
        status = dgMod_.doIt();
        CHECK_MSTATUS_AND_RETURN_IT(status);
        return MS::kSuccess;
    } else if (command_ == kCommandCreate) {
        status = CreateOffsetCurveDeformer();
        CHECK_MSTATUS_AND_RETURN_IT(status);
        return MS::kSuccess;
    }
    
    return MS::kFailure;
}

MStatus OffsetCurveCmd::CreateOffsetCurveDeformer() {
    MStatus status;
    
    // 디포머 생성
    status = dgMod_.doIt();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 경로 재획득
    status = GetGeometryPaths();
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 생성된 OCD 노드 획득
    status = GetLatestOCDNode();
    CHECK_MSTATUS_AND_RETURN_IT(status);

    MFnDependencyNode fnNode(oOCDNode_, &status);
    setResult(fnNode.name());
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // 바인딩 메시 생성
    MDagPath pathBindMesh;
    status = GetExistingBindMesh(pathBindMesh);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    if (newBindMesh_ || !pathBindMesh.isValid()) {
        status = CreateBindMesh(pathBindMesh);
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }
    
    status = ConnectBindMesh(pathBindMesh);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    if (useBinding_) {
        // 기존 바인딩 파일에서 가져오기 (구현 예정)
        MGlobal::displayInfo("Binding file import not yet implemented.");
    } else {
        // 새로운 바인딩 계산
        MDGModifier dgMod;
        OCDBindData bindData;
        status = CalculateBinding(pathBindMesh, bindData, dgMod);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        status = dgMod.doIt();
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    // 영향 곡선들을 OCD 디포머에 연결
    for (unsigned int i = 0; i < pathInfluenceCurves_.length(); ++i) {
        MFnDagNode fnCurve(pathInfluenceCurves_[i]);
        MPlug plugCurveMesh = fnCurve.findPlug("worldSpace", false, &status);
        if (status != MS::kSuccess) {
            plugCurveMesh = fnCurve.findPlug("local", false, &status);
        }
        CHECK_MSTATUS_AND_RETURN_IT(status);
        
        MPlug plugInfluenceCurves(oOCDNode_, OffsetCurveDeformer::aInfluenceCurves);
        MDGModifier dgMod;
        dgMod.connect(plugCurveMesh, plugInfluenceCurves);
        status = dgMod.doIt();
        CHECK_MSTATUS_AND_RETURN_IT(status);
    }

    return MS::kSuccess;
}

MStatus OffsetCurveCmd::GetLatestOCDNode() {
    MStatus status;
    MObject oDriven = pathDriven_[0].node();
    
    MItDependencyGraph itDG(oDriven,
                           MFn::kGeometryFilt,
                           MItDependencyGraph::kUpstream, 
                           MItDependencyGraph::kDepthFirst,
                           MItDependencyGraph::kNodeLevel, 
                           &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    MObject oDeformerNode;
    for (; !itDG.isDone(); itDG.next()) {
        oDeformerNode = itDG.currentItem();
        MFnDependencyNode fnNode(oDeformerNode, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        if (fnNode.typeId() == OffsetCurveDeformer::id) {
            oOCDNode_ = oDeformerNode;
            return MS::kSuccess;
        }
    }
    
    return MS::kFailure;
}

MStatus OffsetCurveCmd::CreateBindMesh(MDagPath& pathBindMesh) {
    MStatus status;
    MStringArray duplicate;
    MFnDependencyNode fnOCD(oOCDNode_, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 변형될 지오메트리 복제하여 바인딩 메시 생성
    MFnDagNode fnDriven(pathDriven_[0]);
    MGlobal::executeCommand("duplicate -rr -n " + fnOCD.name() + "Base " + fnDriven.partialPathName(), duplicate);
    status = GetDagPath(duplicate[0], pathBindMesh);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    status = DeleteIntermediateObjects(pathBindMesh);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    bindMeshes_.append(duplicate[0]);

    // 바인딩 메시 숨기기
    MFnDagNode fnBindMesh(pathBindMesh, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    MPlug plug = fnBindMesh.findPlug("visibility", false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    status = plug.setBool(false);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::ConnectBindMesh(MDagPath& pathBindMesh) {
    MStatus status;
    
    status = GetShapeNode(pathBindMesh);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    MFnDagNode fnBindMeshShape(pathBindMesh, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    MPlug plugBindMessage = fnBindMeshShape.findPlug("message", false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 바인딩 메시를 OCD 노드에 연결 (구현 예정)
    // MPlug plugBindMesh(oOCDNode_, OffsetCurveDeformer::aBindData);
    // MDGModifier dgMod;
    // dgMod.connect(plugBindMessage, plugBindMesh);
    // status = dgMod.doIt();
    // CHECK_MSTATUS_AND_RETURN_IT(status);
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::CalculateBinding(MDagPath& pathBindMesh, OCDBindData& bindData, MDGModifier& dgMod) {
    MStatus status;
    
    // 바인딩 데이터 초기화
    bindData.offsetDistance = offsetDistance_;
    bindData.falloffRadius = falloffRadius_;
    bindData.curveType = curveType_;
    
    // 바인딩 메시 정보 저장
    bindData.driverMatrix = pathBindMesh.inclusiveMatrix();
    
    // 바인딩 메시의 포인트들 가져오기
    MObject oBindMesh = pathBindMesh.node();
    MFnMesh fnBindMesh(oBindMesh, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    fnBindMesh.getPoints(bindData.inputPoints, MSpace::kWorld);
    
    // 영향 곡선 데이터 추출
    status = ExtractInfluenceCurves(pathInfluenceCurves_, bindData.influenceCurves);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 영향 곡선들의 매트릭스 가져오기
    bindData.influenceMatrices.setLength(pathInfluenceCurves_.length());
    for (unsigned int i = 0; i < pathInfluenceCurves_.length(); ++i) {
        bindData.influenceMatrices[i] = pathInfluenceCurves_[i].inclusiveMatrix();
    }
    
    // OCD 알고리즘을 사용하여 바인딩 계산
    OffsetCurveAlgorithm algorithm;
    status = algorithm.bindModelPoints(
        bindData.inputPoints,
        bindData.influenceCurves,
        bindData.offsetDistance,
        bindData.falloffRadius,
        bindData.bindMatrices,
        bindData.samplePoints,
        bindData.sampleWeights,
        bindData.offsetVectors
    );
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 바인딩 데이터를 OCD 노드에 저장 (구현 예정)
    // 이 부분은 offsetCurveDeformer.cpp에서 구현될 예정
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::GetExistingBindMesh(MDagPath &pathBindMesh) {
    // 기존 바인딩 메시 찾기 (구현 예정)
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::Rebind() {
    // 재바인딩 구현 (구현 예정)
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::GetBindMesh(MObject& oOCDNode, MDagPath& pathBindMesh) {
    // 바인딩 메시 가져오기 (구현 예정)
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::CreateRebindSubsetMesh(MDagPath& pathDriverSubset) {
    // 재바인딩용 서브셋 메시 생성 (구현 예정)
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::ExtractInfluenceCurves(
    const MDagPathArray& curvePaths,
    std::vector<OffsetCurveData>& curves) {
    
    MStatus status;
    curves.clear();
    
    for (unsigned int i = 0; i < curvePaths.length(); ++i) {
        OffsetCurveData curveData;
        
        if (curveType_ == kBSpline) {
            status = ExtractBSplineCurve(curvePaths[i], curveData.bspline);
            curveData.type = kBSpline;
        } else {
            status = ExtractArcSegmentCurve(curvePaths[i], curveData.arc);
            curveData.type = kArcSegment;
        }
        
        if (status == MS::kSuccess) {
            curves.push_back(curveData);
        }
    }
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::ExtractBSplineCurve(const MDagPath& curvePath, BSplineCurve& curve) {
    MStatus status;
    
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 제어점 가져오기
    status = fnCurve.getCVs(curve.controlPoints, MSpace::kWorld);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 노트 벡터 가져오기
    status = fnCurve.getKnots(curve.knots);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 차수 가져오기
    curve.degree = fnCurve.degree();
    
    return MS::kSuccess;
}

MStatus OffsetCurveCmd::ExtractArcSegmentCurve(const MDagPath& curvePath, ArcSegmentCurve& curve) {
    MStatus status;
    
    MFnNurbsCurve fnCurve(curvePath, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    // 시작점과 끝점
    curve.startPoint = fnCurve.point(0, MSpace::kWorld);
    curve.endPoint = fnCurve.point(fnCurve.numCVs() - 1, MSpace::kWorld);
    
    // 단순화된 arc-segment 구현
    // 실제로는 더 정교한 곡선 분석이 필요
    curve.isArc = false; // 기본적으로 선분으로 처리
    curve.centerPoint = (curve.startPoint + curve.endPoint) * 0.5;
    curve.radius = curve.startPoint.distanceTo(curve.endPoint) * 0.5;
    curve.startAngle = 0.0;
    curve.endAngle = M_PI;
    
    return MS::kSuccess;
}

void OffsetCurveCmd::CreateTasks(void *data, MThreadRootTask *pRoot) {
    ThreadData<OCDBindData>* threadData = static_cast<ThreadData<OCDBindData>*>(data);

    if (threadData) {
        int numTasks = threadData[0].numTasks;
        for(int i = 0; i < numTasks; i++) {
            MThreadPool::createTask(CalculateBindingTask, (void *)&threadData[i], pRoot);
        }
        MThreadPool::executeAndJoin(pRoot);
    }
}

MThreadRetVal OffsetCurveCmd::CalculateBindingTask(void *pParam) {
    // 바인딩 계산 태스크 (구현 예정)
    return 0;
}

MStatus OffsetCurveCmd::undoIt() {
    MStatus status;
    status = dgMod_.undoIt();
    CHECK_MSTATUS_AND_RETURN_IT(status);

    if (bindMeshes_.length()) {
        // 생성된 바인딩 메시들 삭제
        MDGModifier mod;
        for (unsigned int i = 0; i < bindMeshes_.length(); i++) {
            status = mod.commandToExecute("delete " + bindMeshes_[i]);
            CHECK_MSTATUS_AND_RETURN_IT(status);
        }
        status = mod.doIt();
        CHECK_MSTATUS_AND_RETURN_IT(status);
        bindMeshes_.clear();
    }

    return MS::kSuccess;
}
