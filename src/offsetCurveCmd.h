#ifndef OFFSETCURVECMD_H
#define OFFSETCURVECMD_H

#include <maya/MArgList.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MDGModifier.h>
#include <maya/MFloatArray.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MMatrixArray.h>
#include <maya/MMeshIntersector.h>
#include <maya/MObjectArray.h>
#include <maya/MPlug.h>
#include <maya/MPointArray.h>
#include <maya/MSelectionList.h>
#include <maya/MString.h>
#include <maya/MStringArray.h>
#include <maya/MThreadPool.h>

#include <maya/MPxCommand.h>

#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include "offsetCurveTypes.h"

// OCD 바인딩 데이터 구조체 (cvwrap 방식)
struct OCDBindData {
    MPointArray inputPoints;           // 변형될 지오메트리의 월드 공간 포인트
    std::vector<MDagPath> influenceCurves;  // 영향 곡선들
    std::vector<OffsetPrimitive> offsetPrimitives;  // 오프셋 프리미티브들
    MMatrixArray bindMatrices;         // 바인딩 매트릭스
    std::vector<MIntArray> sampleIds;  // 샘플 ID
    std::vector<MDoubleArray> sampleWeights; // 샘플 가중치
    std::vector<MIntArray> triangleVerts;   // 삼각형 버텍스 (곡선 기반)
    std::vector<MFloatArray> baryCoords;     // Barycentric 좌표 (곡선 기반)
};

/**
  The offsetCurve command is used to create new offsetCurve deformers and to import and export
  curve bindings.
*/
class OffsetCurveCmd : public MPxCommand {              
public:
    enum CommandMode { kCommandCreate, kCommandExport, kCommandImport, kCommandHelp, kCommandRebind };
    OffsetCurveCmd();              
    virtual MStatus  doIt(const MArgList&);
    virtual MStatus  undoIt();
    virtual MStatus  redoIt();
    virtual bool isUndoable() const;
    static void* creator();    
    static MSyntax newSyntax();

    /**
      Distributes the ThreadData objects to the parallel threads for binding calculation.
      @param[in] data The user defined data. In this case, the ThreadData array.
      @param[in] pRoot Maya's root task.
    */
    static void CreateTasks(void *data, MThreadRootTask *pRoot);
    static MThreadRetVal CalculateBindingTask(void *pParam);

    const static char* kName;  /**< The name of the command. */
    
    /**
      Specifies the name of the offsetCurve node.
    */
    const static char* kNameFlagShort;
    const static char* kNameFlagLong;
    
    /**
      Specifies the sample radius of the binding.
    */
    const static char* kRadiusFlagShort;
    const static char* kRadiusFlagLong;

    /**
      Specifies that a new bind mesh should be created for rebinding.
    */
    const static char* kNewBindMeshFlagShort;
    const static char* kNewBindMeshFlagLong;

    /**
      Export file path.
    */
    const static char* kExportFlagShort;
    const static char* kExportFlagLong;

    /**
      Import file path.
    */
    const static char* kImportFlagShort;
    const static char* kImportFlagLong;
    
    /**
      Path of a binding on disk rather than calculating binding from scratch.
    */
    const static char* kBindingFlagShort;  
    const static char* kBindingFlagLong;

    /**
      Specifies that the user wants to rebind the selected vertices.
    */
    const static char* kRebindFlagShort;
    const static char* kRebindFlagLong;

    /**
      Displays help.
    */
    const static char* kHelpFlagShort;
    const static char* kHelpFlagLong;

private:
    /**
      Gathers all the command arguments and sets necessary command states.
      @param[in] args Maya MArgList.
    */
    MStatus GatherCommandArguments(const MArgList& args);

    /**
      Acquires the influence curves and driven dag paths from the input selection list.
    */
    MStatus GetGeometryPaths();

    /**
      Creates a new offset curve deformer.
    */
    MStatus CreateOffsetCurveDeformer();

    /**
      Gets the latest offsetCurve node in the history of the deformed shape.
    */
    MStatus GetLatestDeformerNode();

    /**
      Calculates the binding data for the offset curve deformer to work.
      @param[in] bindData The structure containing all the bind information.
      @param[in,out] dgMod The modifier to hold all the plug operations.
    */
    MStatus CalculateBinding(OCDBindData& bindData, MDGModifier& dgMod);
    
    /**
      Calculates new binding data for the selected components.
    */
    MStatus Rebind();

    MString name_;           /**< Name of offsetCurve node to create. */
    double radius_;          /**< Binding sample radius. */
    CommandMode command_;
    MString filePath_;
    bool useBinding_;
    bool newBindMesh_;
    MSelectionList selectionList_;  /**< Selected command input nodes. */
    MObject oDeformerNode_; /**< MObject to the offsetCurve node in focus. */
    std::vector<MDagPath> influenceCurves_;  /**< Paths to the influence curves. */
    MObjectArray drivenComponents_; /**< Selected driven components used for rebinding. */
    MDagPathArray pathDriven_;     /**< Paths to the shapes being deformed. */
    MDGModifier dgMod_;
    
};  

#endif
