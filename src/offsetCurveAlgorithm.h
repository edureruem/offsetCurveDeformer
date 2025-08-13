/**
 * offsetCurveAlgorithm.h
 * OCD 핵심 알고리즘 구현
 * 소니 특허(US8400455) 기반으로 개선
 */

#ifndef OFFSETCURVEALGORITHM_H
#define OFFSETCURVEALGORITHM_H

// 공통 타입 헤더
#include "offsetCurveTypes.h"

// Maya 핵심 헤더들 (실제 사용되는 것만)
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MDagPath.h>
#include <maya/MStatus.h>

// C++ 표준 라이브러리
#include <vector>
#include <memory>

#include "offsetCurveControlParams.h"
#include "offsetCurveSystems.h"

class offsetCurveAlgorithm {
public:
    offsetCurveAlgorithm();
    ~offsetCurveAlgorithm();
    
    // 초기화 및 바인드
    MStatus initialize(const MPointArray& points, offsetCurveOffsetMode offsetMode);
    MStatus bindToCurves(const std::vector<MDagPath>& curvePaths, 
                       double falloffRadius,
                       int maxInfluences);
    
    // === OCD 알고리즘 ===
    
    // 바인딩 페이즈: 각 정점에 대한 오프셋 프리미티브 생성 (수학적으로만!)
    MStatus performBindingPhase(const MPointArray& modelPoints,
                               const std::vector<MDagPath>& influenceCurves,
                               double falloffRadius = 10.0,
                               int maxInfluences = 3);
    
    // 변형 페이즈: 정확한 수학 공식으로 변형 계산
    MStatus performDeformationPhase(MPointArray& points,
                                   const offsetCurveControlParams& params);
    
    // 레거시 호환성 메서드들 (단순화)
    MStatus computeDeformation(MPointArray& points,
                             const offsetCurveControlParams& params);
    
    // 병렬 처리 활성화/비활성화
    void enableParallelComputation(bool enable);
    
    // 포즈 타겟 설정
    void setPoseTarget(const MPointArray& poseTarget);
    
    // ✅ 추가: 데이터 흐름 관리 메서드들
    MStatus processDataFlow();
    MStatus validateDataFlow();
    MStatus optimizeDataFlow();
    MStatus monitorDataFlowPerformance();
    bool isDataFlowValid() const;
    MStatus getDataFlowStatus() const;
    
    // ✅ 추가: 새로운 시스템 초기화 및 설정
    MStatus initializeBindRemapping(double remappingStrength);
    MStatus initializePoseSpaceDeformation(const std::vector<MObject>& skeletonJoints);
    MStatus initializeAdaptiveSubdivision(double maxCurvatureError);
    
    // ✅ 추가: 시스템 적용 메서드
    MStatus applyBindRemappingToPrimitives(std::vector<OffsetPrimitive>& primitives) const;
    MStatus applyPoseSpaceDeformation(MPointArray& points, 
                                     const std::vector<OffsetPrimitive>& primitives,
                                     const MMatrix& worldMatrix) const;
    std::vector<ArcSegment> getAdaptiveSegments(const MDagPath& curvePath) const;
    MStatus applyPoseBlending(MPointArray& points, 
                             const std::vector<OffsetPrimitive>& primitives,
                             const MMatrix& worldMatrix,
                             double poseBlendingWeight) const;
    
    // 헬퍼 함수들
    void mergeAdjacentSegments(std::vector<ArcSegment>& segments,
                              double maxCurvatureError) const;
    
    MStatus calculatePointOnCurveOnDemand(const MDagPath& curvePath,
                                         double paramU,
                                         MPoint& point) const;
    
    MStatus findClosestPointOnCurveOnDemand(const MDagPath& curvePath,
                                           const MPoint& modelPoint,
                                           double& paramU,
                                           MPoint& closestPoint,
                                           double& distance) const;
    
    // === 아티스트 제어 함수들 (특허 US8400455B2) ===
    MVector applyTwistControl(const MVector& offsetLocal,
                             const MVector& tangent,
                             const MVector& normal,
                             const MVector& binormal,
                             double twistAmount,
                             double paramU) const;
    
    MVector applySlideControl(const MVector& offsetLocal,
                             const MDagPath& curvePath,
                             double& paramU,
                             double slideAmount) const;
    
    MVector applyScaleControl(const MVector& offsetLocal,
                             double scaleAmount,
                             double paramU) const;
    
    // ✅ 수정: Strategy를 사용하는 새로운 함수들
    MStatus calculateFrenetFrameWithStrategy(const MDagPath& curvePath, double paramU,
                                            MVector& tangent, MVector& normal, MVector& binormal) const;
    MStatus getPointAtParamWithStrategy(const MDagPath& curvePath, double paramU, MPoint& point) const;
    MStatus getNormalAtParamWithStrategy(const MDagPath& curvePath, double paramU, MVector& normal) const;
    MStatus getTangentAtParamWithStrategy(const MDagPath& curvePath, double paramU, MVector& tangent) const;
    double getCurvatureAtParamWithStrategy(const MDagPath& curvePath, double paramU) const;
    
    // ✅ 추가: 에러 처리 및 검증 메서드들
    bool validateInputCurves(const std::vector<MDagPath>& curvePaths) const;
    bool validateModelPoints(const MPointArray& points) const;
    bool validateOffsetPrimitives(const std::vector<OffsetPrimitive>& primitives) const;
    
    // ✅ 추가: 성능 최적화 관련 메서드들
    void enableGPUAcceleration(bool enable);
    void setThreadCount(unsigned int count);
    
    // ✅ 추가: 가중치 맵 처리 관련 함수들
    MStatus processWeightMap(const MObject& weightMap, const MMatrix& transform, double strength);
    bool validateWeightMap(const MObject& weightMap) const;
    
    // ✅ 추가: 영향력 혼합 관련 함수들
    MPoint blendAllInfluences(const MPoint& modelPoint, 
                             const std::vector<OffsetPrimitive>& primitives,
                             const offsetCurveControlParams& params) const;
    void optimizeInfluenceBlending(std::vector<OffsetPrimitive>& primitives,
                                  const MPoint& modelPoint) const;
    
    // ✅ 추가: 공간적 보간 관련 함수들
    MPoint applySpatialInterpolation(const MPoint& modelPoint,
                                    const MDagPath& curvePath,
                                    double influenceRadius) const;
    void setSpatialInterpolationQuality(double quality);
    void setSpatialInterpolationSmoothness(double smoothness);
    
    // ✅ 수정: 특허 기반 볼륨 보존 시스템
    double calculateVolumePreservationFactor(const OffsetPrimitive& primitive,
                                           double curvature) const;
    
    bool checkSelfIntersection(const OffsetPrimitive& primitive,
                              double curvature) const;
    
    MVector applySelfIntersectionPrevention(const MVector& deformedOffset,
                                           const OffsetPrimitive& primitive,
                                           double curvature) const;
    
    // 🔬 곡률 계산 함수 (특허 수학 공식)
    double calculateCurvatureAtPoint(const MDagPath& curvePath, double paramU) const;
    
    MVector applyArtistControls(const MVector& bindOffsetLocal,
                               const MVector& currentTangent,
                               const MVector& currentNormal,
                               const MVector& currentBinormal,
                               const MDagPath& curvePath,
                               double& paramU,
                               const offsetCurveControlParams& params) const;

private:
    // ✅ 리팩토링: Repository 패턴 적용
    // === OCD 알고리즘: Repository 기반 데이터 관리 ===
    offsetCurveOffsetMode mOffsetMode;                          // Arc vs B-spline 모드
    
    // Repository 기반 데이터 관리 (기존 직접 데이터 대체)
    std::unique_ptr<ICurveRepository> mCurveRepo;               // 곡선 데이터 Repository
    std::unique_ptr<IBindingRepository> mBindingRepo;           // 바인딩 데이터 Repository
    
    // Service Layer (비즈니스 로직 분리)
    std::unique_ptr<CurveBindingService> mBindingService;       // 곡선 바인딩 서비스
    std::unique_ptr<DeformationService> mDeformationService;    // 변형 처리 서비스
    
    // ✅ 추가: DataFlowController (데이터 흐름 관리)
    std::unique_ptr<IDataFlowController> mDataFlowController;   // 데이터 흐름 제어기
    
    // === 성능 및 기타 ===
    bool mUseParallelComputation;                               // 병렬 처리 플래그
    MPointArray mPoseTargetPoints;                              // 포즈 타겟 (선택사항)
    
    // ✅ 추가: 특허의 고급 시스템들
    BindRemappingSystem mBindRemapping;                         // Bind Remapping 시스템
    PoseSpaceDeformationSystem mPoseSpaceDeformation;           // Pose Space Deformation 시스템
    AdaptiveSubdivisionSystem mAdaptiveSubdivision;             // Adaptive Subdivision 시스템
    
    // ✅ 추가: Strategy Pattern Context
    InfluencePrimitiveContext mStrategyContext;                 // Strategy Context
    
    // ✅ 추가: 가중치 맵 처리 시스템
    WeightMapProcessor mWeightMapProcessor;                     // 가중치 맵 처리기
    
    // ✅ 추가: 영향력 혼합 시스템
    InfluenceBlendingSystem mInfluenceBlending;                 // 영향력 혼합 시스템
    
    // ✅ 추가: 공간적 보간 시스템
    SpatialInterpolationSystem mSpatialInterpolation;           // 공간적 보간 시스템
    
    // 🚨 제거: mOffsetPrimitives는 mVertexData 내부에 포함됨
    // std::vector<OffsetPrimitive> mOffsetPrimitives; // 중복 제거!
    
    // 🚨 제거: mSkeletonJoints는 PoseSpaceDeformationSystem에 포함됨
    // std::vector<MObject> mSkeletonJoints; // 중복 제거!
};

#endif // OFFSETCURVEALGORITHM_H