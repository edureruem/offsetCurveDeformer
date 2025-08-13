/**
 * offsetCurveSystems.h
 * OCD 시스템 클래스들 정의
 * 소니 특허(US8400455) 기반으로 개선
 */

#ifndef OFFSETCURVESYSTEMS_H
#define OFFSETCURVESYSTEMS_H

// 공통 타입 헤더
#include "offsetCurveTypes.h"

// Maya 헤더들
#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MDagPath.h>
#include <maya/MObject.h>
#include <maya/MStatus.h>
#include <maya/MFnDagNode.h>

// C++ 표준 라이브러리
#include <vector>
#include <map>
#include <memory>
#include <string>

// ✅ Bind Remapping 시스템 (특허 기술 보존)
class BindRemappingSystem {
public:
    BindRemappingSystem();
    ~BindRemappingSystem();
    
    MStatus applyBindRemapping(std::vector<OffsetPrimitive>& primitives);
    void setRemappingStrength(double strength);
    
private:
    double mRemappingStrength;
};

// ✅ Pose Space Deformation 시스템 (특허 기술 보존)
class PoseSpaceDeformationSystem {
public:
    PoseSpaceDeformationSystem();
    ~PoseSpaceDeformationSystem();
    
    MVector calculatePoseSpaceOffset(const MPoint& vertex, int jointIndex, 
                                    const MMatrix& jointTransform) const;
    MVector applyAllPoseSpaceOffsets(const MPoint& vertex) const;
    MMatrix getJointTransform(int jointIndex) const;
    
    // ✅ 추가: 스켈레톤 조인트 초기화
    void initializeSkeletonJoints(const std::vector<MObject>& skeletonJoints);
    
private:
    std::vector<MObject> mSkeletonJoints;
    std::map<int, double> mJointWeights;
};

// ✅ Adaptive Subdivision 시스템 (특허 기술 보존)
class AdaptiveSubdivisionSystem {
public:
    AdaptiveSubdivisionSystem();
    ~AdaptiveSubdivisionSystem();
    
    std::vector<ArcSegment> generateArcSegments(const MDagPath& curvePath, 
                                                double maxCurvatureError);
    void mergeAdjacentSegments(std::vector<ArcSegment>& segments,
                               double maxCurvatureError);
    
private:
    double mMaxCurvatureError;
};

// ✅ Weight Map Processor (특허 기술 보존)
class WeightMapProcessor {
public:
    WeightMapProcessor();
    ~WeightMapProcessor();
    
    MStatus processWeightMap(const MObject& weightMap, const MMatrix& transform, double strength);
    bool isValidWeightMap(const MObject& weightMap) const;
    
private:
    double mStrength;
    MMatrix mTransform;
};

// ✅ Influence Blending System (특허 기술 보존)
class InfluenceBlendingSystem {
public:
    InfluenceBlendingSystem();
    ~InfluenceBlendingSystem();
    
    MPoint blendAllInfluences(const MPoint& modelPoint, 
                              const std::vector<OffsetPrimitive>& primitives,
                              double blendStrength);
    void optimizeInfluenceBlending(std::vector<OffsetPrimitive>& primitives,
                                  const MPoint& modelPoint);
    
private:
    double mBlendStrength;
};

// ✅ Spatial Interpolation System (특허 기술 보존)
class SpatialInterpolationSystem {
public:
    SpatialInterpolationSystem();
    ~SpatialInterpolationSystem();
    
    MPoint applySpatialInterpolation(const MPoint& modelPoint,
                                     const MDagPath& curvePath,
                                     double influenceRadius);
    void setInterpolationQuality(double quality);
    void setInterpolationSmoothness(double smoothness);
    
private:
    double mQuality;
    double mSmoothness;
    double mInfluenceRadius;
};

// ✅ Influence Primitive Strategy (전략 패턴)
class InfluencePrimitiveStrategy {
public:
    virtual ~InfluencePrimitiveStrategy() = default;
    
    // 가상 함수들
    virtual MStatus calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                        MVector& tangent, MVector& normal, MVector& binormal) const = 0;
    virtual MStatus getPointAtParam(const MDagPath& curvePath, double paramU, MPoint& point) const = 0;
    virtual MStatus getNormalAtParam(const MDagPath& curvePath, double paramU, MVector& normal) const = 0;
    virtual MStatus getTangentAtParam(const MDagPath& curvePath, double paramU, MVector& tangent) const = 0;
    virtual double getCurvatureAtParam(const MDagPath& curvePath, double paramU) const = 0;
    virtual std::string getStrategyName() const = 0;
    virtual bool isOptimizedForCurveType(const MDagPath& curvePath) const = 0;
};

// ✅ Arc Segment Strategy (특허 기술 보존)
class ArcSegmentStrategy : public InfluencePrimitiveStrategy {
public:
    ArcSegmentStrategy();
    ~ArcSegmentStrategy();
    
    // 전략 구현
    MStatus calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                 MVector& tangent, MVector& normal, MVector& binormal) const override;
    MStatus getPointAtParam(const MDagPath& curvePath, double paramU, MPoint& point) const override;
    MStatus getNormalAtParam(const MDagPath& curvePath, double paramU, MVector& normal) const override;
    MStatus getTangentAtParam(const MDagPath& curvePath, double paramU, MVector& tangent) const override;
    double getCurvatureAtParam(const MDagPath& curvePath, double paramU) const override;
    std::string getStrategyName() const override;
    bool isOptimizedForCurveType(const MDagPath& curvePath) const override;
};

// ✅ B-Spline Strategy (특허 기술 보존)
class BSplineStrategy : public InfluencePrimitiveStrategy {
public:
    BSplineStrategy();
    ~BSplineStrategy();
    
    // 전략 구현
    MStatus calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                 MVector& tangent, MVector& normal, MVector& binormal) const override;
    MStatus getPointAtParam(const MDagPath& curvePath, double paramU, MPoint& point) const override;
    MStatus getNormalAtParam(const MDagPath& curvePath, double paramU, MVector& normal) const override;
    MStatus getTangentAtParam(const MDagPath& curvePath, double paramU, MVector& tangent) const override;
    double getCurvatureAtParam(const MDagPath& curvePath, double paramU) const override;
    std::string getStrategyName() const override;
    bool isOptimizedForCurveType(const MDagPath& curvePath) const override;
};

// ✅ Strategy Factory (특허 기술 보존)
class InfluencePrimitiveStrategyFactory {
public:
    static std::unique_ptr<InfluencePrimitiveStrategy> createStrategy(offsetCurveOffsetMode mode);
    static std::unique_ptr<InfluencePrimitiveStrategy> createOptimalStrategy(const MDagPath& curvePath);
};

// ✅ Strategy Context (특허 기술 보존)
class InfluencePrimitiveContext {
public:
    InfluencePrimitiveContext();
    ~InfluencePrimitiveContext();
    
    MStatus calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                 MVector& tangent, MVector& normal, MVector& binormal) const;
    MStatus getPointAtParam(const MDagPath& curvePath, double paramU, MPoint& point) const;
    double getCurvatureAtParam(const MDagPath& curvePath, double paramU) const;
    
    void setOptimalStrategy(const MDagPath& curvePath);
    
private:
    std::unique_ptr<InfluencePrimitiveStrategy> mCurrentStrategy;
};

// ✅ Repository 구현 클래스들
class CurveRepository : public ICurveRepository {
public:
    CurveRepository();
    ~CurveRepository();
    
    // ICurveRepository 인터페이스 구현
    void addCurve(const MDagPath& curvePath) override;
    void removeCurve(int index) override;
    void clearCurves() override;
    
    const std::vector<MDagPath>& getAllCurves() const override;
    MDagPath getCurve(int index) const override;
    int getCurveCount() const override;
    
    bool hasCurve(const MDagPath& curvePath) const override;
    bool isValidCurve(const MDagPath& curvePath) const override;
    
    // 인터페이스 구현 (추상 클래스 오류 해결)
    void updateCurveValidityCache(const MDagPath& curvePath, bool isValid) override;
    
private:
    std::vector<MDagPath> mCurves;
    std::map<std::string, bool> mCurveValidityCache;  // 곡선 유효성 캐시 (문자열 키 사용)
};

class BindingRepository : public IBindingRepository {
public:
    BindingRepository();
    ~BindingRepository();
    
    // IBindingRepository 인터페이스 구현
    void addVertexBinding(int vertexIndex, const MPoint& bindPosition) override;
    void addOffsetPrimitive(int vertexIndex, const OffsetPrimitive& primitive) override;
    void removeVertexBinding(int vertexIndex) override;
    void clearBindings() override;
    
    const std::vector<VertexDeformationData>& getAllVertexBindings() const override;
    VertexDeformationData& getVertexBinding(int vertexIndex) override;
    const std::vector<OffsetPrimitive>& getVertexPrimitives(int vertexIndex) const override;
    int getBindingCount() const override;
    
    bool hasVertexBinding(int vertexIndex) const override;
    bool isValidBinding(int vertexIndex) const override;
    
private:
    std::vector<VertexDeformationData> mVertexBindings;
    std::map<int, int> mVertexIndexMap;  // vertexIndex -> vector index 매핑
};

// ✅ Service Layer 클래스들 (비즈니스 로직 담당)
class CurveBindingService {
public:
    CurveBindingService(ICurveRepository* curveRepo, IBindingRepository* bindingRepo);
    ~CurveBindingService();
    
    // 곡선 바인딩 관련 비즈니스 로직
    MStatus bindCurveToVertex(int vertexIndex, const MDagPath& curvePath, double falloffRadius);
    MStatus unbindCurveFromVertex(int vertexIndex, int curveIndex);
    MStatus updateBinding(int vertexIndex, const OffsetPrimitive& primitive);
    
    // 바인딩 계산 및 검증
    std::vector<OffsetPrimitive> calculateBindings(int vertexIndex) const;
    bool validateBinding(int vertexIndex) const;
    double calculateBindingStrength(int vertexIndex, const MDagPath& curvePath) const;
    
    // 바인딩 상태 조회
    bool isVertexBound(int vertexIndex) const;
    int getBoundCurveCount(int vertexIndex) const;
    
private:
    ICurveRepository* mCurveRepo;        // Repository 인터페이스 (소유권 없음)
    IBindingRepository* mBindingRepo;    // Repository 인터페이스 (소유권 없음)
    
    // 바인딩 계산 헬퍼 메서드
    double calculateFalloffDistance(const MPoint& vertexPos, const MDagPath& curvePath) const;
    bool isCurveInRange(const MDagPath& curvePath, const MPoint& vertexPos, double falloffRadius) const;
};

// ✅ DeformationService 클래스 (변형 알고리즘 핵심 비즈니스 로직)
class DeformationService {
public:
    DeformationService(ICurveRepository* curveRepo, IBindingRepository* bindingRepo);
    ~DeformationService();
    
    // 메인 변형 처리
    MStatus processDeformation(const MPointArray& inputPoints, MPointArray& outputPoints);
    
    // 개별 정점 변형 처리
    MStatus deformVertex(int vertexIndex, const MPoint& inputPoint, MPoint& outputPoint);
    
    // 변형 파라미터 설정
    void setDeformationParameters(double strength, double falloffRadius, bool useParallel);
    
    // 변형 품질 설정
    void setDeformationQuality(double quality, double smoothness);
    
    // 에러 처리 및 검증
    bool validateDeformationParameters() const;
    MStatus getLastError() const;
    
private:
    ICurveRepository* mCurveRepo;        // Repository 인터페이스 (소유권 없음)
    IBindingRepository* mBindingRepo;    // Repository 인터페이스 (소유권 없음)
    
    // 변형 파라미터
    double mDeformationStrength;
    double mFalloffRadius;
    bool mUseParallelComputation;
    double mDeformationQuality;
    double mSmoothness;
    
    // 에러 상태
    mutable MStatus mLastError;
    
    // 내부 헬퍼 메서드들
    MStatus calculateVertexDeformation(int vertexIndex, const MPoint& inputPoint, 
                                      MPoint& outputPoint, const std::vector<OffsetPrimitive>& primitives);
    double calculateInfluenceWeight(const MPoint& vertexPos, const MDagPath& curvePath, 
                                   const OffsetPrimitive& primitive) const;
    MVector calculateOffsetVector(const MDagPath& curvePath, const OffsetPrimitive& primitive) const;
    MStatus applyFrenetFrameDeformation(const MPoint& inputPoint, const MVector& offset, 
                                       MPoint& outputPoint) const;
};

// ✅ DataFlowController 구현 클래스 (데이터 흐름 관리)
class DataFlowController : public IDataFlowController {
public:
    DataFlowController(ICurveRepository* curveRepo, 
                      IBindingRepository* bindingRepo,
                      CurveBindingService* bindingService,
                      DeformationService* deformationService);
    ~DataFlowController();
    
    // IDataFlowController 인터페이스 구현
    MStatus initializeDataFlow() override;
    MStatus processDataFlow() override;
    MStatus validateDataFlow() override;
    
    MStatus synchronizeRepositories() override;
    MStatus transferDataBetweenServices() override;
    
    bool isDataFlowValid() const override;
    MStatus getDataFlowStatus() const override;
    
    MStatus handleDataFlowError(const MStatus& error) override;
    MStatus recoverDataFlow() override;
    
    // 추가 데이터 흐름 제어 메서드들
    MStatus optimizeDataFlow();
    MStatus monitorDataFlowPerformance();
    MStatus cleanupDataFlow();
    
private:
    ICurveRepository* mCurveRepo;           // Repository 인터페이스 (소유권 없음)
    IBindingRepository* mBindingRepo;       // Repository 인터페이스 (소유권 없음)
    CurveBindingService* mBindingService;   // Service 인터페이스 (소유권 없음)
    DeformationService* mDeformationService; // Service 인터페이스 (소유권 없음)
    
    // 데이터 흐름 상태
    mutable MStatus mDataFlowStatus;
    bool mIsDataFlowValid;
    bool mIsInitialized;
    
    // 데이터 흐름 성능 모니터링
    double mLastProcessingTime;
    unsigned int mProcessedDataCount;
    
    // 내부 헬퍼 메서드들
    MStatus validateRepositoryConnections() const;
    MStatus validateServiceConnections() const;
    MStatus performDataValidation() const;
    MStatus updateDataFlowStatus(MStatus newStatus);
};

#endif // OFFSETCURVESYSTEMS_H
