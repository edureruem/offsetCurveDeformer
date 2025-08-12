/**
 * offsetCurveAlgorithm.h
 * OCD 핵심 알고리즘 구현
 * 소니 특허(US8400455) 기반으로 개선
 */

#ifndef OFFSETCURVEALGORITHM_H
#define OFFSETCURVEALGORITHM_H

#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MVector.h>
#include <maya/MVectorArray.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MDagPath.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MThreadPool.h>
#include <vector>
#include <map>
#include <memory>

#include "offsetCurveControlParams.h"

// ✅ 추가: Strategy Pattern 인터페이스
class InfluencePrimitiveStrategy {
public:
    virtual ~InfluencePrimitiveStrategy() = default;
    
    // 핵심 연산들
    virtual MStatus calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                       MVector& tangent, MVector& normal, MVector& binormal) const = 0;
    
    virtual MStatus getPointAtParam(const MDagPath& curvePath, double paramU,
                                   MPoint& point) const = 0;
    
    virtual MStatus getNormalAtParam(const MDagPath& curvePath, double paramU,
                                    MVector& normal) const = 0;
    
    virtual MStatus getTangentAtParam(const MDagPath& curvePath, double paramU,
                                     MVector& tangent) const = 0;
    
    virtual double getCurvatureAtParam(const MDagPath& curvePath, double paramU) const = 0;
    
    // 전략별 고유 기능
    virtual std::string getStrategyName() const = 0;
    virtual bool isOptimizedForCurveType(const MDagPath& curvePath) const = 0;
};

// ✅ 추가: Arc Segment Strategy 구현
class ArcSegmentStrategy : public InfluencePrimitiveStrategy {
public:
    ArcSegmentStrategy();
    ~ArcSegmentStrategy() override = default;
    
    // 핵심 연산 구현
    MStatus calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                MVector& tangent, MVector& normal, MVector& binormal) const override;
    
    MStatus getPointAtParam(const MDagPath& curvePath, double paramU,
                           MPoint& point) const override;
    
    MStatus getNormalAtParam(const MDagPath& curvePath, double paramU,
                            MVector& normal) const override;
    
    MStatus getTangentAtParam(const MDagPath& curvePath, double paramU,
                             MVector& tangent) const override;
    
    double getCurvatureAtParam(const MDagPath& curvePath, double paramU) const override;
    
    // Arc Segment 전용 기능
    std::string getStrategyName() const override { return "ArcSegment"; }
    bool isOptimizedForCurveType(const MDagPath& curvePath) const override;
    
private:
    // Arc Segment 최적화를 위한 헬퍼 함수들
    MStatus calculateFrenetFrameOptimized(const MDagPath& curvePath, double paramU,
                                         MVector& tangent, MVector& normal, MVector& binormal) const;
    bool isLinearSegment(const MDagPath& curvePath, double paramU) const;
    double calculateArcRadius(const MDagPath& curvePath, double paramU) const;
};

// ✅ 추가: B-Spline Strategy 구현
class BSplineStrategy : public InfluencePrimitiveStrategy {
public:
    BSplineStrategy();
    ~BSplineStrategy() override = default;
    
    // 핵심 연산 구현
    MStatus calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                MVector& tangent, MVector& normal, MVector& binormal) const override;
    
    MStatus getPointAtParam(const MDagPath& curvePath, double paramU,
                           MPoint& point) const override;
    
    MStatus getNormalAtParam(const MDagPath& curvePath, double paramU,
                            MVector& normal) const override;
    
    MStatus getTangentAtParam(const MDagPath& curvePath, double paramU,
                             MVector& tangent) const override;
    
    double getCurvatureAtParam(const MDagPath& curvePath, double paramU) const override;
    
    // B-Spline 전용 기능
    std::string getStrategyName() const override { return "BSpline"; }
    bool isOptimizedForCurveType(const MDagPath& curvePath) const override;
    
private:
    // B-Spline 최적화를 위한 헬퍼 함수들
    MStatus calculateFrenetFrameAccurate(const MDagPath& curvePath, double paramU,
                                        MVector& tangent, MVector& normal, MVector& binormal) const;
    double calculateCurvatureAccurate(const MDagPath& curvePath, double paramU) const;
    MStatus calculateHigherOrderDerivatives(const MDagPath& curvePath, double paramU,
                                          MVector& firstDeriv, MVector& secondDeriv) const;
};

// ✅ 추가: Strategy Factory
class InfluencePrimitiveStrategyFactory {
public:
    static std::unique_ptr<InfluencePrimitiveStrategy> createStrategy(offsetCurveOffsetMode mode);
    static std::unique_ptr<InfluencePrimitiveStrategy> createOptimalStrategy(const MDagPath& curvePath);
    
private:
    static bool isArcSegmentOptimal(const MDagPath& curvePath);
    static bool isBSplineOptimal(const MDagPath& curvePath);
};

// ✅ 추가: Strategy Context
class InfluencePrimitiveContext {
public:
    InfluencePrimitiveContext();
    ~InflucePrimitiveContext();
    
    // Strategy 설정
    void setStrategy(std::unique_ptr<InfluencePrimitiveStrategy> strategy);
    void setStrategy(offsetCurveOffsetMode mode);
    void setOptimalStrategy(const MDagPath& curvePath);
    
    // Strategy를 통한 연산 실행
    MStatus calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                MVector& tangent, MVector& normal, MVector& binormal) const;
    
    MStatus getPointAtParam(const MDagPath& curvePath, double paramU,
                           MPoint& point) const;
    
    MStatus getNormalAtParam(const MDagPath& curvePath, double paramU,
                            MVector& normal) const;
    
    MStatus getTangentAtParam(const MDagPath& curvePath, double paramU,
                             MVector& tangent) const;
    
    double getCurvatureAtParam(const MDagPath& curvePath, double paramU) const;
    
    // 현재 Strategy 정보
    std::string getCurrentStrategyName() const;
    bool hasStrategy() const;
    
private:
    std::unique_ptr<InfluencePrimitiveStrategy> mStrategy;
};

// ✅ 추가: 가중치 맵 처리 시스템
class WeightMapProcessor {
public:
    WeightMapProcessor();
    ~WeightMapProcessor();
    
    // 가중치 맵에서 가중치 값 추출
    double getWeight(const MPoint& modelPoint,
                    const MObject& weightMap,
                    const MMatrix& transform) const;
    
    // 여러 가중치 맵의 가중치 값들을 조합
    double combineWeights(const MPoint& modelPoint,
                         const std::vector<MObject>& weightMaps,
                         const std::vector<MMatrix>& transforms) const;
    
    // 가중치 맵 유효성 검사
    bool isValidWeightMap(const MObject& weightMap) const;
    
    // 가중치 맵 정보 가져오기
    bool getWeightMapInfo(const MObject& weightMap,
                         int& width, int& height,
                         std::string& format) const;
    
private:
    // 이중선형 보간 구현
    double sampleWeightWithBilinearInterpolation(const MImage& image,
                                               float u, float v) const;
    
    // 픽셀 값 추출
    double getPixelValue(const MImage& image, int x, int y) const;
    
    // 모델 포인트를 UV 좌표로 변환
    MPoint transformPointToUV(const MPoint& modelPoint,
                             const MMatrix& transform) const;
    
    // UV 좌표를 텍스처 좌표로 변환
    void convertUVToTextureCoords(double u, double v,
                                 float& texU, float& texV) const;
    
    // 가중치 값 정규화
    double normalizeWeight(double weight) const;
};

// ✅ 추가: 영향력 혼합 시스템
class InfluenceBlendingSystem {
public:
    InfluenceBlendingSystem();
    ~InfluenceBlendingSystem();
    
    // 여러 Influence Primitive의 영향력을 혼합
    MPoint blendInfluences(const MPoint& modelPoint,
                          const std::vector<OffsetPrimitive>& primitives,
                          const std::vector<MDagPath>& influenceCurves,
                          const offsetCurveControlParams& params) const;
    
    // 개별 Influence Primitive의 영향력 계산
    MPoint calculateInfluenceContribution(const MPoint& modelPoint,
                                        const OffsetPrimitive& primitive,
                                        const MDagPath& curvePath,
                                        const offsetCurveControlParams& params) const;
    
    // 영향력 혼합 품질 최적화
    void optimizeBlendingQuality(std::vector<OffsetPrimitive>& primitives,
                                const MPoint& modelPoint) const;
    
    // 영향력 충돌 해결
    void resolveInfluenceConflicts(std::vector<OffsetPrimitive>& primitives) const;
    
private:
    // 기본 영향력 계산 (가우시안 함수)
    double calculateBaseInfluence(const MPoint& modelPoint,
                                const OffsetPrimitive& primitive,
                                const MDagPath& curvePath) const;
    
    // 오프셋 위치 계산
    MPoint calculateOffsetPosition(const MPoint& modelPoint,
                                 const OffsetPrimitive& primitive,
                                 const MDagPath& curvePath) const;
    
    // 영향력 가중치 정규화
    void normalizeInfluenceWeights(std::vector<OffsetPrimitive>& primitives) const;
    
    // 영향력 품질 평가
    double evaluateInfluenceQuality(const std::vector<OffsetPrimitive>& primitives) const;
    
    // 영향력 충돌 감지
    bool detectInfluenceConflict(const OffsetPrimitive& primitive1,
                               const OffsetPrimitive& primitive2) const;
    
    // 영향력 충돌 해결 전략
    void applyConflictResolutionStrategy(OffsetPrimitive& primitive1,
                                       OffsetPrimitive& primitive2) const;
};

// ✅ 추가: 공간적 보간 시스템
class SpatialInterpolationSystem {
public:
    SpatialInterpolationSystem();
    ~SpatialInterpolationSystem();
    
    // B-spline 곡선을 따른 공간적 보간
    MPoint interpolateAlongBSpline(const MPoint& modelPoint,
                                   const MDagPath& curvePath,
                                   double influenceRadius) const;
    
    // Arc-segment 곡선을 따른 공간적 보간
    MPoint interpolateAlongArcSegment(const MPoint& modelPoint,
                                     const MDagPath& curvePath,
                                     double influenceRadius) const;
    
    // 곡선 타입에 따른 자동 보간 방식 선택
    MPoint interpolateAlongCurve(const MPoint& modelPoint,
                                const MDagPath& curvePath,
                                double influenceRadius,
                                offsetCurveOffsetMode curveType) const;
    
    // 공간적 변화 계산
    double calculateSpatialVariation(const MDagPath& curvePath, double paramU) const;
    
    // 거리 기반 영향력 계산
    double calculateDistanceInfluence(double distance, double radius) const;
    
    // 공간적 오프셋 계산
    MVector calculateSpatialOffset(const MDagPath& curvePath,
                                  double paramU,
                                  double spatialVariation) const;
    
    // 보간 품질 설정
    void setInterpolationQuality(double quality);
    void setSmoothnessFactor(double factor);
    void setMaxInterpolationSteps(int steps);
    
private:
    // 보간 품질 설정
    double mInterpolationQuality;    // 보간 품질 (0.0 ~ 1.0)
    double mSmoothnessFactor;        // 부드러움 계수 (0.0 ~ 1.0)
    int mMaxInterpolationSteps;      // 최대 보간 단계 수
    
    // 공간적 변화 계산 헬퍼 함수들
    double calculateCurvatureBasedVariation(const MDagPath& curvePath, double paramU) const;
    double calculateTorsionBasedVariation(const MDagPath& curvePath, double paramU) const;
    double calculateParameterBasedVariation(double paramU) const;
    
    // 이징 함수들
    double smoothstep(double edge0, double edge1, double x) const;
    double smootherstep(double edge0, double edge1, double x) const;
    double easeInOutCubic(double t) const;
    
    // 보간 단계별 계산
    MPoint calculateInterpolationStep(const MPoint& startPoint,
                                    const MPoint& endPoint,
                                    double t,
                                    const MVector& spatialOffset) const;
    
    // 곡선 구간 분석
    std::vector<std::pair<double, double>> analyzeCurveSegments(const MDagPath& curvePath) const;
    
    // 보간 경로 최적화
    std::vector<MPoint> optimizeInterpolationPath(const std::vector<MPoint>& path) const;
};

// Offset Curve 오프셋 방식 정의
enum offsetCurveOffsetMode {
    ARC_SEGMENT = 0,    // 아크 세그먼트 방식
    B_SPLINE = 1        // B-스플라인 방식
};

// 오프셋 프리미티브: 최소한의 수학적 파라미터만 저장 (실제 곡선 생성 안 함)
struct OffsetPrimitive {
    // === 핵심: 4개 값만 저장 ===
    int influenceCurveIndex;             // 영향 곡선 인덱스 (MDagPath 참조용)
    double bindParamU;                   // 바인드 시점의 곡선 파라미터 u
    MVector bindOffsetLocal;             // 바인드 시점의 로컬 오프셋 벡터 (T,N,B 좌표계)
    double weight;                       // 영향 가중치
    
    // ✅ 추가: 가중치 맵 시스템
    MObject weightMap;                   // 가중치 맵 (Maya 텍스처 객체)
    MMatrix weightMapTransform;          // 가중치 맵 변환 행렬 (UV 좌표 변환용)
    double weightMapStrength;            // 가중치 맵 강도 (0.0 ~ 2.0)
    bool useWeightMap;                   // 가중치 맵 사용 여부
    
    OffsetPrimitive() : 
        influenceCurveIndex(-1), bindParamU(0.0), weight(0.0),
        weightMapStrength(1.0), useWeightMap(false) {
        weightMapTransform = MMatrix::identity;
    }
};

// 🎯 적응형 Arc Segment 구조체
struct ArcSegment {
    double startParamU;                  // 세그먼트 시작 파라미터
    double endParamU;                    // 세그먼트 끝 파라미터
    MPoint center;                       // 원의 중심 (직선인 경우 무시)
    double radius;                       // 원의 반지름 (직선인 경우 0)
    double totalAngle;                   // 총 호의 각도 (직선인 경우 0)
    bool isLinear;                       // 직선 세그먼트 여부
    double curvatureMagnitude;           // 곡률의 크기
    
    ArcSegment() : 
        startParamU(0.0), endParamU(1.0), radius(0.0), 
        totalAngle(0.0), isLinear(true), curvatureMagnitude(0.0) {}
};

// ✅ 추가: Bind Remapping 시스템
class BindRemappingSystem {
private:
    // 파라미터별 정점 그룹화
    std::map<double, std::vector<int>> mBindParameterToVertices;
    // 정점별 바인드 파라미터 매핑
    std::map<int, double> mVertexToBindParameter;
    
public:
    BindRemappingSystem();
    ~BindRemappingSystem();
    
    // 바인딩 시점에 파라미터별 정점 그룹화
    void groupVerticesByBindParameter(const std::vector<OffsetPrimitive>& primitives);
    
    // 역방향 바인드 리매핑 적용
    MVector applyInvertedBindRemapping(const MVector& offset, 
                                      double bindParamU, 
                                      double currentParamU);
    
    // 바인드 파라미터 충돌 해결
    double resolveBindParameterConflict(double paramU, int vertexIndex);
    
    // 정점 그룹 정보 가져오기
    const std::vector<int>& getVerticesAtParameter(double paramU) const;
    
    // 정점의 바인드 파라미터 가져오기
    double getBindParameterForVertex(int vertexIndex) const;
};

// ✅ 추가: Pose Space Deformation 시스템
class PoseSpaceDeformationSystem {
private:
    // 스켈레톤 관절들
    std::vector<MDagPath> mSkeletonJoints;
    // 관절별 오프셋
    std::map<int, std::vector<MVector>> mJointOffsets;
    // 관절별 가중치
    std::map<int, double> mJointWeights;
    
public:
    PoseSpaceDeformationSystem();
    ~PoseSpaceDeformationSystem();
    
    // 스켈레톤 관절 추가
    void addSkeletonJoint(const MDagPath& jointPath);
    
    // 관절별 로컬 오프셋 설정
    void setJointLocalOffset(int jointIndex, const MVector& offset);
    
    // 관절별 가중치 설정
    void setJointWeight(int jointIndex, double weight);
    
    // 스켈레톤 기반 포즈 공간 변형 계산
    MVector calculatePoseSpaceOffset(const MPoint& vertex, 
                                   int jointIndex,
                                   const MMatrix& jointTransform);
    
    // 모든 관절의 포즈 공간 변형 적용
    MVector applyAllPoseSpaceOffsets(const MPoint& vertex);
};

// ✅ 추가: Adaptive Subdivision 시스템
class AdaptiveSubdivisionSystem {
private:
    double mCurvatureThreshold;  // 곡률 임계값
    double mMaxSegmentLength;    // 최대 세그먼트 길이
    double mMinSegmentLength;    // 최소 세그먼트 길이
    
public:
    AdaptiveSubdivisionSystem();
    ~AdaptiveSubdivisionSystem();
    
    // 곡률 기반 적응적 분할
    std::vector<ArcSegment> subdivideAdaptively(const MDagPath& curvePath);
    
    // 절차적 세그먼트 생성
    ArcSegment generateArcSegment(const MPoint& start, const MPoint& end, 
                                const MVector& curvature);
    
    // 최적 세그먼트 길이 계산
    double calculateOptimalSegmentLength(double curvature) const;
    
    // 설정값 변경
    void setCurvatureThreshold(double threshold);
    void setMaxSegmentLength(double maxLength);
    void setMinSegmentLength(double minLength);
};

// 정점 변형 데이터 (단순화)
struct VertexDeformationData {
    unsigned int vertexIndex;                    // 정점 인덱스
    MPoint bindPosition;                         // 바인드 시점의 위치
    std::vector<OffsetPrimitive> offsetPrimitives; // 핵심: 수학적 파라미터만!
    
    VertexDeformationData() : vertexIndex(0) {}
};

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
    
    // ✅ 추가: 새로운 시스템 초기화 및 설정
    void initializeBindRemapping();
    void initializePoseSpaceDeformation();
    void initializeAdaptiveSubdivision();
    
    // Bind Remapping 적용
    MStatus applyBindRemappingToPrimitives();
    
    // Pose Space Deformation 적용
    MVector applyPoseSpaceDeformation(const MPoint& vertex, int vertexIndex);
    
    // Adaptive Subdivision 적용
    std::vector<ArcSegment> getAdaptiveSegments(const MDagPath& curvePath);

private:
    // 포즈 블렌딩 적용
    MPoint applyPoseBlending(const MPoint& deformedPoint, 
                           unsigned int vertexIndex,
                           double blendWeight);
    // === OCD 알고리즘: 최소한의 데이터만 ===
    offsetCurveOffsetMode mOffsetMode;                          // Arc vs B-spline 모드
    std::vector<MDagPath> mInfluenceCurvePaths;                 // 영향 곡선 경로들 (데이터 저장 안 함!)
    std::vector<VertexDeformationData> mVertexData;             // 정점별 오프셋 프리미티브들
    
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
    
    // === 실시간 계산 함수들 (캐싱 없음!) ===
    MStatus calculateFrenetFrameOnDemand(const MDagPath& curvePath, 
                                        double paramU,
                                        MVector& tangent,
                                        MVector& normal, 
                                        MVector& binormal) const;
    
    // 🚀 Arc Segment 모드: 고성능 실시간 계산 (3-5배 빠름!)
    MStatus calculateFrenetFrameArcSegment(const MDagPath& curvePath,
                                          double paramU,
                                          MVector& tangent,
                                          MVector& normal,
                                          MVector& binormal) const;
    
    // 🔬 고차 미분을 이용한 정확한 곡률 계산
    MStatus calculateCurvatureVector(const MDagPath& curvePath,
                                    double paramU,
                                    MVector& curvature,
                                    double& curvatureMagnitude) const;
    
    // 🎯 적응형 Arc Segment 세분화
    std::vector<ArcSegment> subdivideByKappa(const MDagPath& curvePath,
                                            double maxCurvatureError = 0.01) const;
    
    // 🚀 병렬 처리용 헬퍼 함수
    void processVertexDeformation(int vertexIndex, 
                                 MPointArray& points,
                                 const offsetCurveControlParams& params) const;
    
    // 제거됨: 적응형 변형 처리 (일관된 결과를 위해 제거)
    
    // 헬퍼 함수들
    void mergeAdjacentSegments(std::vector<ArcSegment>& segments,
                              double maxCurvatureError) const;
    
    // 🔥 GPU 가속 지원 (CUDA/OpenCL)
    #ifdef CUDA_ENABLED
    void processVertexDeformationGPU(MPointArray& points,
                                    const offsetCurveControlParams& params) const;
    #endif
    
    // 제거됨: 적응형 세밀도 조절 (예측 불가능한 결과 방지)
    
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
    
    MStatus getPointAtParamWithStrategy(const MDagPath& curvePath, double paramU,
                                       MPoint& point) const;
    
    double getCurvatureAtParamWithStrategy(const MDagPath& curvePath, double paramU) const;
    
    // ✅ 추가: 가중치 맵 관련 함수들
    double getEffectiveWeight(const OffsetPrimitive& primitive, const MPoint& modelPoint) const;
    void setWeightMapForPrimitive(OffsetPrimitive& primitive, const MObject& weightMap, 
                                 const MMatrix& transform, double strength);
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
};

#endif // OFFSETCURVEALGORITHM_H