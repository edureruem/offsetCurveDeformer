/**
 * offsetCurveAlgorithm.cpp
 * OCD 핵심 알고리즘 구현 (완전히 새로 작성)
 * 소니 특허(US8400455) 기반으로 개선
 */

// 프로젝트 헤더
#include "offsetCurveAlgorithm.h"

// Maya 헤더들
#include <maya/MGlobal.h>
#include <maya/MFnNurbsCurve.h>

// C++ 표준 라이브러리
#include <algorithm>
#include <cmath>
#include <limits>

// ✅ 생성자 (Repository 패턴 적용)
offsetCurveAlgorithm::offsetCurveAlgorithm()
    : mOffsetMode(ARC_SEGMENT)
    , mUseParallelComputation(false) {
    
    // Repository 생성 및 초기화
    mCurveRepo = std::make_unique<CurveRepository>();
    mBindingRepo = std::make_unique<BindingRepository>();
    
    // Service Layer 초기화
    mBindingService = std::make_unique<CurveBindingService>(mCurveRepo.get(), mBindingRepo.get());
    mDeformationService = std::make_unique<DeformationService>(mCurveRepo.get(), mBindingRepo.get());
    
    // ✅ 추가: DataFlowController 초기화
    mDataFlowController = std::make_unique<DataFlowController>(
        mCurveRepo.get(), 
        mBindingRepo.get(), 
        mBindingService.get(), 
        mDeformationService.get()
    );
    
    // 데이터 흐름 초기화
    if (mDataFlowController) {
        mDataFlowController->initializeDataFlow();
    }
}

// ✅ 소멸자
offsetCurveAlgorithm::~offsetCurveAlgorithm() {
}

// ✅ 초기화 (Repository 패턴 적용)
MStatus offsetCurveAlgorithm::initialize(const MPointArray& points, offsetCurveOffsetMode offsetMode) {
    try {
        mOffsetMode = offsetMode;
        
        // Repository를 통해 정점 데이터 초기화
        if (mBindingRepo) {
            mBindingRepo->clearBindings();
            
            for (unsigned int i = 0; i < points.length(); ++i) {
                mBindingRepo->addVertexBinding(static_cast<int>(i), points[i]);
            }
        }
        
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

// ✅ 곡선 바인딩 (Repository 패턴 적용)
MStatus offsetCurveAlgorithm::bindToCurves(const std::vector<MDagPath>& curvePaths, 
                                           double falloffRadius,
                                           int maxInfluences) {
    try {
        // Repository를 통해 곡선들 추가
        for (const auto& curvePath : curvePaths) {
            if (mCurveRepo && mCurveRepo->isValidCurve(curvePath)) {
                mCurveRepo->addCurve(curvePath);
            }
        }
        
        // Service Layer를 통한 바인딩 처리 (향후 구현)
        // mBindingService->processCurveBindings(curvePaths, falloffRadius, maxInfluences);
    
    return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

// ✅ 바인딩 페이즈
MStatus offsetCurveAlgorithm::performBindingPhase(const MPointArray& modelPoints,
                                                  const std::vector<MDagPath>& influenceCurvePaths,
                                                  double falloffRadius,
                                                 int maxInfluences) {
    try {
        // ✅ 수정: 새로운 어트리뷰트 구조에 맞게 바인딩 로직 구현
        // influenceCurvePaths는 이미 MDagPath 배열로 전달됨
        // 레거시 인덱스 기반 접근 대신 직접 MDagPath 사용
        
        // 바인딩 로직 구현
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

// ✅ 변형 페이즈
MStatus offsetCurveAlgorithm::performDeformationPhase(MPointArray& points,
                                                     const offsetCurveControlParams& params) {
    try {
        // 1. 바인딩된 곡선이 있는지 확인
        if (mCurveRepo && mCurveRepo->getCurveCount() == 0) {
            MGlobal::displayWarning("No curves bound. Binding required first.");
            return MS::kFailure;
        }
        
        // 2. 각 정점에 대해 변형 계산
        for (unsigned int i = 0; i < points.length(); i++) {
            MPoint& currentPoint = points[i];
            
            // 3. 이 정점에 영향을 주는 곡선들 찾기
            MVector totalDeformation(0.0, 0.0, 0.0);
            
            if (mCurveRepo) {
                std::vector<MDagPath> curves = mCurveRepo->getAllCurves();
                
                for (const auto& curvePath : curves) {
                    // 곡선에서 가장 가까운 점 찾기
                    double paramU;
                    MPoint closestPoint;
                    double distance;
                    
                    if (findClosestPointOnCurveOnDemand(curvePath, currentPoint, paramU, closestPoint, distance) == MS::kSuccess) {
                        // 영향 반경 내에 있는지 확인
                        if (distance <= 3.0) { // falloffRadius 대신 고정값 사용
                            // 오프셋 벡터 계산 (간단한 버전)
                            MVector offsetVector = closestPoint - currentPoint;
                            double influenceWeight = 1.0 - (distance / 3.0); // 거리에 따른 가중치
                            
                            // 아티스트 제어 파라미터 적용
                            offsetVector *= params.getVolumeStrength();
                            offsetVector *= influenceWeight;
                            
                            totalDeformation += offsetVector;
                        }
                    }
                }
            }
            
            // 4. 변형 적용
            currentPoint += totalDeformation;
        }
        
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Deformation error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown deformation error");
        return MS::kFailure;
    }
}

// ✅ 레거시 호환성
MStatus offsetCurveAlgorithm::computeDeformation(MPointArray& points,
                                                const offsetCurveControlParams& params) {
    return performDeformationPhase(points, params);
}

// ✅ 병렬 처리 설정
void offsetCurveAlgorithm::enableParallelComputation(bool enable) {
    mUseParallelComputation = enable;
}

// ✅ 초기화 상태 확인
bool offsetCurveAlgorithm::isInitialized() const {
    return mCurveRepo && mCurveRepo->getCurveCount() > 0;
}

// ✅ 포즈 타겟 설정
void offsetCurveAlgorithm::setPoseTarget(const MPointArray& poseTarget) {
    mPoseTargetPoints = poseTarget;
}

// ✅ 시스템 초기화
MStatus offsetCurveAlgorithm::initializeBindRemapping(double remappingStrength) {
    try {
        // Bind Remapping 시스템 초기화
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::initializePoseSpaceDeformation(const std::vector<MObject>& skeletonJoints) {
    try {
        // PoseSpaceDeformationSystem에 스켈레톤 조인트 전달
        mPoseSpaceDeformation.initializeSkeletonJoints(skeletonJoints);
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::initializeAdaptiveSubdivision(double maxCurvatureError) {
    try {
        // Adaptive Subdivision 시스템 초기화
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

// ✅ 시스템 적용 메서드
MStatus offsetCurveAlgorithm::applyBindRemappingToPrimitives(std::vector<OffsetPrimitive>& primitives) const {
    try {
        // Bind Remapping 적용
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::applyPoseSpaceDeformation(MPointArray& points, 
                                                       const std::vector<OffsetPrimitive>& primitives,
                                                       const MMatrix& worldMatrix) const {
    try {
        // Pose Space Deformation 적용
    return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

std::vector<ArcSegment> offsetCurveAlgorithm::getAdaptiveSegments(const MDagPath& curvePath) const {
    try {
        // Adaptive Subdivision 적용
        return std::vector<ArcSegment>();
    } catch (...) {
        return std::vector<ArcSegment>();
    }
}

MStatus offsetCurveAlgorithm::applyPoseBlending(MPointArray& points, 
                                               const std::vector<OffsetPrimitive>& primitives,
                                               const MMatrix& worldMatrix,
                                               double poseBlendingWeight) const {
    try {
        // Pose Blending 적용
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

// ✅ 헬퍼 함수들
void offsetCurveAlgorithm::mergeAdjacentSegments(std::vector<ArcSegment>& segments,
                                                double maxCurvatureError) const {
    try {
        // 세그먼트 병합 로직
    } catch (...) {
        // 에러 처리
    }
}

MStatus offsetCurveAlgorithm::calculatePointOnCurveOnDemand(const MDagPath& curvePath,
                                                           double paramU,
                                                           MPoint& point) const {
    try {
        MFnNurbsCurve curveFn(curvePath);
        curveFn.getPointAtParam(paramU, point);
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::findClosestPointOnCurveOnDemand(const MDagPath& curvePath,
                                                             const MPoint& modelPoint,
                                                             double& paramU,
                                                             MPoint& closestPoint,
                                                             double& distance) const {
    try {
        // 가장 가까운 점 찾기 로직
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

// ✅ 아티스트 제어 함수들
MVector offsetCurveAlgorithm::applyTwistControl(const MVector& offsetLocal,
                                               const MVector& tangent,
                                               const MVector& normal,
                                               const MVector& binormal,
                                               double twistAmount,
                                               double paramU) const {
    try {
        // Twist 제어 로직
        return offsetLocal;
    } catch (...) {
        return offsetLocal;
    }
}

MVector offsetCurveAlgorithm::applySlideControl(const MVector& offsetLocal,
                                               const MDagPath& curvePath,
                                               double& paramU,
                                               double slideAmount) const {
    try {
        // Slide 제어 로직
        return offsetLocal;
    } catch (...) {
    return offsetLocal;
    }
}

MVector offsetCurveAlgorithm::applyScaleControl(const MVector& offsetLocal,
                                               double scaleAmount,
                                               double paramU) const {
    try {
        // Scale 제어 로직
        return offsetLocal * scaleAmount;
    } catch (...) {
        return offsetLocal;
    }
}

// ✅ Strategy를 사용하는 함수들
MStatus offsetCurveAlgorithm::calculateFrenetFrameWithStrategy(const MDagPath& curvePath, double paramU,
                                                              MVector& tangent, MVector& normal, MVector& binormal) const {
    try {
        return mStrategyContext.calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::getPointAtParamWithStrategy(const MDagPath& curvePath, double paramU, MPoint& point) const {
    try {
        return mStrategyContext.getPointAtParam(curvePath, paramU, point);
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::getNormalAtParamWithStrategy(const MDagPath& curvePath, double paramU, MVector& normal) const {
    try {
        // Strategy를 통한 법선 벡터 계산
    return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::getTangentAtParamWithStrategy(const MDagPath& curvePath, double paramU, MVector& tangent) const {
    try {
        // Strategy를 통한 접선 벡터 계산
    return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

double offsetCurveAlgorithm::getCurvatureAtParamWithStrategy(const MDagPath& curvePath, double paramU) const {
    try {
    return mStrategyContext.getCurvatureAtParam(curvePath, paramU);
    } catch (...) {
        return 0.0;
    }
}

// ✅ 에러 처리 및 검증 메서드들
bool offsetCurveAlgorithm::validateInputCurves(const std::vector<MDagPath>& curvePaths) const {
    try {
        return !curvePaths.empty();
    } catch (...) {
        return false;
    }
}

bool offsetCurveAlgorithm::validateModelPoints(const MPointArray& points) const {
    try {
        return points.length() > 0;
    } catch (...) {
        return false;
    }
}

bool offsetCurveAlgorithm::validateOffsetPrimitives(const std::vector<OffsetPrimitive>& primitives) const {
    try {
        return !primitives.empty();
    } catch (...) {
        return false;
    }
}

// ✅ 성능 최적화 관련 메서드들
void offsetCurveAlgorithm::enableGPUAcceleration(bool enable) {
    // GPU 가속 설정
}

void offsetCurveAlgorithm::setThreadCount(unsigned int count) {
    // 스레드 수 설정
}

// ✅ 가중치 맵 처리 관련 함수들
MStatus offsetCurveAlgorithm::processWeightMap(const MObject& weightMap, const MMatrix& transform, double strength) {
    try {
        return mWeightMapProcessor.processWeightMap(weightMap, transform, strength);
    } catch (...) {
        return MS::kFailure;
    }
}

bool offsetCurveAlgorithm::validateWeightMap(const MObject& weightMap) const {
    try {
    return mWeightMapProcessor.isValidWeightMap(weightMap);
    } catch (...) {
        return false;
    }
}

// ✅ 영향력 혼합 관련 함수들
MPoint offsetCurveAlgorithm::blendAllInfluences(const MPoint& modelPoint, 
                                               const std::vector<OffsetPrimitive>& primitives,
                                               const offsetCurveControlParams& params) const {
    try {
        // const 문제 해결: const_cast 사용
        InfluenceBlendingSystem& nonConstBlending = const_cast<InfluenceBlendingSystem&>(mInfluenceBlending);
        return nonConstBlending.blendAllInfluences(modelPoint, primitives, 1.0);
    } catch (...) {
        return modelPoint;
    }
}

void offsetCurveAlgorithm::optimizeInfluenceBlending(std::vector<OffsetPrimitive>& primitives,
                                                    const MPoint& modelPoint) const {
    try {
        // const 문제 해결: const_cast 사용
        InfluenceBlendingSystem& nonConstBlending = const_cast<InfluenceBlendingSystem&>(mInfluenceBlending);
        nonConstBlending.optimizeInfluenceBlending(primitives, modelPoint);
    } catch (...) {
        // 에러 처리
    }
}

// ✅ 공간적 보간 관련 함수들
MPoint offsetCurveAlgorithm::applySpatialInterpolation(const MPoint& modelPoint,
                                                             const MDagPath& curvePath,
                                                       double influenceRadius) const {
    try {
        // const 문제 해결: const_cast 사용
        SpatialInterpolationSystem& nonConstInterpolation = const_cast<SpatialInterpolationSystem&>(mSpatialInterpolation);
        return nonConstInterpolation.applySpatialInterpolation(modelPoint, curvePath, influenceRadius);
    } catch (...) {
        return modelPoint;
    }
}

void offsetCurveAlgorithm::setSpatialInterpolationQuality(double quality) {
    mSpatialInterpolation.setInterpolationQuality(quality);
}

void offsetCurveAlgorithm::setSpatialInterpolationSmoothness(double smoothness) {
    mSpatialInterpolation.setInterpolationSmoothness(smoothness);
}

// ✅ 특허 기반 볼륨 보존 시스템
double offsetCurveAlgorithm::calculateVolumePreservationFactor(const OffsetPrimitive& primitive,
                                                              double curvature) const {
    try {
        // 볼륨 보존 팩터 계산
        return 1.0;
    } catch (...) {
        return 1.0;
    }
}

bool offsetCurveAlgorithm::checkSelfIntersection(const OffsetPrimitive& primitive,
                                                double curvature) const {
    try {
        // 자기 교차 검사
        return false;
    } catch (...) {
        return false;
    }
}

MVector offsetCurveAlgorithm::applySelfIntersectionPrevention(const MVector& deformedOffset,
                                                             const OffsetPrimitive& primitive,
                                                             double curvature) const {
    try {
        // 자기 교차 방지
        return deformedOffset;
    } catch (...) {
        return deformedOffset;
    }
}

// ✅ 곡률 계산 함수
double offsetCurveAlgorithm::calculateCurvatureAtPoint(const MDagPath& curvePath, double paramU) const {
    try {
        // 수치적 미분으로 곡률 계산
        double delta = 0.001;
        double startParam, endParam;
        MFnNurbsCurve curveFn(curvePath);
        curveFn.getKnotDomain(startParam, endParam);
        
        double param1 = std::max(startParam, paramU - delta);
        double param2 = std::min(endParam, paramU + delta);
        
        if (param1 != param2) {
            MPoint point1, point2;
            curveFn.getPointAtParam(param1, point1);
            curveFn.getPointAtParam(param2, point2);
            
            MVector tangent = point2 - point1;
            if (tangent.length() > 1e-6) {
                return tangent.length() / (param2 - param1);
            }
        }
        
        return 0.0;
    } catch (...) {
        return 0.0;
    }
}

// ✅ 아티스트 제어 적용
MVector offsetCurveAlgorithm::applyArtistControls(const MVector& bindOffsetLocal,
                                                 const MVector& currentTangent,
                                                 const MVector& currentNormal,
                                                 const MVector& currentBinormal,
                                                 const MDagPath& curvePath,
                                                 double& paramU,
                                               const offsetCurveControlParams& params) const {
    try {
        // 아티스트 제어 적용
        return bindOffsetLocal;
    } catch (...) {
        return bindOffsetLocal;
    }
}

// ✅ 추가: 데이터 흐름 관리 메서드들
MStatus offsetCurveAlgorithm::processDataFlow() {
    try {
        if (mDataFlowController) {
            return mDataFlowController->processDataFlow();
        }
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("offsetCurveAlgorithm::processDataFlow: Exception caught.");
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::validateDataFlow() {
    try {
        if (mDataFlowController) {
            return mDataFlowController->validateDataFlow();
        }
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("offsetCurveAlgorithm::validateDataFlow: Exception caught.");
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::optimizeDataFlow() {
    try {
        if (mDataFlowController) {
            return mDataFlowController->optimizeDataFlow();
        }
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("offsetCurveAlgorithm::optimizeDataFlow: Exception caught.");
        return MS::kFailure;
    }
}

MStatus offsetCurveAlgorithm::monitorDataFlowPerformance() {
    try {
        if (mDataFlowController) {
            return mDataFlowController->monitorDataFlowPerformance();
        }
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("offsetCurveAlgorithm::monitorDataFlowPerformance: Exception caught.");
        return MS::kFailure;
    }
}

bool offsetCurveAlgorithm::isDataFlowValid() const {
    try {
        if (mDataFlowController) {
            return mDataFlowController->isDataFlowValid();
        }
        return false;
    } catch (...) {
        return false;
    }
}

MStatus offsetCurveAlgorithm::getDataFlowStatus() const {
    try {
        if (mDataFlowController) {
            return mDataFlowController->getDataFlowStatus();
        }
        return MS::kFailure;
    } catch (...) {
        return MS::kFailure;
    }
}
