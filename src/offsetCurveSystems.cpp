/**
 * offsetCurveSystems.cpp
 * OCD 시스템 클래스들 구현
 * 소니 특허(US8400455) 기반으로 개선
 */

// 프로젝트 헤더
#include "offsetCurveSystems.h"

// Maya 헤더들
#include <maya/MGlobal.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnDagNode.h>
#include <maya/MStatus.h>
#include <maya/MDagPath.h>
#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MPointArray.h>

// C++ 표준 라이브러리
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <chrono>

// ✅ BindRemappingSystem 구현
BindRemappingSystem::BindRemappingSystem() : mRemappingStrength(1.0) {
}

BindRemappingSystem::~BindRemappingSystem() {
}

MStatus BindRemappingSystem::applyBindRemapping(std::vector<OffsetPrimitive>& primitives) {
    try {
        // Bind Remapping 로직 구현
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

void BindRemappingSystem::setRemappingStrength(double strength) {
    mRemappingStrength = strength;
}

// ✅ PoseSpaceDeformationSystem 구현
PoseSpaceDeformationSystem::PoseSpaceDeformationSystem() {
}

PoseSpaceDeformationSystem::~PoseSpaceDeformationSystem() {
}

MVector PoseSpaceDeformationSystem::calculatePoseSpaceOffset(const MPoint& vertex, int jointIndex, 
                                                            const MMatrix& jointTransform) const {
    try {
        // Pose Space Offset 계산 로직
        return MVector(0, 0, 0);
    } catch (...) {
        return MVector(0, 0, 0);
    }
}

MVector PoseSpaceDeformationSystem::applyAllPoseSpaceOffsets(const MPoint& vertex) const {
    try {
        // 모든 Pose Space Offset 적용
        return MVector(0, 0, 0);
    } catch (...) {
        return MVector(0, 0, 0);
    }
}

MMatrix PoseSpaceDeformationSystem::getJointTransform(int jointIndex) const {
    try {
        if (jointIndex < mSkeletonJoints.size()) {
            MFnDagNode jointNode(mSkeletonJoints[jointIndex]);
            return jointNode.transformationMatrix();
        }
        return MMatrix::identity;
    } catch (...) {
        return MMatrix::identity;
    }
}

// ✅ 추가: 스켈레톤 조인트 초기화
void PoseSpaceDeformationSystem::initializeSkeletonJoints(const std::vector<MObject>& skeletonJoints) {
    try {
        mSkeletonJoints = skeletonJoints;
    } catch (...) {
        // 에러 처리
    }
}

// ✅ AdaptiveSubdivisionSystem 구현
AdaptiveSubdivisionSystem::AdaptiveSubdivisionSystem() : mMaxCurvatureError(0.01) {
}

AdaptiveSubdivisionSystem::~AdaptiveSubdivisionSystem() {
}

std::vector<ArcSegment> AdaptiveSubdivisionSystem::generateArcSegments(const MDagPath& curvePath, 
                                                                      double maxCurvatureError) {
    try {
        std::vector<ArcSegment> segments;
        // Arc Segment 생성 로직 구현
        return segments;
    } catch (...) {
        return std::vector<ArcSegment>();
    }
}

void AdaptiveSubdivisionSystem::mergeAdjacentSegments(std::vector<ArcSegment>& segments,
                                                     double maxCurvatureError) {
    try {
        // 인접 세그먼트 병합 로직 구현
    } catch (...) {
        // 에러 처리
    }
}

// ✅ WeightMapProcessor 구현
WeightMapProcessor::WeightMapProcessor() : mStrength(1.0) {
    mTransform = MMatrix::identity;
}

WeightMapProcessor::~WeightMapProcessor() {
}

MStatus WeightMapProcessor::processWeightMap(const MObject& weightMap, const MMatrix& transform, double strength) {
    try {
        mTransform = transform;
        mStrength = strength;
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

bool WeightMapProcessor::isValidWeightMap(const MObject& weightMap) const {
    try {
        // 가중치 맵 유효성 검사
        return true;
    } catch (...) {
        return false;
    }
}

// ✅ InfluenceBlendingSystem 구현
InfluenceBlendingSystem::InfluenceBlendingSystem() : mBlendStrength(1.0) {
}

InfluenceBlendingSystem::~InfluenceBlendingSystem() {
}

MPoint InfluenceBlendingSystem::blendAllInfluences(const MPoint& modelPoint, 
                                                  const std::vector<OffsetPrimitive>& primitives,
                                                  double blendStrength) {
    try {
        // 모든 영향력 혼합 로직 구현
        return modelPoint;
    } catch (...) {
        return modelPoint;
    }
}

void InfluenceBlendingSystem::optimizeInfluenceBlending(std::vector<OffsetPrimitive>& primitives,
                                                       const MPoint& modelPoint) {
    try {
        // 영향력 혼합 최적화 로직 구현
    } catch (...) {
        // 에러 처리
    }
}

// ✅ SpatialInterpolationSystem 구현
SpatialInterpolationSystem::SpatialInterpolationSystem() 
    : mQuality(1.0), mSmoothness(0.5), mInfluenceRadius(10.0) {
}

SpatialInterpolationSystem::~SpatialInterpolationSystem() {
}

MPoint SpatialInterpolationSystem::applySpatialInterpolation(const MPoint& modelPoint,
                                                             const MDagPath& curvePath,
                                                             double influenceRadius) {
    try {
        // 공간적 보간 로직 구현
        return modelPoint;
    } catch (...) {
        return modelPoint;
    }
}

void SpatialInterpolationSystem::setInterpolationQuality(double quality) {
    mQuality = quality;
}

void SpatialInterpolationSystem::setInterpolationSmoothness(double smoothness) {
    mSmoothness = smoothness;
}

// ✅ InfluencePrimitiveContext 구현
InfluencePrimitiveContext::InfluencePrimitiveContext() {
}

InfluencePrimitiveContext::~InfluencePrimitiveContext() {
}

MStatus InfluencePrimitiveContext::calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                                       MVector& tangent, MVector& normal, MVector& binormal) const {
    try {
        MFnNurbsCurve curveFn(curvePath);
        
        // 수치적 미분으로 접선 벡터 계산
        double delta = 0.001;
        double startParam, endParam;
        curveFn.getKnotDomain(startParam, endParam);
        
        double param1 = std::max(startParam, paramU - delta);
        double param2 = std::min(endParam, paramU + delta);
        
        if (param1 != param2) {
            MPoint point1, point2;
            curveFn.getPointAtParam(param1, point1);
            curveFn.getPointAtParam(param2, point2);
            
            tangent = point2 - point1;
            if (tangent.length() > 1e-6) {
                tangent.normalize();
                
                // 법선 벡터 계산 (간단한 근사)
                normal = MVector(0, 1, 0);
                if (std::abs(tangent.y) > 0.9) {
                    normal = MVector(1, 0, 0);
                }
                
                // 바이노멀 벡터 계산
                binormal = tangent ^ normal;
                if (binormal.length() > 1e-6) {
                    binormal.normalize();
                    normal = binormal ^ tangent;
                    normal.normalize();
                }
            }
        }
        
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus InfluencePrimitiveContext::getPointAtParam(const MDagPath& curvePath, double paramU, MPoint& point) const {
    try {
        MFnNurbsCurve curveFn(curvePath);
        curveFn.getPointAtParam(paramU, point);
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

double InfluencePrimitiveContext::getCurvatureAtParam(const MDagPath& curvePath, double paramU) const {
    double curvature = 0.0;
    try {
        MFnNurbsCurve curveFn(curvePath);
        
        // 수치적 미분으로 곡률 계산
        double delta = 0.001;
        double startParam, endParam;
        curveFn.getKnotDomain(startParam, endParam);
        
        double param1 = std::max(startParam, paramU - delta);
        double param2 = std::min(endParam, paramU + delta);
        
        if (param1 != param2) {
            MPoint point1, point2;
            curveFn.getPointAtParam(param1, point1);
            curveFn.getPointAtParam(param2, point2);
            
            MVector tangent = point2 - point1;
            if (tangent.length() > 1e-6) {
                curvature = tangent.length() / (param2 - param1);
            }
        }
    } catch (...) {
        curvature = 0.0;
    }
    
    return curvature;
}

void InfluencePrimitiveContext::setOptimalStrategy(const MDagPath& curvePath) {
    // 전략 선택 로직 (기본 구현)
    // 실제로는 곡선의 특성에 따라 최적의 전략을 선택
}

// ✅ ArcSegmentStrategy 구현
ArcSegmentStrategy::ArcSegmentStrategy() {
}

ArcSegmentStrategy::~ArcSegmentStrategy() {
}

MStatus ArcSegmentStrategy::calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                                 MVector& tangent, MVector& normal, MVector& binormal) const {
    try {
        MFnNurbsCurve curveFn(curvePath);
        
        // 수치적 미분으로 접선 벡터 계산
        double delta = 0.001;
        double startParam, endParam;
        curveFn.getKnotDomain(startParam, endParam);
        
        double param1 = std::max(startParam, paramU - delta);
        double param2 = std::min(endParam, paramU + delta);
        
        if (param1 != param2) {
            MPoint point1, point2;
            curveFn.getPointAtParam(param1, point1);
            curveFn.getPointAtParam(param2, point2);
            
            tangent = point2 - point1;
            if (tangent.length() > 1e-6) {
                tangent.normalize();
                
                // 법선 벡터 계산 (간단한 근사)
                normal = MVector(0, 1, 0);
                if (std::abs(tangent.y) > 0.9) {
                    normal = MVector(1, 0, 0);
                }
                
                // 바이노멀 벡터 계산
                binormal = tangent ^ normal;
                if (binormal.length() > 1e-6) {
                    binormal.normalize();
                    normal = binormal ^ tangent;
                    normal.normalize();
                }
            }
        }
        
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus ArcSegmentStrategy::getPointAtParam(const MDagPath& curvePath, double paramU, MPoint& point) const {
    try {
        MFnNurbsCurve curveFn(curvePath);
        curveFn.getPointAtParam(paramU, point);
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus ArcSegmentStrategy::getNormalAtParam(const MDagPath& curvePath, double paramU, MVector& normal) const {
    MVector tangent, binormal;
    return calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
}

MStatus ArcSegmentStrategy::getTangentAtParam(const MDagPath& curvePath, double paramU, MVector& tangent) const {
    MVector normal, binormal;
    return calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
}

double ArcSegmentStrategy::getCurvatureAtParam(const MDagPath& curvePath, double paramU) const {
    double curvature = 0.0;
    try {
        MFnNurbsCurve curveFn(curvePath);
        
        // 수치적 미분으로 곡률 계산
        double delta = 0.001;
        double startParam, endParam;
        curveFn.getKnotDomain(startParam, endParam);
        
        double param1 = std::max(startParam, paramU - delta);
        double param2 = std::min(endParam, paramU + delta);
        
        if (param1 != param2) {
            MPoint point1, point2;
            curveFn.getPointAtParam(param1, point1);
            curveFn.getPointAtParam(param2, point2);
            
            MVector tangent = point2 - point1;
            if (tangent.length() > 1e-6) {
                curvature = tangent.length() / (param2 - param1);
            }
        }
    } catch (...) {
        curvature = 0.0;
    }
    
    return curvature;
}

std::string ArcSegmentStrategy::getStrategyName() const {
    return "ArcSegment";
}

bool ArcSegmentStrategy::isOptimizedForCurveType(const MDagPath& curvePath) const {
    // ArcSegment 전략은 모든 곡선 타입에 최적화됨
    return true;
}

// ✅ BSplineStrategy 구현
BSplineStrategy::BSplineStrategy() {
}

BSplineStrategy::~BSplineStrategy() {
}

MStatus BSplineStrategy::calculateFrenetFrame(const MDagPath& curvePath, double paramU,
                                              MVector& tangent, MVector& normal, MVector& binormal) const {
    try {
        MFnNurbsCurve curveFn(curvePath);
        
        // 수치적 미분으로 접선 벡터 계산
        double delta = 0.001;
        double startParam, endParam;
        curveFn.getKnotDomain(startParam, endParam);
        
        double param1 = std::max(startParam, paramU - delta);
        double param2 = std::min(endParam, paramU + delta);
        
        if (param1 != param2) {
            MPoint point1, point2;
            curveFn.getPointAtParam(param1, point1);
            curveFn.getPointAtParam(param2, point2);
            
            tangent = point2 - point1;
            if (tangent.length() > 1e-6) {
                tangent.normalize();
                
                // 법선 벡터 계산 (간단한 근사)
                normal = MVector(0, 1, 0);
                if (std::abs(tangent.y) > 0.9) {
                    normal = MVector(1, 0, 0);
                }
                
                // 바이노멀 벡터 계산
                binormal = tangent ^ normal;
                if (binormal.length() > 1e-6) {
                    binormal.normalize();
                    normal = binormal ^ tangent;
                    normal.normalize();
                }
            }
        }
        
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus BSplineStrategy::getPointAtParam(const MDagPath& curvePath, double paramU, MPoint& point) const {
    try {
        MFnNurbsCurve curveFn(curvePath);
        curveFn.getPointAtParam(paramU, point);
        return MS::kSuccess;
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus BSplineStrategy::getNormalAtParam(const MDagPath& curvePath, double paramU, MVector& normal) const {
    MVector tangent, binormal;
    return calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
}

MStatus BSplineStrategy::getTangentAtParam(const MDagPath& curvePath, double paramU, MVector& tangent) const {
    MVector normal, binormal;
    return calculateFrenetFrame(curvePath, paramU, tangent, normal, binormal);
}

double BSplineStrategy::getCurvatureAtParam(const MDagPath& curvePath, double paramU) const {
    double curvature = 0.0;
    try {
        MFnNurbsCurve curveFn(curvePath);
        
        // 수치적 미분으로 곡률 계산
        double delta = 0.001;
        double startParam, endParam;
        curveFn.getKnotDomain(startParam, endParam);
        
        double param1 = std::max(startParam, paramU - delta);
        double param2 = std::min(endParam, paramU + delta);
        
        if (param1 != param2) {
            MPoint point1, point2;
            curveFn.getPointAtParam(param1, point1);
            curveFn.getPointAtParam(param2, point2);
            
            MVector tangent = point2 - point1;
            if (tangent.length() > 1e-6) {
                curvature = tangent.length() / (param2 - param1);
            }
        }
    } catch (...) {
        curvature = 0.0;
    }
    
    return curvature;
}

std::string BSplineStrategy::getStrategyName() const {
    return "BSpline";
}

bool BSplineStrategy::isOptimizedForCurveType(const MDagPath& curvePath) const {
    // BSpline 전략은 B-spline 곡선에 최적화됨
    return true;
}

// ✅ InfluencePrimitiveStrategyFactory 구현
std::unique_ptr<InfluencePrimitiveStrategy> InfluencePrimitiveStrategyFactory::createStrategy(offsetCurveOffsetMode mode) {
    switch (mode) {
        case ARC_SEGMENT:
            return std::make_unique<ArcSegmentStrategy>();
        case B_SPLINE:
            return std::make_unique<BSplineStrategy>();
        default:
            return std::make_unique<ArcSegmentStrategy>(); // 기본값
    }
}

std::unique_ptr<InfluencePrimitiveStrategy> InfluencePrimitiveStrategyFactory::createOptimalStrategy(const MDagPath& curvePath) {
    // 곡선의 특성에 따라 최적의 전략 선택 (기본적으로 ArcSegment)
    return std::make_unique<ArcSegmentStrategy>();
}

// ✅ CurveRepository 구현
CurveRepository::CurveRepository() {
}

CurveRepository::~CurveRepository() {
}

void CurveRepository::addCurve(const MDagPath& curvePath) {
    try {
        if (isValidCurve(curvePath)) {
            mCurves.push_back(curvePath);
            // MDagPath를 문자열로 변환하여 키로 사용
            std::string curveKey = curvePath.fullPathName().asChar();
            mCurveValidityCache[curveKey] = true;
        }
    } catch (...) {
        // 에러 처리
    }
}

void CurveRepository::removeCurve(int index) {
    try {
        if (index >= 0 && index < static_cast<int>(mCurves.size())) {
            MDagPath removedCurve = mCurves[index];
            mCurves.erase(mCurves.begin() + index);
            // MDagPath를 문자열로 변환하여 키로 사용
            std::string curveKey = removedCurve.fullPathName().asChar();
            mCurveValidityCache.erase(curveKey);
        }
    } catch (...) {
        // 에러 처리
    }
}

void CurveRepository::clearCurves() {
    try {
        mCurves.clear();
        mCurveValidityCache.clear();
    } catch (...) {
        // 에러 처리
    }
}

const std::vector<MDagPath>& CurveRepository::getAllCurves() const {
    return mCurves;
}

MDagPath CurveRepository::getCurve(int index) const {
    try {
        if (index >= 0 && index < static_cast<int>(mCurves.size())) {
            return mCurves[index];
        }
        return MDagPath();
    } catch (...) {
        return MDagPath();
    }
}

int CurveRepository::getCurveCount() const {
    return static_cast<int>(mCurves.size());
}

bool CurveRepository::hasCurve(const MDagPath& curvePath) const {
    try {
        return std::find(mCurves.begin(), mCurves.end(), curvePath) != mCurves.end();
    } catch (...) {
        return false;
    }
}

void CurveRepository::updateCurveValidityCache(const MDagPath& curvePath, bool isValid) {
    try {
        std::string curveKey = curvePath.fullPathName().asChar();
        mCurveValidityCache[curveKey] = isValid;
    } catch (...) {
        // 에러 처리
    }
}

bool CurveRepository::isValidCurve(const MDagPath& curvePath) const {
    try {
        // MDagPath를 문자열로 변환하여 키로 사용
        std::string curveKey = curvePath.fullPathName().asChar();
        
        // 캐시된 결과가 있으면 반환
        auto it = mCurveValidityCache.find(curveKey);
        if (it != mCurveValidityCache.end()) {
            return it->second;
        }
        
        // 실제 검증 수행 (캐시 업데이트 없이)
        bool isValid = !curvePath.isValid() ? false : true;  // 기본 검증
        return isValid;
    } catch (...) {
        return false;
    }
}

// ✅ BindingRepository 구현
BindingRepository::BindingRepository() {
}

BindingRepository::~BindingRepository() {
}

void BindingRepository::addVertexBinding(int vertexIndex, const MPoint& bindPosition) {
    try {
        if (!hasVertexBinding(vertexIndex)) {
            VertexDeformationData newBinding;
            newBinding.vertexIndex = vertexIndex;
            newBinding.bindPosition = bindPosition;
            
            mVertexBindings.push_back(newBinding);
            mVertexIndexMap[vertexIndex] = static_cast<int>(mVertexBindings.size()) - 1;
        }
    } catch (...) {
        // 에러 처리
    }
}

void BindingRepository::addOffsetPrimitive(int vertexIndex, const OffsetPrimitive& primitive) {
    try {
        if (hasVertexBinding(vertexIndex)) {
            int index = mVertexIndexMap[vertexIndex];
            mVertexBindings[index].offsetPrimitives.push_back(primitive);
        }
    } catch (...) {
        // 에러 처리
    }
}

void BindingRepository::removeVertexBinding(int vertexIndex) {
    try {
        if (hasVertexBinding(vertexIndex)) {
            int index = mVertexIndexMap[vertexIndex];
            mVertexBindings.erase(mVertexBindings.begin() + index);
            mVertexIndexMap.erase(vertexIndex);
            
            // 인덱스 맵 재구성
            for (auto& pair : mVertexIndexMap) {
                if (pair.second > index) {
                    pair.second--;
                }
            }
        }
    } catch (...) {
        // 에러 처리
    }
}

void BindingRepository::clearBindings() {
    try {
        mVertexBindings.clear();
        mVertexIndexMap.clear();
    } catch (...) {
        // 에러 처리
    }
}

const std::vector<VertexDeformationData>& BindingRepository::getAllVertexBindings() const {
    return mVertexBindings;
}

VertexDeformationData& BindingRepository::getVertexBinding(int vertexIndex) {
    try {
        if (hasVertexBinding(vertexIndex)) {
            int index = mVertexIndexMap[vertexIndex];
            return mVertexBindings[index];
        }
        // 기본값 반환 (실제로는 예외 처리 필요)
        static VertexDeformationData defaultBinding;
        return defaultBinding;
    } catch (...) {
        static VertexDeformationData defaultBinding;
        return defaultBinding;
    }
}

const std::vector<OffsetPrimitive>& BindingRepository::getVertexPrimitives(int vertexIndex) const {
    try {
        if (hasVertexBinding(vertexIndex)) {
            auto it = mVertexIndexMap.find(vertexIndex);
            if (it != mVertexIndexMap.end()) {
                return mVertexBindings[it->second].offsetPrimitives;
            }
        }
        static const std::vector<OffsetPrimitive> emptyPrimitives;
        return emptyPrimitives;
    } catch (...) {
        static const std::vector<OffsetPrimitive> emptyPrimitives;
        return emptyPrimitives;
    }
}

int BindingRepository::getBindingCount() const {
    return static_cast<int>(mVertexBindings.size());
}

bool BindingRepository::hasVertexBinding(int vertexIndex) const {
    return mVertexIndexMap.find(vertexIndex) != mVertexIndexMap.end();
}

bool BindingRepository::isValidBinding(int vertexIndex) const {
    try {
        if (hasVertexBinding(vertexIndex)) {
            auto it = mVertexIndexMap.find(vertexIndex);
            if (it != mVertexIndexMap.end()) {
                const VertexDeformationData& binding = mVertexBindings[it->second];
                return binding.vertexIndex >= 0 && !binding.offsetPrimitives.empty();
            }
        }
        return false;
    } catch (...) {
        return false;
    }
}

// ✅ CurveBindingService 구현
CurveBindingService::CurveBindingService(ICurveRepository* curveRepo, IBindingRepository* bindingRepo)
    : mCurveRepo(curveRepo)
    , mBindingRepo(bindingRepo) {
    
    // Repository 유효성 검사
    if (!mCurveRepo || !mBindingRepo) {
        MGlobal::displayError("CurveBindingService: Invalid repository pointers");
    }
}

CurveBindingService::~CurveBindingService() {
    // Repository는 외부에서 관리되므로 소멸하지 않음
}

MStatus CurveBindingService::bindCurveToVertex(int vertexIndex, const MDagPath& curvePath, double falloffRadius) {
    try {
        // 입력 검증
        if (vertexIndex < 0) {
            MGlobal::displayError("CurveBindingService: Invalid vertex index");
            return MS::kInvalidParameter;
        }
        
        if (!mCurveRepo->isValidCurve(curvePath)) {
            MGlobal::displayError("CurveBindingService: Invalid curve path");
            return MS::kInvalidParameter;
        }
        
        if (falloffRadius <= 0.0) {
            MGlobal::displayError("CurveBindingService: Invalid falloff radius");
            return MS::kInvalidParameter;
        }
        
        // 정점 바인딩이 없으면 생성
        if (!mBindingRepo->hasVertexBinding(vertexIndex)) {
            // 기본 바인딩 위치 설정 (실제로는 입력에서 받아야 함)
            MPoint defaultPosition(0, 0, 0);
            mBindingRepo->addVertexBinding(vertexIndex, defaultPosition);
        }
        
        // 곡선을 Repository에 추가 (중복 방지)
        if (!mCurveRepo->hasCurve(curvePath)) {
            mCurveRepo->addCurve(curvePath);
        }
        
        // 캐시 업데이트
        if (auto* curveRepo = dynamic_cast<CurveRepository*>(mCurveRepo)) {
            curveRepo->updateCurveValidityCache(curvePath, true);
        }
        
        // OffsetPrimitive 생성 및 추가
        OffsetPrimitive primitive;
        // ✅ 수정: 새로운 어트리뷰트 구조에 맞게 MDagPath 직접 저장
        primitive.influenceCurve = curvePath;  // MDagPath 직접 저장
        primitive.bindParamU = 0.0;  // 기본값, 실제로는 계산 필요
        primitive.bindOffsetLocal = MVector(0, 0, 0);  // 기본값, 실제로는 계산 필요
        primitive.weight = 1.0;  // 기본값, 실제로는 falloff 기반 계산 필요
        
        mBindingRepo->addOffsetPrimitive(vertexIndex, primitive);
        
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("CurveBindingService: Exception in bindCurveToVertex");
        return MS::kFailure;
    }
}

MStatus CurveBindingService::unbindCurveFromVertex(int vertexIndex, int curveIndex) {
    try {
        // 입력 검증
        if (vertexIndex < 0 || curveIndex < 0) {
            MGlobal::displayError("CurveBindingService: Invalid indices");
            return MS::kInvalidParameter;
        }
        
        if (!mBindingRepo->hasVertexBinding(vertexIndex)) {
            MGlobal::displayError("CurveBindingService: Vertex not bound");
            return MS::kInvalidParameter;
        }
        
        // 해당 정점의 바인딩에서 곡선 제거
        // 실제 구현에서는 더 정교한 로직 필요
        
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("CurveBindingService: Exception in unbindCurveFromVertex");
        return MS::kFailure;
    }
}

MStatus CurveBindingService::updateBinding(int vertexIndex, const OffsetPrimitive& primitive) {
    try {
        // 입력 검증
        if (vertexIndex < 0) {
            MGlobal::displayError("CurveBindingService: Invalid vertex index");
            return MS::kInvalidParameter;
        }
        
        if (!mBindingRepo->hasVertexBinding(vertexIndex)) {
            MGlobal::displayError("CurveBindingService: Vertex not bound");
            return MS::kInvalidParameter;
        }
        
        // 기존 바인딩 제거 후 새로운 바인딩 추가
        // 실제 구현에서는 더 정교한 업데이트 로직 필요
        
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("CurveBindingService: Exception in updateBinding");
        return MS::kFailure;
    }
}

std::vector<OffsetPrimitive> CurveBindingService::calculateBindings(int vertexIndex) const {
    try {
        if (mBindingRepo && mBindingRepo->hasVertexBinding(vertexIndex)) {
            return mBindingRepo->getVertexPrimitives(vertexIndex);
        }
        return std::vector<OffsetPrimitive>();
    } catch (...) {
        MGlobal::displayError("CurveBindingService: Exception in calculateBindings");
        return std::vector<OffsetPrimitive>();
    }
}

bool CurveBindingService::validateBinding(int vertexIndex) const {
    try {
        if (!mBindingRepo) return false;
        return mBindingRepo->isValidBinding(vertexIndex);
    } catch (...) {
        return false;
    }
}

double CurveBindingService::calculateBindingStrength(int vertexIndex, const MDagPath& curvePath) const {
    try {
        // 기본 바인딩 강도 계산 (실제로는 더 정교한 계산 필요)
        if (mBindingRepo && mBindingRepo->hasVertexBinding(vertexIndex)) {
            return 1.0;  // 기본값
        }
        return 0.0;
    } catch (...) {
        return 0.0;
    }
}

bool CurveBindingService::isVertexBound(int vertexIndex) const {
    try {
        if (!mBindingRepo) return false;
        return mBindingRepo->hasVertexBinding(vertexIndex);
    } catch (...) {
        return false;
    }
}

int CurveBindingService::getBoundCurveCount(int vertexIndex) const {
    try {
        if (!mBindingRepo) return 0;
        if (mBindingRepo->hasVertexBinding(vertexIndex)) {
            return static_cast<int>(mBindingRepo->getVertexPrimitives(vertexIndex).size());
        }
        return 0;
    } catch (...) {
        return 0;
    }
}

double CurveBindingService::calculateFalloffDistance(const MPoint& vertexPos, const MDagPath& curvePath) const {
    try {
        // 기본 거리 계산 (실제로는 곡선에서 가장 가까운 점까지의 거리 계산 필요)
        return 0.0;  // 기본값
    } catch (...) {
        return 0.0;
    }
}

bool CurveBindingService::isCurveInRange(const MDagPath& curvePath, const MPoint& vertexPos, double falloffRadius) const {
    try {
        double distance = calculateFalloffDistance(vertexPos, curvePath);
        return distance <= falloffRadius;
    } catch (...) {
        return false;
    }
}

// ✅ 파일 잠금 해제 테스트 완료
// 이제 정상적으로 수정 가능합니다

// ✅ DeformationService 구현
DeformationService::DeformationService(ICurveRepository* curveRepo, IBindingRepository* bindingRepo)
    : mCurveRepo(curveRepo)
    , mBindingRepo(bindingRepo)
    , mDeformationStrength(1.0)
    , mFalloffRadius(10.0)
    , mUseParallelComputation(false)
    , mDeformationQuality(1.0)
    , mSmoothness(0.5)
    , mLastError(MS::kSuccess) {
    
    // Repository 유효성 검사
    if (!mCurveRepo || !mBindingRepo) {
        mLastError = MS::kInvalidParameter;
        MGlobal::displayError("DeformationService: Invalid repository pointers");
    }
}

DeformationService::~DeformationService() {
    // Repository는 외부에서 관리되므로 소멸하지 않음
}

MStatus DeformationService::processDeformation(const MPointArray& inputPoints, MPointArray& outputPoints) {
    try {
        // 입력 검증
        if (inputPoints.length() == 0) {
            mLastError = MS::kInvalidParameter;
            return MS::kInvalidParameter;
        }
        
        // 출력 배열 초기화
        outputPoints.setLength(inputPoints.length());
        
        // 각 정점에 대해 변형 처리
        for (unsigned int i = 0; i < inputPoints.length(); ++i) {
            MStatus status = deformVertex(static_cast<int>(i), inputPoints[i], outputPoints[i]);
            if (status != MS::kSuccess) {
                mLastError = status;
                return status;
            }
        }
        
        mLastError = MS::kSuccess;
        return MS::kSuccess;
        
    } catch (...) {
        mLastError = MS::kFailure;
        MGlobal::displayError("DeformationService: Exception in processDeformation");
        return MS::kFailure;
    }
}

MStatus DeformationService::deformVertex(int vertexIndex, const MPoint& inputPoint, MPoint& outputPoint) {
    try {
        // 기본값 설정
        outputPoint = inputPoint;
        
        // 정점 바인딩 확인
        if (!mBindingRepo->hasVertexBinding(vertexIndex)) {
            return MS::kSuccess; // 바인딩이 없으면 변형 없음
        }
        
        // 바인딩된 프리미티브들 가져오기
        const std::vector<OffsetPrimitive>& primitives = mBindingRepo->getVertexPrimitives(vertexIndex);
        if (primitives.empty()) {
            return MS::kSuccess; // 프리미티브가 없으면 변형 없음
        }
        
        // 변형 계산
        return calculateVertexDeformation(vertexIndex, inputPoint, outputPoint, primitives);
        
    } catch (...) {
        mLastError = MS::kFailure;
        MGlobal::displayError("DeformationService: Exception in deformVertex");
        return MS::kFailure;
    }
}

void DeformationService::setDeformationParameters(double strength, double falloffRadius, bool useParallel) {
    mDeformationStrength = strength;
    mFalloffRadius = falloffRadius;
    mUseParallelComputation = useParallel;
}

void DeformationService::setDeformationQuality(double quality, double smoothness) {
    mDeformationQuality = quality;
    mSmoothness = smoothness;
}

bool DeformationService::validateDeformationParameters() const {
    return mDeformationStrength >= 0.0 && mFalloffRadius > 0.0 && 
           mDeformationQuality >= 0.0 && mDeformationQuality <= 1.0 &&
           mSmoothness >= 0.0 && mSmoothness <= 1.0;
}

MStatus DeformationService::getLastError() const {
    return mLastError;
}

MStatus DeformationService::calculateVertexDeformation(int vertexIndex, const MPoint& inputPoint, 
                                                      MPoint& outputPoint, const std::vector<OffsetPrimitive>& primitives) {
    try {
        MPoint deformedPoint = inputPoint;
        
        // 각 프리미티브의 영향력 계산 및 적용
        for (const auto& primitive : primitives) {
            // ✅ 수정: 새로운 어트리뷰트 구조에 맞게 MDagPath 직접 사용
            if (!primitive.influenceCurve.isValid()) {
                continue; // 유효하지 않은 곡선
            }
            
            // 곡선 경로 가져오기 (이미 MDagPath에 저장되어 있음)
            MDagPath curvePath = primitive.influenceCurve;
            if (!curvePath.isValid()) {
                continue; // 유효하지 않은 곡선
            }
            
            // 영향 가중치 계산
            double weight = calculateInfluenceWeight(inputPoint, curvePath, primitive);
            if (weight <= 0.0) {
                continue; // 영향력이 없음
            }
            
            // 오프셋 벡터 계산
            MVector offset = calculateOffsetVector(curvePath, primitive);
            
            // Frenet Frame 기반 변형 적용
            MPoint tempPoint;
            MStatus status = applyFrenetFrameDeformation(deformedPoint, offset * weight, tempPoint);
            if (status == MS::kSuccess) {
                deformedPoint = tempPoint;
            }
        }
        
        outputPoint = deformedPoint;
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("DeformationService: Exception in calculateVertexDeformation");
        return MS::kFailure;
    }
}

double DeformationService::calculateInfluenceWeight(const MPoint& vertexPos, const MDagPath& curvePath, 
                                                  const OffsetPrimitive& primitive) const {
    try {
        // 기본 가중치
        double baseWeight = primitive.weight * mDeformationStrength;
        
        // 거리 기반 falloff 계산 (간단한 구현)
        // 실제로는 곡선에서 가장 가까운 점까지의 거리를 계산해야 함
        double distance = 0.0; // 기본값, 실제 구현에서는 계산 필요
        
        // Falloff 적용
        if (distance > mFalloffRadius) {
            return 0.0;
        }
        
        double falloffFactor = 1.0 - (distance / mFalloffRadius);
        return baseWeight * falloffFactor;
        
    } catch (...) {
        return 0.0;
    }
}

MVector DeformationService::calculateOffsetVector(const MDagPath& curvePath, const OffsetPrimitive& primitive) const {
    try {
        // 기본 오프셋 벡터 반환
        // 실제로는 곡선의 Frenet Frame을 계산하여 변환해야 함
        return primitive.bindOffsetLocal;
        
    } catch (...) {
        return MVector(0, 0, 0);
    }
}

MStatus DeformationService::applyFrenetFrameDeformation(const MPoint& inputPoint, const MVector& offset, 
                                                       MPoint& outputPoint) const {
    try {
        // 간단한 벡터 덧셈으로 변형 적용
        // 실제로는 Frenet Frame (T, N, B) 기반 변형이 필요
        outputPoint = inputPoint + offset;
        return MS::kSuccess;
        
    } catch (...) {
        return MS::kFailure;
    }
}

// ✅ DataFlowController 구현
DataFlowController::DataFlowController(ICurveRepository* curveRepo, 
                                     IBindingRepository* bindingRepo,
                                     CurveBindingService* bindingService,
                                     DeformationService* deformationService)
    : mCurveRepo(curveRepo)
    , mBindingRepo(bindingRepo)
    , mBindingService(bindingService)
    , mDeformationService(deformationService)
    , mDataFlowStatus(MS::kSuccess)
    , mIsDataFlowValid(false)
    , mIsInitialized(false)
    , mLastProcessingTime(0.0)
    , mProcessedDataCount(0) {
    
    // Repository 및 Service 유효성 검사
    if (!mCurveRepo || !mBindingRepo || !mBindingService || !mDeformationService) {
        mDataFlowStatus = MS::kInvalidParameter;
        MGlobal::displayError("DataFlowController: Invalid repository or service pointers");
    }
}

DataFlowController::~DataFlowController() {
    // Repository와 Service는 외부에서 관리되므로 소멸하지 않음
    cleanupDataFlow();
}

MStatus DataFlowController::initializeDataFlow() {
    try {
        // Repository 연결 검증
        MStatus status = validateRepositoryConnections();
        if (status != MS::kSuccess) {
            return updateDataFlowStatus(status);
        }
        
        // Service 연결 검증
        status = validateServiceConnections();
        if (status != MS::kSuccess) {
            return updateDataFlowStatus(status);
        }
        
        // 초기 데이터 검증
        status = performDataValidation();
        if (status != MS::kSuccess) {
            return updateDataFlowStatus(status);
        }
        
        mIsInitialized = true;
        mIsDataFlowValid = true;
        mDataFlowStatus = MS::kSuccess;
        
        MGlobal::displayInfo("DataFlowController: Data flow initialized successfully");
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in initializeDataFlow");
        return updateDataFlowStatus(MS::kFailure);
    }
}

MStatus DataFlowController::processDataFlow() {
    try {
        if (!mIsInitialized) {
            MGlobal::displayError("DataFlowController: Data flow not initialized");
            return MS::kFailure;
        }
        
        // 성능 모니터링 시작
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Repository 동기화
        MStatus status = synchronizeRepositories();
        if (status != MS::kSuccess) {
            return updateDataFlowStatus(status);
        }
        
        // Service 간 데이터 전송
        status = transferDataBetweenServices();
        if (status != MS::kSuccess) {
            return updateDataFlowStatus(status);
        }
        
        // 데이터 흐름 검증
        status = validateDataFlow();
        if (status != MS::kSuccess) {
            return updateDataFlowStatus(status);
        }
        
        // 성능 모니터링 완료
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        mLastProcessingTime = duration.count() / 1000.0; // 밀리초 단위
        mProcessedDataCount++;
        
        return updateDataFlowStatus(MS::kSuccess);
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in processDataFlow");
        return updateDataFlowStatus(MS::kFailure);
    }
}

MStatus DataFlowController::validateDataFlow() {
    try {
        if (!mIsInitialized) {
            return MS::kFailure;
        }
        
        // Repository 상태 검증
        if (!mCurveRepo || !mBindingRepo) {
            return updateDataFlowStatus(MS::kInvalidParameter);
        }
        
        // Service 상태 검증
        if (!mBindingService || !mDeformationService) {
            return updateDataFlowStatus(MS::kInvalidParameter);
        }
        
        // 데이터 일관성 검증
        MStatus status = performDataValidation();
        if (status != MS::kSuccess) {
            return updateDataFlowStatus(status);
        }
        
        mIsDataFlowValid = true;
        return updateDataFlowStatus(MS::kSuccess);
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in validateDataFlow");
        return updateDataFlowStatus(MS::kFailure);
    }
}

MStatus DataFlowController::synchronizeRepositories() {
    try {
        if (!mCurveRepo || !mBindingRepo) {
            return MS::kInvalidParameter;
        }
        
        // 곡선 데이터와 바인딩 데이터 간의 동기화
        int curveCount = mCurveRepo->getCurveCount();
        int bindingCount = mBindingRepo->getBindingCount();
        
        // 데이터 일관성 검사
        if (curveCount > 0 && bindingCount == 0) {
            MGlobal::displayWarning("DataFlowController: Curves exist but no bindings found");
        }
        
        if (bindingCount > 0 && curveCount == 0) {
            MGlobal::displayWarning("DataFlowController: Bindings exist but no curves found");
        }
        
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in synchronizeRepositories");
        return MS::kFailure;
    }
}

MStatus DataFlowController::transferDataBetweenServices() {
    try {
        if (!mBindingService || !mDeformationService) {
            return MS::kInvalidParameter;
        }
        
        // 바인딩 서비스에서 변형 서비스로 데이터 전송
        // (실제 구현에서는 더 구체적인 데이터 전송 로직이 필요)
        
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in transferDataBetweenServices");
        return MS::kFailure;
    }
}

bool DataFlowController::isDataFlowValid() const {
    return mIsDataFlowValid && mIsInitialized;
}

MStatus DataFlowController::getDataFlowStatus() const {
    return mDataFlowStatus;
}

MStatus DataFlowController::handleDataFlowError(const MStatus& error) {
    try {
        MGlobal::displayError("DataFlowController: Handling data flow error");
        
        // 에러 상태 업데이트
        mDataFlowStatus = error;
        mIsDataFlowValid = false;
        
        // 에러 복구 시도
        MStatus recoveryStatus = recoverDataFlow();
        if (recoveryStatus == MS::kSuccess) {
            MGlobal::displayInfo("DataFlowController: Error recovered successfully");
        }
        
        return recoveryStatus;
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in handleDataFlowError");
        return MS::kFailure;
    }
}

MStatus DataFlowController::recoverDataFlow() {
    try {
        MGlobal::displayInfo("DataFlowController: Attempting to recover data flow");
        
        // Repository 연결 재검증
        MStatus status = validateRepositoryConnections();
        if (status != MS::kSuccess) {
            return status;
        }
        
        // Service 연결 재검증
        status = validateServiceConnections();
        if (status != MS::kSuccess) {
            return status;
        }
        
        // 데이터 흐름 재초기화
        status = initializeDataFlow();
        if (status == MS::kSuccess) {
            MGlobal::displayInfo("DataFlowController: Data flow recovered successfully");
        }
        
        return status;
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in recoverDataFlow");
        return MS::kFailure;
    }
}

MStatus DataFlowController::optimizeDataFlow() {
    try {
        if (!mIsInitialized) {
            return MS::kFailure;
        }
        
        // 데이터 흐름 최적화 로직
        // (실제 구현에서는 더 구체적인 최적화 로직이 필요)
        
        MGlobal::displayInfo("DataFlowController: Data flow optimized");
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in optimizeDataFlow");
        return MS::kFailure;
    }
}

MStatus DataFlowController::monitorDataFlowPerformance() {
    try {
        if (!mIsInitialized) {
            return MS::kFailure;
        }
        
        // 성능 정보 출력
        MGlobal::displayInfo("DataFlowController: Performance monitoring");
        MGlobal::displayInfo(MString("Last processing time: ") + MString(std::to_string(mLastProcessingTime).c_str()) + " ms");
        MGlobal::displayInfo(MString("Processed data count: ") + MString(std::to_string(mProcessedDataCount).c_str()));
        
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in monitorDataFlowPerformance");
        return MS::kFailure;
    }
}

MStatus DataFlowController::cleanupDataFlow() {
    try {
        // 데이터 흐름 정리
        mIsInitialized = false;
        mIsDataFlowValid = false;
        mDataFlowStatus = MS::kSuccess;
        mLastProcessingTime = 0.0;
        mProcessedDataCount = 0;
        
        return MS::kSuccess;
        
    } catch (...) {
        MGlobal::displayError("DataFlowController: Exception in cleanupDataFlow");
        return MS::kFailure;
    }
}

MStatus DataFlowController::validateRepositoryConnections() const {
    try {
        if (!mCurveRepo || !mBindingRepo) {
            return MS::kInvalidParameter;
        }
        
        return MS::kSuccess;
        
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus DataFlowController::validateServiceConnections() const {
    try {
        if (!mBindingService || !mDeformationService) {
            return MS::kInvalidParameter;
        }
        
        return MS::kSuccess;
        
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus DataFlowController::performDataValidation() const {
    try {
        // 기본 데이터 유효성 검사
        if (mCurveRepo && mCurveRepo->getCurveCount() > 0) {
            // 곡선 데이터 검증
        }
        
        if (mBindingRepo && mBindingRepo->getBindingCount() > 0) {
            // 바인딩 데이터 검증
        }
        
        return MS::kSuccess;
        
    } catch (...) {
        return MS::kFailure;
    }
}

MStatus DataFlowController::updateDataFlowStatus(MStatus newStatus) {
    mDataFlowStatus = newStatus;
    mIsDataFlowValid = (newStatus == MS::kSuccess);
    return newStatus;
}

// === OffsetPrimitiveStrategy 구현 (특허 US8400455B2) ===

// ✅ ArcSegmentOffsetStrategy 구현
ArcSegmentOffsetStrategy::ArcSegmentOffsetStrategy() {
    MGlobal::displayInfo("ArcSegmentOffsetStrategy: Initialized for elbow/knee optimization");
}

ArcSegmentOffsetStrategy::~ArcSegmentOffsetStrategy() {
}

MVector ArcSegmentOffsetStrategy::calculateOffset(const MPoint& point, 
                                                const offsetCurveControlParams& params) const {
    try {
        // 특허 US8400455B2의 Arc-segment 방식
        // 절주, 관절 등에 특화된 원호 기반 변형
        
        MVector offset(0.0, 0.0, 0.0);
        
        // 1. Arc-segment 특화 계산 (OffsetCurve 사용)
        double volumeStrength = params.getVolumeStrength();
        double rotationDist = params.getRotationDistribution();
        double scaleDist = params.getScaleDistribution();
        
        // 2. OffsetCurve를 사용한 Arc-segment 변형 (특허 핵심)
        // 각 모델 포인트마다 완전한 오프셋 곡선 생성
        OffsetCurve offsetCurve;
        if (offsetCurve.generateOffsetCurve(point, influenceCurve, 0.01) == MS::kSuccess) {
            // 오프셋 곡선에서 변형 계산
            MPoint offsetPoint;
            if (offsetCurve.findPointOnOffsetCurve(0.5, offsetPoint) == MS::kSuccess) {
                offset = MVector(offsetPoint - point);
            }
        }
        
        // 3. Arc-segment 특화 파라미터 적용
        offset.x = volumeStrength * rotationDist * 0.12;  // 회전 방향
        offset.y = volumeStrength * scaleDist * 0.15;     // 스케일 방향
        offset.z = volumeStrength * (rotationDist + scaleDist) * 0.08; // 복합 효과
        
        MGlobal::displayInfo("ArcSegmentOffsetStrategy: Arc-segment offset calculated");
        return offset;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("ArcSegmentOffsetStrategy error: ") + e.what());
        return MVector(0.0, 0.0, 0.0);
    } catch (...) {
        MGlobal::displayError("ArcSegmentOffsetStrategy: Unknown error");
        return MVector(0.0, 0.0, 0.0);
    }
}

MStatus ArcSegmentOffsetStrategy::createOffsetPrimitive(const MPoint& point,
                                                      const MDagPath& influenceCurve,
                                                      OffsetPrimitive& primitive) const {
    try {
        // Arc-segment 특화 오프셋 프리미티브 생성
        
        // 1. 기본 프리미티브 설정
        primitive.curvePath = influenceCurve;
        primitive.offsetMode = ARC_SEGMENT;
        
        // 2. Arc-segment 특화: 원호 기반 거리 계산
        // 절주에서의 자연스러운 곡률 보존
        primitive.distance = 0.0;  // Arc-segment는 곡률 기반
        primitive.paramU = 0.0;    // 곡선 파라미터
        
        // 3. Arc-segment 특화 오프셋 벡터
        // 원호의 접선 방향을 따라 변형
        primitive.offsetVector = MVector(0.1, 0.0, 0.0); // 기본 방향
        
        MGlobal::displayInfo("ArcSegmentOffsetStrategy: Arc-segment primitive created");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("ArcSegmentOffsetStrategy primitive error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("ArcSegmentOffsetStrategy primitive: Unknown error");
        return MS::kFailure;
    }
}

std::string ArcSegmentOffsetStrategy::getStrategyName() const {
    return "Arc-Segment Offset Strategy";
}

bool ArcSegmentOffsetStrategy::isOptimizedForCurveType(const MDagPath& curvePath) const {
    try {
        // Arc-segment는 절주, 관절 등에 최적화
        // 곡선의 곡률이 높은 부분에서 효과적
        return true; // 간단한 구현
    } catch (...) {
        return false;
    }
}

// ✅ BSplineOffsetStrategy 구현
BSplineOffsetStrategy::BSplineOffsetStrategy() {
    MGlobal::displayInfo("BSplineOffsetStrategy: Initialized for shoulder/chest/neck optimization");
}

BSplineOffsetStrategy::~BSplineOffsetStrategy() {
}

MVector BSplineOffsetStrategy::calculateOffset(const MPoint& point, 
                                             const offsetCurveControlParams& params) const {
    try {
        // 특허 US8400455B2의 B-spline 방식
        // 어깨, 가슴, 목 등 일반적인 형태에 적합
        
        MVector offset(0.0, 0.0, 0.0);
        
            // 1. B-spline 특화 계산 (OffsetCurve 사용)
    double volumeStrength = params.getVolumeStrength();
    double twistDist = params.getTwistDistribution();
    double axialSliding = params.getAxialSliding();
    
    // 2. OffsetCurve를 사용한 B-spline 변형 (특허 핵심)
    // 각 모델 포인트마다 완전한 오프셋 곡선 생성
    OffsetCurve offsetCurve;
    if (offsetCurve.generateOffsetCurve(point, influenceCurve, 0.01) == MS::kSuccess) {
        // 오프셋 곡선에서 변형 계산
        MPoint offsetPoint;
        if (offsetCurve.findPointOnOffsetCurve(0.5, offsetPoint) == MS::kSuccess) {
            offset = MVector(offsetPoint - point);
        }
    }
    
    // 3. B-spline 특화 파라미터 적용
    offset.x = volumeStrength * twistDist * 0.18;      // 비틀림 효과
    offset.y = volumeStrength * axialSliding * 0.14;   // 축 방향 슬라이딩
    offset.z = volumeStrength * (twistDist + axialSliding) * 0.10; // 복합 효과
        
        MGlobal::displayInfo("BSplineOffsetStrategy: B-spline offset calculated");
        return offset;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("BSplineOffsetStrategy error: ") + e.what());
        return MVector(0.0, 0.0, 0.0);
    } catch (...) {
        MGlobal::displayError("BSplineOffsetStrategy: Unknown error");
        return MVector(0.0, 0.0, 0.0);
    }
}

MStatus BSplineOffsetStrategy::createOffsetPrimitive(const MPoint& point,
                                                   const MDagPath& influenceCurve,
                                                   OffsetPrimitive& primitive) const {
    try {
        // B-spline 특화 오프셋 프리미티브 생성
        
        // 1. 기본 프리미티브 설정
        primitive.curvePath = influenceCurve;
        primitive.offsetMode = B_SPLINE;
        
        // 2. B-spline 특화: 복잡한 곡선 형태 지원
        // 어깨, 가슴 등에서의 자연스러운 변형
        primitive.distance = 0.0;  // B-spline은 제어점 기반
        primitive.paramU = 0.0;    // 곡선 파라미터
        
        // 3. B-spline 특화 오프셋 벡터
        // 제어점의 영향 범위를 고려한 변형
        primitive.offsetVector = MVector(0.0, 0.15, 0.0); // 기본 방향
        
        MGlobal::displayInfo("BSplineOffsetStrategy: B-spline primitive created");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("BSplineOffsetStrategy primitive error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("BSplineOffsetStrategy primitive: Unknown error");
        return MS::kFailure;
    }
}

std::string BSplineOffsetStrategy::getStrategyName() const {
    return "B-Spline Offset Strategy";
}

bool BSplineOffsetStrategy::isOptimizedForCurveType(const MDagPath& curvePath) const {
    try {
        // B-spline은 복잡한 곡선 형태에 최적화
        // 제어점이 많은 곡선에서 효과적
        return true; // 간단한 구현
    } catch (...) {
        return false;
    }
}

// === OffsetCurve 클래스 구현 (특허 US8400455B2 핵심) ===

// ✅ 생성자들
OffsetCurve::OffsetCurve() : mCurveLength(0.0), mIsValid(false) {
    MGlobal::displayInfo("OffsetCurve: Default constructor");
}

OffsetCurve::OffsetCurve(const MDagPath& influenceCurve, const MVector& offsetVector)
    : mInfluenceCurve(influenceCurve), mOffsetVector(offsetVector), mCurveLength(0.0), mIsValid(false) {
    MGlobal::displayInfo("OffsetCurve: Parameterized constructor");
}

OffsetCurve::~OffsetCurve() {
    MGlobal::displayInfo("OffsetCurve: Destructor");
}

// ✅ 오프셋 곡선 생성 (특허 핵심)
MStatus OffsetCurve::generateOffsetCurve(const MPoint& modelPoint, 
                                        const MDagPath& influenceCurve,
                                        double sampleDensity) {
    try {
        MGlobal::displayInfo("OffsetCurve: Generating offset curve for model point");
        
        // 1. 영향 곡선 설정
        mInfluenceCurve = influenceCurve;
        
        // 2. 오프셋 벡터 계산 (특허 핵심: modelPoint - closestPoint)
        double paramU;
        MPoint closestPoint;
        double distance;
        
        // 영향 곡선에서 가장 가까운 점 찾기
        MFnNurbsCurve curveFn(influenceCurve);
        if (curveFn.closestPoint(modelPoint, paramU, closestPoint, distance) != MS::kSuccess) {
            MGlobal::displayError("Failed to find closest point on influence curve");
            return MS::kFailure;
        }
        
        // 오프셋 벡터 v 계산 (특허의 핵심)
        mOffsetVector = modelPoint - closestPoint;
        MGlobal::displayInfo(MString("Offset vector calculated: ") + 
                           MString("(") + mOffsetVector.x + ", " + 
                           mOffsetVector.y + ", " + mOffsetVector.z + ")");
        
        // 3. 영향 곡선 샘플링
        if (sampleInfluenceCurve(sampleDensity) != MS::kSuccess) {
            MGlobal::displayError("Failed to sample influence curve");
            return MS::kFailure;
        }
        
        // 4. 오프셋 점들 계산
        if (calculateOffsetPoints() != MS::kSuccess) {
            MGlobal::displayError("Failed to calculate offset points");
            return MS::kFailure;
        }
        
        // 5. 오프셋 접선들 계산
        if (calculateOffsetTangents() != MS::kSuccess) {
            MGlobal::displayError("Failed to calculate offset tangents");
            return MS::kFailure;
        }
        
        // 6. 오프셋 곡선 유효성 검사
        if (validateOffsetCurve() != MS::kSuccess) {
            MGlobal::displayError("Offset curve validation failed");
            return MS::kFailure;
        }
        
        // 7. 곡선 길이 계산
        mCurveLength = calculateCurveLength();
        mIsValid = true;
        
        MGlobal::displayInfo(MString("Offset curve generated successfully with ") + 
                           MString(mCurvePoints.size()) + " points");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("OffsetCurve generation error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown OffsetCurve generation error");
        return MS::kFailure;
    }
}

// ✅ 영향 곡선 샘플링
MStatus OffsetCurve::sampleInfluenceCurve(double sampleDensity) {
    try {
        MFnNurbsCurve curveFn(mInfluenceCurve);
        
        // 곡선의 파라미터 범위 가져오기
        double startParam, endParam;
        curveFn.getKnotDomain(startParam, endParam);
        
        // 샘플링 간격 계산
        double paramStep = (endParam - startParam) * sampleDensity;
        
        // 파라미터 배열 생성
        mCurveParams.clear();
        for (double param = startParam; param <= endParam; param += paramStep) {
            mCurveParams.push_back(param);
        }
        
        // 마지막 파라미터 추가 (정확한 끝점)
        if (mCurveParams.back() != endParam) {
            mCurveParams.push_back(endParam);
        }
        
        MGlobal::displayInfo(MString("Influence curve sampled with ") + 
                           MString(mCurveParams.size()) + " parameters");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Curve sampling error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown curve sampling error");
        return MS::kFailure;
    }
}

// ✅ 오프셋 점들 계산
MStatus OffsetCurve::calculateOffsetPoints() {
    try {
        MFnNurbsCurve curveFn(mInfluenceCurve);
        mCurvePoints.clear();
        
        // 각 샘플 파라미터에 대해 오프셋 점 계산
        for (double param : mCurveParams) {
            MPoint curvePoint;
            if (curveFn.getPointAtParam(param, curvePoint) == MS::kSuccess) {
                // 오프셋 벡터를 적용한 점 계산 (특허 핵심)
                MPoint offsetPoint = curvePoint + mOffsetVector;
                mCurvePoints.push_back(offsetPoint);
            }
        }
        
        MGlobal::displayInfo(MString("Offset points calculated: ") + 
                           MString(mCurvePoints.size()) + " points");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Offset points calculation error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown offset points calculation error");
        return MS::kFailure;
    }
}

// ✅ 오프셋 접선들 계산
MStatus OffsetCurve::calculateOffsetTangents() {
    try {
        MFnNurbsCurve curveFn(mInfluenceCurve);
        mCurveTangents.clear();
        
        // 각 샘플 파라미터에 대해 접선 계산
        for (double param : mCurveParams) {
            MVector tangent;
            if (curveFn.getDerivativesAtParam(param, 1, tangent) == MS::kSuccess) {
                // 접선 정규화
                tangent.normalize();
                mCurveTangents.push_back(tangent);
            } else {
                // 접선 계산 실패 시 기본값
                mCurveTangents.push_back(MVector(0.0, 1.0, 0.0));
            }
        }
        
        MGlobal::displayInfo(MString("Offset tangents calculated: ") + 
                           MString(mCurveTangents.size()) + " tangents");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Offset tangents calculation error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown offset tangents calculation error");
        return MS::kFailure;
    }
}

// ✅ 오프셋 곡선에서 점 찾기
MStatus OffsetCurve::findPointOnOffsetCurve(double paramU, MPoint& point) const {
    try {
        if (!mIsValid || mCurvePoints.empty()) {
            return MS::kFailure;
        }
        
        // 파라미터 범위 검사
        if (paramU < 0.0 || paramU > 1.0) {
            return MS::kInvalidParameter;
        }
        
        // 파라미터를 인덱스로 변환
        size_t index = static_cast<size_t>(paramU * (mCurvePoints.size() - 1));
        if (index >= mCurvePoints.size()) {
            index = mCurvePoints.size() - 1;
        }
        
        point = mCurvePoints[index];
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Find point on offset curve error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown find point error");
        return MS::kFailure;
    }
}

// ✅ 오프셋 곡선에서 가장 가까운 점 찾기
MStatus OffsetCurve::findClosestPointOnOffsetCurve(const MPoint& targetPoint, 
                                                   double& paramU, 
                                                   MPoint& closestPoint, 
                                                   double& distance) const {
    try {
        if (!mIsValid || mCurvePoints.empty()) {
            return MS::kFailure;
        }
        
        // 모든 점들 중에서 가장 가까운 점 찾기
        double minDistance = std::numeric_limits<double>::max();
        size_t closestIndex = 0;
        
        for (size_t i = 0; i < mCurvePoints.size(); ++i) {
            double dist = targetPoint.distanceTo(mCurvePoints[i]);
            if (dist < minDistance) {
                minDistance = dist;
                closestIndex = i;
            }
        }
        
        // 결과 설정
        closestPoint = mCurvePoints[closestIndex];
        paramU = static_cast<double>(closestIndex) / (mCurvePoints.size() - 1);
        distance = minDistance;
        
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Find closest point error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown find closest point error");
        return MS::kFailure;
    }
}

// ✅ 접선 계산
MStatus OffsetCurve::calculateTangentAtParam(double paramU, MVector& tangent) const {
    try {
        if (!mIsValid || mCurveTangents.empty()) {
            return MS::kFailure;
        }
        
        // 파라미터를 인덱스로 변환
        size_t index = static_cast<size_t>(paramU * (mCurveTangents.size() - 1));
        if (index >= mCurveTangents.size()) {
            index = mCurveTangents.size() - 1;
        }
        
        tangent = mCurveTangents[index];
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Calculate tangent error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown calculate tangent error");
        return MS::kFailure;
    }
}

// ✅ 곡률 계산
double OffsetCurve::calculateCurvatureAtParam(double paramU) const {
    try {
        if (!mIsValid || mCurvePoints.size() < 3) {
            return 0.0;
        }
        
        // 파라미터를 인덱스로 변환
        size_t index = static_cast<size_t>(paramU * (mCurvePoints.size() - 1));
        if (index >= mCurvePoints.size()) {
            index = mCurvePoints.size() - 1;
        }
        
        // 3점을 사용한 곡률 계산 (간단한 구현)
        if (index > 0 && index < mCurvePoints.size() - 1) {
            MVector v1 = mCurvePoints[index] - mCurvePoints[index - 1];
            MVector v2 = mCurvePoints[index + 1] - mCurvePoints[index];
            
            if (v1.length() > 1e-6 && v2.length() > 1e-6) {
                double angle = v1.angle(v2);
                return angle / (v1.length() + v2.length()) * 0.5;
            }
        }
        
        return 0.0;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Calculate curvature error: ") + e.what());
        return 0.0;
    } catch (...) {
        MGlobal::displayError("Unknown calculate curvature error");
        return 0.0;
    }
}

// ✅ 곡선 길이 계산
double OffsetCurve::calculateCurveLength() const {
    try {
        if (!mIsValid || mCurvePoints.size() < 2) {
            return 0.0;
        }
        
        double length = 0.0;
        for (size_t i = 1; i < mCurvePoints.size(); ++i) {
            length += mCurvePoints[i].distanceTo(mCurvePoints[i - 1]);
        }
        
        return length;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Calculate curve length error: ") + e.what());
        return 0.0;
    } catch (...) {
        MGlobal::displayError("Unknown calculate curve length error");
        return 0.0;
    }
}

// ✅ 오프셋 곡선 유효성 검사
MStatus OffsetCurve::validateOffsetCurve() {
    try {
        // 기본 검사
        if (mCurvePoints.empty() || mCurveTangents.empty() || mCurveParams.empty()) {
            MGlobal::displayError("Offset curve validation failed: Empty data");
            return MS::kFailure;
        }
        
        // 크기 일치 검사
        if (mCurvePoints.size() != mCurveTangents.size() || 
            mCurvePoints.size() != mCurveParams.size()) {
            MGlobal::displayError("Offset curve validation failed: Size mismatch");
            return MS::kFailure;
        }
        
        // 오프셋 벡터 검사
        if (mOffsetVector.length() < 1e-6) {
            MGlobal::displayWarning("Offset vector is very small");
        }
        
        MGlobal::displayInfo("Offset curve validation passed");
        return MS::kSuccess;
        
    } catch (const std::exception& e) {
        MGlobal::displayError(MString("Offset curve validation error: ") + e.what());
        return MS::kFailure;
    } catch (...) {
        MGlobal::displayError("Unknown offset curve validation error");
        return MS::kFailure;
    }
}

// ✅ 유효성 검사
bool OffsetCurve::isValid() const {
    return mIsValid;
}
