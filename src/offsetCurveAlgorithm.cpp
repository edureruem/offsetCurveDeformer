/**
 * offsetCurveAlgorithm.cpp
 * OCD 핵심 알고리즘 구현
 */

#include "offsetCurveAlgorithm.h"
#include <maya/MGlobal.h>
#include <algorithm>
#include <cmath>
#include <limits>

// offsetCurveInfluence 구현
offsetCurveInfluence::offsetCurveInfluence()
    : mCurveIndex(-1),
      mParamU(0.0),
      mWeight(0.0),
      mBindLocalPoint(0.0, 0.0, 0.0),
      mBindMatrix(),
      mTangent(1.0, 0.0, 0.0),
      mNormal(0.0, 1.0, 0.0),
      mBinormal(0.0, 0.0, 1.0),
      mCurvature(0.0),
      mSegmentLength(0.0),
      mSegmentIndex(0),
      mIsJunction(false),
      mJunctionRadius(0.0)
{
}

offsetCurveInfluence::~offsetCurveInfluence()
{
}

void offsetCurveInfluence::setCurveIndex(int index) { mCurveIndex = index; }
void offsetCurveInfluence::setParamU(double param) { mParamU = param; }
void offsetCurveInfluence::setWeight(double weight) { mWeight = weight; }
void offsetCurveInfluence::setBindLocalPoint(const MPoint& point) { mBindLocalPoint = point; }
void offsetCurveInfluence::setBindMatrix(const MMatrix& matrix) { mBindMatrix = matrix; }
void offsetCurveInfluence::setTangent(const MVector& tangent) { mTangent = tangent; }
void offsetCurveInfluence::setNormal(const MVector& normal) { mNormal = normal; }
void offsetCurveInfluence::setBinormal(const MVector& binormal) { mBinormal = binormal; }
void offsetCurveInfluence::setCurvature(double curvature) { mCurvature = curvature; }
void offsetCurveInfluence::setSegmentIndex(int index) { mSegmentIndex = index; }
void offsetCurveInfluence::setIsJunction(bool isJunction) { mIsJunction = isJunction; }
void offsetCurveInfluence::setJunctionRadius(double radius) { mJunctionRadius = radius; }
void offsetCurveInfluence::setSegmentLength(double length) { mSegmentLength = length; }

int offsetCurveInfluence::getCurveIndex() const { return mCurveIndex; }
double offsetCurveInfluence::getParamU() const { return mParamU; }
double offsetCurveInfluence::getWeight() const { return mWeight; }
MPoint offsetCurveInfluence::getBindLocalPoint() const { return mBindLocalPoint; }
const MMatrix& offsetCurveInfluence::getBindMatrix() const { return mBindMatrix; }
const MVector& offsetCurveInfluence::getTangent() const { return mTangent; }
const MVector& offsetCurveInfluence::getNormal() const { return mNormal; }
const MVector& offsetCurveInfluence::getBinormal() const { return mBinormal; }
double offsetCurveInfluence::getCurvature() const { return mCurvature; }
int offsetCurveInfluence::getSegmentIndex() const { return mSegmentIndex; }
bool offsetCurveInfluence::isJunction() const { return mIsJunction; }
double offsetCurveInfluence::getJunctionRadius() const { return mJunctionRadius; }
double offsetCurveInfluence::getSegmentLength() const { return mSegmentLength; }

// offsetCurveAlgorithm 구현
offsetCurveAlgorithm::offsetCurveAlgorithm()
    : mOffsetMode(ARC_SEGMENT), mUseParallelComputation(false), mStrategy(nullptr)
{
}

offsetCurveAlgorithm::~offsetCurveAlgorithm()
{
    // mStrategy는 unique_ptr이므로 자동으로 해제됨
}

// 알고리즘 초기화
MStatus offsetCurveAlgorithm::initialize(const MPointArray& points, offsetCurveOffsetMode offsetMode)
{
    mOffsetMode = offsetMode;
    mOriginalPoints = points;
    mCurveDataList.clear();
    mVertexDataMap.clear();
    
    // 선택된 모드에 따라 전략 객체 생성
    mStrategy.reset(OffsetCurveStrategyFactory::createStrategy(offsetMode));
    
    return MS::kSuccess;
}

// 영향 곡선에 바인딩
MStatus offsetCurveAlgorithm::bindToCurves(const std::vector<MDagPath>& curvePaths, 
                                 double falloffRadius,
                                 int maxInfluences)
{
    MStatus status;
    
    // 곡선 데이터 초기화
    mCurveDataList.clear();
    for (const MDagPath& curvePath : curvePaths) {
        offsetCurveData curveData;
        curveData.initialize(curvePath);
        curveData.cacheBindPoseData();
        curveData.computeOrientations();
        mCurveDataList.push_back(curveData);
    }
    
    // 각 정점의 영향 계산
    for (unsigned int i = 0; i < mOriginalPoints.length(); i++) {
        offsetCurveVertexData vertexData;
        vertexData.vertexIndex = i;
        vertexData.originalPosition = mOriginalPoints[i];
        
        // 영향 맵 계산
        computeInfluenceWeights(i, mOriginalPoints[i], mCurveDataList, falloffRadius, maxInfluences);
    }
    
    return MS::kSuccess;
}

// 변형 계산 (아티스트 제어 파라미터 추가)
MStatus offsetCurveAlgorithm::computeDeformation(MPointArray& points,
                                      const offsetCurveControlParams& params)
{
    if (mVertexDataMap.empty() || mCurveDataList.empty()) {
        return MS::kFailure;
    }
    
    // 전략 객체 확인
    if (!mStrategy) {
        mStrategy.reset(OffsetCurveStrategyFactory::createStrategy(mOffsetMode));
    }
    
    // 병렬 처리 사용 여부에 따라 다르게 처리
    if (mUseParallelComputation) {
        // 병렬 처리 설정
        const unsigned int numThreads = std::min(8u, MThreadPool::getNumThreads());
        offsetCurveTaskData* taskData = new offsetCurveTaskData[numThreads];
        
        // 작업 분할
        unsigned int pointsPerTask = points.length() / numThreads;
        unsigned int remainingPoints = points.length() % numThreads;
        
        unsigned int startIdx = 0;
        MThreadPool::init();
        
        // 각 스레드에 작업 할당
        for (unsigned int i = 0; i < numThreads; ++i) {
            taskData[i].startIdx = startIdx;
            taskData[i].endIdx = startIdx + pointsPerTask + (i < remainingPoints ? 1 : 0);
            taskData[i].points = &points;
            taskData[i].curveData = &mCurveDataList;
            taskData[i].vertexData = &mVertexDataMap;
            taskData[i].params = params;
            taskData[i].offsetMode = mOffsetMode;
            taskData[i].strategy = mStrategy.get();
            
            // 작업 추가
            MThreadPool::createTask(parallelDeformationTask, (void*)&taskData[i], NULL);
            
            startIdx = taskData[i].endIdx;
        }
        
        // 작업 실행 및 완료 대기
        MThreadPool::executeAndJoin();
        
        delete[] taskData;
    }
    else {
        // 단일 스레드 처리
        // 각 정점에 대한 변형 계산
        for (auto& vertexPair : mVertexDataMap) {
            offsetCurveVertexData& vertexData = vertexPair.second;
            unsigned int vertexIdx = vertexData.vertexIndex;
            
            if (vertexIdx >= points.length()) {
                continue;
            }
            
            MPoint& currentPoint = points[vertexIdx];
            const MPoint& originalPoint = vertexData.originalPosition;
            std::vector<offsetCurveInfluence>& influences = vertexData.influences;
            
            if (influences.empty()) {
                continue;
            }
            
            // 변형된 위치 계산
            MPoint deformedPoint(0.0, 0.0, 0.0, 1.0);
            double totalWeight = 0.0;
            
            // 각 영향 곡선 기여도 계산
            for (const offsetCurveInfluence& influence : influences) {
                if (influence.getCurveIndex() < 0 || 
                    influence.getCurveIndex() >= static_cast<int>(mCurveDataList.size())) {
                    continue;
                }
                
                offsetCurveData& curveData = mCurveDataList[influence.getCurveIndex()];
                
                // 현재 곡선 상태 업데이트
                curveData.updateCurveData();
                
                // 바인드 포즈 매트릭스
                MMatrix bindMatrix = influence.getBindMatrix();
                MMatrix bindMatrixInverse = bindMatrix.inverse();
                
                // 영향 곡선 위의 해당 지점 위치
                MPoint curvePoint;
                curveData.getPoint(influence.getParamU(), curvePoint);
                
                // 현재 탄젠트, 노멀, 바이노멀 계산
                MVector currentTangent, currentNormal, currentBinormal;
                mStrategy->computeFrenetFrame(curveData, influence.getParamU(), 
                                           currentTangent, currentNormal, currentBinormal);
                
                // 꼬임 분포 적용
                if (params.getTwistDistribution() != 1.0) {
                    applyTwistDistribution(currentNormal, currentBinormal, influence, params.getTwistDistribution());
                }
                
                // 현재 로컬 프레임 매트릭스
                double currentMatrixArray[4][4] = {
                    {currentTangent.x, currentNormal.x, currentBinormal.x, curvePoint.x},
                    {currentTangent.y, currentNormal.y, currentBinormal.y, curvePoint.y},
                    {currentTangent.z, currentNormal.z, currentBinormal.z, curvePoint.z},
                    {0.0, 0.0, 0.0, 1.0}
                };
                MMatrix currentMatrix(currentMatrixArray);
                
                // 바인드 포즈에서 현재 포즈까지의 변환 매트릭스
                MMatrix transformMatrix = currentMatrix * bindMatrixInverse;
                
                // 회전 분포 적용
                if (params.getRotationDistribution() != 1.0) {
                    applyRotationDistribution(transformMatrix, influence, params.getRotationDistribution());
                }
                
                // 스케일 분포 적용
                if (params.getScaleDistribution() != 1.0) {
                    applyScaleDistribution(transformMatrix, influence, params.getScaleDistribution());
                }
                
                // 오프셋 위치 계산
                MPoint localBindPoint = influence.getBindLocalPoint();
                MPoint transformedPoint = localBindPoint * transformMatrix;
                
                // 축 방향 슬라이딩 적용
                if (params.getAxialSliding() != 0.0) {
                    transformedPoint = applyAxialSliding(transformedPoint, influence, params.getAxialSliding());
                }
                
                // 가중치 적용
                deformedPoint.x += transformedPoint.x * influence.getWeight();
                deformedPoint.y += transformedPoint.y * influence.getWeight();
                deformedPoint.z += transformedPoint.z * influence.getWeight();
                totalWeight += influence.getWeight();
            }
            
            // 가중치 정규화
            if (totalWeight > 0.0) {
                deformedPoint = deformedPoint / totalWeight;
            } else {
                deformedPoint = originalPoint;
            }
            
            // 볼륨 보존 계산 및 적용
            if (params.getVolumeStrength() > 0.0) {
                MVector volumeOffset = computeVolumePreservation(originalPoint, 
                                                             deformedPoint, 
                                                             influences, 
                                                             params.getVolumeStrength());
                deformedPoint += volumeOffset;
            }
            
            // 슬라이딩 효과 적용
            if (params.getSlideEffect() != 0.0) {
                deformedPoint = computeSlideEffect(originalPoint, 
                                                deformedPoint, 
                                                influences, 
                                                params.getSlideEffect());
            }
            
            // 포즈 블렌딩 적용
            if (params.isPoseBlendingEnabled() && params.getPoseWeight() > 0.0) {
                deformedPoint = applyPoseBlending(deformedPoint, vertexIdx, params.getPoseWeight());
            }
            
            // 최종 위치 설정
            currentPoint = deformedPoint;
        }
    }
    
    return MS::kSuccess;
}

// 오프셋 곡선 계산 - 전략 패턴 활용
MStatus offsetCurveAlgorithm::computeOffsetCurves(const MPointArray& points,
                                       const std::vector<MDagPath>& curvePaths)
{
    MStatus status;
    
    // 전략 객체 확인
    if (!mStrategy) {
        mStrategy.reset(OffsetCurveStrategyFactory::createStrategy(mOffsetMode));
    }
    
    // 정점 영향 맵 비우기
    mVertexDataMap.clear();
    
    // 각 정점에 대해 오프셋 곡선 계산
    for (unsigned int i = 0; i < points.length(); i++) {
        offsetCurveVertexData vertexData;
        vertexData.vertexIndex = i;
        vertexData.originalPosition = points[i];
        
        // 각 곡선에 대한 오프셋 계산
        for (size_t curveIdx = 0; curveIdx < mCurveDataList.size(); curveIdx++) {
            offsetCurveData& curveData = mCurveDataList[curveIdx];
            
            // 전략 객체에 계산 위임
            status = mStrategy->computeOffsets(i, points[i], curveData, vertexData.influences);
            
            if (status != MS::kSuccess) {
                continue;
            }
            
            // 곡선 인덱스 설정 (전략에서는 알 수 없음)
            for (offsetCurveInfluence& influence : vertexData.influences) {
                if (influence.getCurveIndex() == -1) {
                    influence.setCurveIndex(curveIdx);
                }
            }
        }
        
        // 정점 데이터 저장
        mVertexDataMap[i] = vertexData;
    }
    
    return MS::kSuccess;
}

// 병렬 처리 활성화/비활성화
void offsetCurveAlgorithm::enableParallelComputation(bool enable)
{
    mUseParallelComputation = enable;
}

// 포즈 타겟 설정
void offsetCurveAlgorithm::setPoseTarget(const MPointArray& poseTarget)
{
    mPoseTargetPoints = poseTarget;
}

// 각 정점의 영향 맵 계산
void offsetCurveAlgorithm::computeInfluenceWeights(unsigned int vertexIndex,
                                        const MPoint& point,
                                        const std::vector<offsetCurveData>& curves,
                                        double falloffRadius,
                                        int maxInfluences)
{
    // 전략 객체 확인
    if (!mStrategy) {
        mStrategy.reset(OffsetCurveStrategyFactory::createStrategy(mOffsetMode));
    }
    
    struct InfluenceDistance {
        int curveIndex;
        double paramU;
        double distance;
        MPoint closestPoint;
        
        bool operator<(const InfluenceDistance& other) const {
            return distance < other.distance;
        }
    };
    
    std::vector<InfluenceDistance> allInfluences;
    
    // 모든 곡선에서 가장 가까운 점 찾기
    for (size_t i = 0; i < curves.size(); i++) {
        const offsetCurveData& curveData = curves[i];
        
        InfluenceDistance influence;
        influence.curveIndex = static_cast<int>(i);
        
        MPoint closestPoint;
        influence.distance = mStrategy->findClosestPointOnCurve(point, curveData, 
                                                            influence.paramU, closestPoint);
        influence.closestPoint = closestPoint;
        
        if (influence.distance <= falloffRadius) {
            allInfluences.push_back(influence);
        }
    }
    
    // 거리에 따라 정렬
    std::sort(allInfluences.begin(), allInfluences.end());
    
    // 최대 영향 수 제한
    if (static_cast<int>(allInfluences.size()) > maxInfluences) {
        allInfluences.resize(maxInfluences);
    }
    
    // 영향 가중치 계산
    double totalWeight = 0.0;
    offsetCurveVertexData& vertexData = mVertexDataMap[vertexIndex];
    vertexData.influences.clear();
    
    for (const InfluenceDistance& influence : allInfluences) {
        // 거리 기반 가중치 계산
        double weight = 1.0 - (influence.distance / falloffRadius);
        weight = std::max(0.0, std::min(1.0, weight));
        weight = weight * weight;  // 제곱하여 더 부드러운 폴오프
        
        // 곡선 데이터 가져오기
        const offsetCurveData& curveData = curves[influence.curveIndex];
        
        // 프레넷 프레임 계산
        MVector tangent, normal, binormal;
        mStrategy->computeFrenetFrame(curveData, influence.paramU, tangent, normal, binormal);
        
        // 곡률 계산
        double curvature;
        curveData.getCurvature(influence.paramU, curvature);
        
        // 영향 객체 생성
        offsetCurveInfluence curveInfluence;
        curveInfluence.setCurveIndex(influence.curveIndex);
        
        // 로컬 좌표 계산
        MVector toPoint = point - influence.closestPoint;
        double localX = toPoint * normal;
        double localY = toPoint * binormal;
        double localZ = toPoint * tangent;
        
        // 나머지 속성 설정
        curveInfluence.setParamU(influence.paramU);
        curveInfluence.setWeight(weight);
        curveInfluence.setBindLocalPoint(MPoint(localX, localY, localZ));
        curveInfluence.setTangent(tangent);
        curveInfluence.setNormal(normal);
        curveInfluence.setBinormal(binormal);
        curveInfluence.setCurvature(curvature);
        
        // 바인드 행렬 계산 및 설정
        MMatrix matrix = createFrenetFrame(tangent, normal, binormal, influence.closestPoint);
        curveInfluence.setBindMatrix(matrix);
        
        // 세그먼트 정보 설정
        if (mOffsetMode == ARC_SEGMENT) {
            int segmentIndex = curveData.getSegmentIndex(influence.paramU);
            curveInfluence.setSegmentIndex(segmentIndex);
            
            // 세그먼트 길이 계산
            MPoint segStart, segEnd;
            if (curveData.getSegmentPoints(segmentIndex, segStart, segEnd)) {
                curveInfluence.setSegmentLength(segStart.distanceTo(segEnd));
            }
        }
        
        // 영향 추가
        vertexData.influences.push_back(curveInfluence);
        totalWeight += weight;
    }
    
    // 가중치 정규화
    if (totalWeight > 0.0) {
        for (offsetCurveInfluence& influence : vertexData.influences) {
            influence.setWeight(influence.getWeight() / totalWeight);
        }
    }
}

// 볼륨 보존 계산
MVector offsetCurveAlgorithm::computeVolumePreservation(const MPoint& originalPoint,
                                             const MPoint& deformedPoint,
                                             const std::vector<offsetCurveInfluence>& influences,
                                             double volumeStrength)
{
    MVector volumeOffset(0.0, 0.0, 0.0);
    
    // 영향이 없으면 오프셋 없음
    if (influences.empty()) {
        return volumeOffset;
    }
    
    // 평균 곡률과 가중치 계산
    double weightedCurvature = 0.0;
    double totalWeight = 0.0;
    
    for (const offsetCurveInfluence& influence : influences) {
        weightedCurvature += influence.getCurvature() * influence.getWeight();
        totalWeight += influence.getWeight();
    }
    
    if (totalWeight > 0.0) {
        weightedCurvature /= totalWeight;
    }
    
    // 볼륨 인자 계산 (0.0 ~ 1.0)
    double volumeFactor = std::min(weightedCurvature * volumeStrength * 0.5, 1.0);
    
    // 메인 영향 (가장 높은 가중치)
    const offsetCurveInfluence* mainInfluence = &influences[0];
    for (const offsetCurveInfluence& influence : influences) {
        if (influence.getWeight() > mainInfluence->getWeight()) {
            mainInfluence = &influence;
        }
    }
    
    // 볼륨 방향 벡터 계산
    MVector normal = mainInfluence->getNormal();
    
    // 오프셋 방향 및 크기 계산
    MVector moveDirection = originalPoint - deformedPoint;
    double moveDistance = moveDirection.length();
    
    if (moveDistance > 0.0) {
        // 법선 방향으로 볼륨 보존 오프셋 적용
        volumeOffset = normal * (moveDistance * volumeFactor);
    }
    
    return volumeOffset;
}

// 슬라이딩 효과 계산
MPoint offsetCurveAlgorithm::computeSlideEffect(const MPoint& originalPoint,
                                     const MPoint& deformedPoint,
                                     const std::vector<offsetCurveInfluence>& influences,
                                     double slideEffect)
{
    // 영향이 없으면 변형 없음
    if (influences.empty()) {
        return deformedPoint;
    }
    
    // 가장 높은 가중치의 영향 찾기
    const offsetCurveInfluence* mainInfluence = &influences[0];
    for (const offsetCurveInfluence& influence : influences) {
        if (influence.getWeight() > mainInfluence->getWeight()) {
            mainInfluence = &influence;
        }
    }
    
    // 슬라이딩 방향 계산
    MVector tangent = mainInfluence->getTangent();
    
    // 원래 위치에서 변형 위치까지의 벡터
    MVector moveVec = deformedPoint - originalPoint;
    
    // 탄젠트 방향으로의 투영 계산
    double tangentProj = moveVec * tangent;
    
    // 슬라이딩 효과 적용
    MPoint slidPoint = deformedPoint + tangent * (tangentProj * slideEffect);
    
    return slidPoint;
}

// 회전 분포 적용
void offsetCurveAlgorithm::applyRotationDistribution(MMatrix& transformMatrix,
                                          const offsetCurveInfluence& influence,
                                          double rotationFactor)
{
    // 곡률에 기반한 회전 분포 적용
    double curvature = influence.getCurvature();
    
    // 회전 인자 계산 (곡률 기반)
    double rotationScale = 1.0 + (curvature * (rotationFactor - 1.0));
    rotationScale = std::max(0.1, rotationScale);
    
    // 회전 부분만 스케일링
    for (unsigned int i = 0; i < 3; i++) {
        for (unsigned int j = 0; j < 3; j++) {
            transformMatrix(i, j) *= rotationScale;
        }
    }
}

// 스케일 분포 적용
void offsetCurveAlgorithm::applyScaleDistribution(MMatrix& transformMatrix,
                                       const offsetCurveInfluence& influence,
                                       double scaleFactor)
{
    // 곡률에 기반한 스케일 분포 적용
    double curvature = influence.getCurvature();
    
    // 스케일 인자 계산 (곡률 기반)
    double scaleValue = 1.0 + (curvature * (scaleFactor - 1.0));
    scaleValue = std::max(0.1, scaleValue);
    
    // 변환 행렬에 균일 스케일 적용
    transformMatrix(0, 0) *= scaleValue;
    transformMatrix(1, 1) *= scaleValue;
    transformMatrix(2, 2) *= scaleValue;
}

// 꼬임 분포 적용
void offsetCurveAlgorithm::applyTwistDistribution(MVector& normal,
                                       MVector& binormal,
                                       const offsetCurveInfluence& influence,
                                       double twistFactor)
{
    // 곡률에 기반한 꼬임 분포 적용
    double curvature = influence.getCurvature();
    
    // 꼬임 각도 계산 (곡률 기반)
    double twistAngle = curvature * (twistFactor - 1.0) * M_PI * 0.5;
    
    // 회전 행렬 계산
    double cosAngle = cos(twistAngle);
    double sinAngle = sin(twistAngle);
    
    // 노멀과 바이노멀 회전
    MVector newNormal = normal * cosAngle + binormal * sinAngle;
    MVector newBinormal = binormal * cosAngle - normal * sinAngle;
    
    // 정규화
    newNormal.normalize();
    newBinormal.normalize();
    
    // 결과 설정
    normal = newNormal;
    binormal = newBinormal;
}

// 축 방향 슬라이딩 적용
MPoint offsetCurveAlgorithm::applyAxialSliding(const MPoint& point,
                                    const offsetCurveInfluence& influence,
                                    double slidingFactor)
{
    // 축 방향 슬라이딩 효과 계산
    MVector tangent = influence.getTangent();
    
    // 슬라이딩 거리 계산
    double slidingDistance = influence.getBindLocalPoint().z * slidingFactor;
    
    // 축 방향으로 이동
    return point + (tangent * slidingDistance);
}

// 포즈 블렌딩 적용
MPoint offsetCurveAlgorithm::applyPoseBlending(const MPoint& deformedPoint, 
                                    unsigned int vertexIndex,
                                    double blendWeight)
{
    // 포즈 타겟이 없거나 인덱스가 범위 밖이면 변형 없음
    if (mPoseTargetPoints.length() <= vertexIndex) {
        return deformedPoint;
    }
    
    // 포즈 타겟 위치
    MPoint targetPoint = mPoseTargetPoints[vertexIndex];
    
    // 블렌드 계산
    return deformedPoint * (1.0 - blendWeight) + targetPoint * blendWeight;
}

// 프레넷 프레임 생성
MMatrix offsetCurveAlgorithm::createFrenetFrame(const MVector& tangent, 
                                     const MVector& normal, 
                                     const MVector& binormal, 
                                     const MPoint& origin)
{
    // 프레넷 프레임 행렬 생성
    double matrixArray[4][4] = {
        {tangent.x, normal.x, binormal.x, origin.x},
        {tangent.y, normal.y, binormal.y, origin.y},
        {tangent.z, normal.z, binormal.z, origin.z},
        {0.0, 0.0, 0.0, 1.0}
    };
    return MMatrix(matrixArray);
}

// 병렬 변형 계산 태스크
void offsetCurveAlgorithm::parallelDeformationTask(void* data, MThreadRootTask* root)
{
    offsetCurveTaskData* taskData = static_cast<offsetCurveTaskData*>(data);
    
    MPointArray& points = *(taskData->points);
    const std::vector<offsetCurveData>& curveData = *(taskData->curveData);
    std::map<unsigned int, offsetCurveVertexData>& vertexData = *(taskData->vertexData);
    const offsetCurveControlParams& params = taskData->params;
    BaseOffsetCurveStrategy* strategy = taskData->strategy;
    
    // 할당된 점들에 대해 변형 계산
    for (unsigned int i = taskData->startIdx; i < taskData->endIdx; ++i) {
        auto it = vertexData.find(i);
        if (it == vertexData.end() || i >= points.length()) {
            continue;
        }
        
        offsetCurveVertexData& vData = it->second;
        unsigned int vertexIdx = vData.vertexIndex;
        MPoint& currentPoint = points[vertexIdx];
        const MPoint& originalPoint = vData.originalPosition;
        std::vector<offsetCurveInfluence>& influences = vData.influences;
        
        if (influences.empty()) {
            continue;
        }
        
        // 단일 스레드 로직과 동일하게 변형 계산
        // 이 부분은 computeDeformation 메서드의 단일 스레드 코드와 동일함
        
        // 변형된 위치 계산
        MPoint deformedPoint(0.0, 0.0, 0.0, 1.0);
        double totalWeight = 0.0;
        
        // 각 영향 곡선 기여도 계산
        for (const offsetCurveInfluence& influence : influences) {
            if (influence.getCurveIndex() < 0 || 
                influence.getCurveIndex() >= static_cast<int>(curveData.size())) {
                continue;
            }
            
            const offsetCurveData& cData = curveData[influence.getCurveIndex()];
            
            // 바인드 포즈 매트릭스
            MMatrix bindMatrix = influence.getBindMatrix();
            MMatrix bindMatrixInverse = bindMatrix.inverse();
            
            // 영향 곡선 위의 해당 지점 위치
            MPoint curvePoint;
            cData.getPoint(influence.getParamU(), curvePoint);
            
            // 현재 탄젠트, 노멀, 바이노멀 계산
            MVector currentTangent, currentNormal, currentBinormal;
            strategy->computeFrenetFrame(cData, influence.getParamU(), 
                                       currentTangent, currentNormal, currentBinormal);
            
            // 현재 로컬 프레임 매트릭스
            double currentMatrixArray[4][4] = {
                {currentTangent.x, currentNormal.x, currentBinormal.x, curvePoint.x},
                {currentTangent.y, currentNormal.y, currentBinormal.y, curvePoint.y},
                {currentTangent.z, currentNormal.z, currentBinormal.z, curvePoint.z},
                {0.0, 0.0, 0.0, 1.0}
            };
            MMatrix currentMatrix(currentMatrixArray);
            
            // 바인드 포즈에서 현재 포즈까지의 변환 매트릭스
            MMatrix transformMatrix = currentMatrix * bindMatrixInverse;
            
            // 오프셋 위치 계산
            MPoint localBindPoint = influence.getBindLocalPoint();
            MPoint transformedPoint = localBindPoint * transformMatrix;
            
            // 가중치 적용
            deformedPoint.x += transformedPoint.x * influence.getWeight();
            deformedPoint.y += transformedPoint.y * influence.getWeight();
            deformedPoint.z += transformedPoint.z * influence.getWeight();
            totalWeight += influence.getWeight();
        }
        
        // 가중치 정규화
        if (totalWeight > 0.0) {
            deformedPoint = deformedPoint / totalWeight;
        } else {
            deformedPoint = originalPoint;
        }
        
        // 최종 위치 설정
        currentPoint = deformedPoint;
    }
}