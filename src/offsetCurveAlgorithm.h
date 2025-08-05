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

#include "offsetCurveData.h"
#include "offsetCurveStrategy.h"

// Offset Curve 오프셋 방식 정의
enum offsetCurveOffsetMode {
    ARC_SEGMENT = 0,    // 아크 세그먼트 방식
    B_SPLINE = 1        // B-스플라인 방식
};

// 정점 데이터 저장 구조체
struct offsetCurveVertexData {
    unsigned int vertexIndex;            // 정점 인덱스
    MPoint originalPosition;             // 원본 위치
    std::vector<offsetCurveInfluence> influences;   // 영향 데이터
};

// 아티스트 제어 파라미터 구조체
struct offsetCurveControlParams {
    double volumeStrength;       // 볼륨 보존 강도
    double slideEffect;          // 슬라이딩 효과
    double rotationDistribution; // 회전 분포
    double scaleDistribution;    // 스케일 분포
    double twistDistribution;    // 꼬임 분포
    double axialSliding;         // 축 방향 슬라이딩
    double normalOffset;         // 법선 오프셋 강도
    bool enablePoseBlending;     // 포즈 블렌딩 활성화
    MPointArray poseTarget;      // 포즈 타겟
    double poseWeight;           // 포즈 가중치
    
    // 기본 생성자
    offsetCurveControlParams() : 
        volumeStrength(1.0),
        slideEffect(0.0),
        rotationDistribution(1.0),
        scaleDistribution(1.0),
        twistDistribution(1.0),
        axialSliding(0.0),
        normalOffset(1.0),
        enablePoseBlending(false),
        poseWeight(0.0)
    {}
};

// 병렬 처리를 위한 작업 데이터
struct offsetCurveTaskData {
    unsigned int startIdx;
    unsigned int endIdx;
    MPointArray* points;
    const std::vector<offsetCurveData>* curveData;
    std::map<unsigned int, offsetCurveVertexData>* vertexData;
    offsetCurveControlParams params;
    offsetCurveOffsetMode offsetMode;
    IOffsetCurveStrategy* strategy;
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
    
    // 변형 계산 (아티스트 제어 파라미터 추가)
    MStatus computeDeformation(MPointArray& points,
                             const offsetCurveControlParams& params);
    
    // 오프셋 곡선 계산
    MStatus computeOffsetCurves(const MPointArray& points,
                              const std::vector<MDagPath>& curvePaths);
    
    // 병렬 처리 활성화/비활성화
    void enableParallelComputation(bool enable);
    
    // 포즈 타겟 설정
    void setPoseTarget(const MPointArray& poseTarget);
    
private:
    // 각 정점의 영향 맵 계산
    void computeInfluenceWeights(unsigned int vertexIndex,
                               const MPoint& point,
                               const std::vector<offsetCurveData>& curves,
                               double falloffRadius,
                               int maxInfluences);
    
    // 부피 보존 계산 (향상된 알고리즘)
    MVector computeVolumePreservation(const MPoint& originalPoint,
                                    const MPoint& deformedPoint,
                                    const std::vector<offsetCurveInfluence>& influences,
                                    double volumeStrength);
    
    // 슬라이딩 효과 계산 (확장 기능)
    MPoint computeSlideEffect(const MPoint& originalPoint,
                            const MPoint& deformedPoint,
                            const std::vector<offsetCurveInfluence>& influences,
                            double slideEffect);
    
    // 회전 분포 적용
    void applyRotationDistribution(MMatrix& transformMatrix, 
                                 const offsetCurveInfluence& influence,
                                 double rotationFactor);
    
    // 스케일 분포 적용
    void applyScaleDistribution(MMatrix& transformMatrix, 
                              const offsetCurveInfluence& influence,
                              double scaleFactor);
    
    // 꼬임 분포 적용
    void applyTwistDistribution(MVector& normal, 
                              MVector& binormal,
                              const offsetCurveInfluence& influence,
                              double twistFactor);
    
    // 축 방향 슬라이딩 적용
    MPoint applyAxialSliding(const MPoint& point,
                           const offsetCurveInfluence& influence,
                           double slidingFactor);
    
    // 병렬 처리 작업 함수
    static void parallelDeformationTask(void* data, MThreadRootTask* root);
    
    // 포즈 블렌딩 적용
    MPoint applyPoseBlending(const MPoint& deformedPoint, 
                           unsigned int vertexIndex,
                           double blendWeight);

private:
    // 내부 상태 및 데이터
    offsetCurveOffsetMode mOffsetMode;
    std::vector<offsetCurveData> mCurveDataList;
    std::map<unsigned int, offsetCurveVertexData> mVertexDataMap;
    bool mUseParallelComputation;
    
    // 전략 패턴 구현
    std::unique_ptr<IOffsetCurveStrategy> mStrategy;
    
    // 입력 메시 및 포즈 데이터
    MPointArray mOriginalPoints;
    MPointArray mPoseTargetPoints;
};

#endif // OFFSETCURVEALGORITHM_H