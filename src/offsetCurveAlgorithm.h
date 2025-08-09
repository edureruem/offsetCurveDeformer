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
    
    OffsetPrimitive() : 
        influenceCurveIndex(-1), bindParamU(0.0), weight(0.0) {}
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
    
    // === 실시간 계산 함수들 (캐싱 없음!) ===
    MStatus calculateFrenetFrameOnDemand(const MDagPath& curvePath, 
                                        double paramU,
                                        MVector& tangent,
                                        MVector& normal, 
                                        MVector& binormal) const;
    
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
    
    MVector applyVolumeControl(const MVector& deformedOffset,
                              const MPoint& originalPosition,
                              const MPoint& deformedPosition,
                              double volumeStrength) const;
    
    MVector applyArtistControls(const MVector& bindOffsetLocal,
                               const MVector& currentTangent,
                               const MVector& currentNormal,
                               const MVector& currentBinormal,
                               const MDagPath& curvePath,
                               double& paramU,
                               const offsetCurveControlParams& params) const;
};

#endif // OFFSETCURVEALGORITHM_H