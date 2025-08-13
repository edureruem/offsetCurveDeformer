/**
 * offsetCurveTypes.h
 * OCD 공통 타입 및 구조체 정의
 * 소니 특허(US8400455) 기반으로 개선
 */

#ifndef OFFSETCURVETYPES_H
#define OFFSETCURVETYPES_H

// Maya 헤더들 (최소한만)
#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MObject.h>

// C++ 표준 라이브러리 (최소한만)
#include <vector>

// ✅ 오프셋 모드 enum 정의 (단일 정의)
enum offsetCurveOffsetMode {
    ARC_SEGMENT = 0,    // 아크 세그먼트 방식
    B_SPLINE = 1        // B-스플라인 방식
};

// ✅ OffsetPrimitive 구조체 정의 (특허 기술 완벽 보존)
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

// ✅ ArcSegment 구조체 정의 (특허 기술 보존)
struct ArcSegment {
    double startParamU;           // 시작 파라미터
    double endParamU;             // 끝 파라미터
    double curvatureMagnitude;    // 곡률 크기
    bool isLinear;                // 직선 여부
    double radius;                // 원의 반지름 (곡선인 경우)
    double totalAngle;            // 총 각도 (곡선인 경우)
    MPoint center;                // 원의 중심 (곡선인 경우)
    
    ArcSegment() : 
        startParamU(0.0), endParamU(1.0), curvatureMagnitude(0.0),
        isLinear(true), radius(0.0), totalAngle(0.0) {
        center = MPoint(0, 0, 0);
    }
};

// ✅ VertexDeformationData 구조체 정의 (특허 기술 보존)
struct VertexDeformationData {
    int vertexIndex;
    MPoint bindPosition;
    std::vector<OffsetPrimitive> offsetPrimitives;
    
    VertexDeformationData() : vertexIndex(-1) {
        bindPosition = MPoint(0, 0, 0);
    }
};

// ✅ Repository 인터페이스들 (데이터 접근 추상화)
class ICurveRepository {
public:
    virtual ~ICurveRepository() = default;
    
    // 곡선 데이터 관리
    virtual void addCurve(const MDagPath& curvePath) = 0;
    virtual void removeCurve(int index) = 0;
    virtual void clearCurves() = 0;
    
    // 곡선 데이터 접근
    virtual const std::vector<MDagPath>& getAllCurves() const = 0;
    virtual MDagPath getCurve(int index) const = 0;
    virtual int getCurveCount() const = 0;
    
    // 곡선 데이터 검증
    virtual bool hasCurve(const MDagPath& curvePath) const = 0;
    virtual bool isValidCurve(const MDagPath& curvePath) const = 0;
    
    // 캐시 관리 (const 문제 해결을 위해 추가)
    virtual void updateCurveValidityCache(const MDagPath& curvePath, bool isValid) = 0;
};

class IBindingRepository {
public:
    virtual ~IBindingRepository() = default;
    
    // 바인딩 데이터 관리
    virtual void addVertexBinding(int vertexIndex, const MPoint& bindPosition) = 0;
    virtual void addOffsetPrimitive(int vertexIndex, const OffsetPrimitive& primitive) = 0;
    virtual void removeVertexBinding(int vertexIndex) = 0;
    virtual void clearBindings() = 0;
    
    // 바인딩 데이터 접근
    virtual const std::vector<VertexDeformationData>& getAllVertexBindings() const = 0;
    virtual VertexDeformationData& getVertexBinding(int vertexIndex) = 0;
    virtual const std::vector<OffsetPrimitive>& getVertexPrimitives(int vertexIndex) const = 0;
    virtual int getBindingCount() const = 0;
    
    // 바인딩 데이터 검증
    virtual bool hasVertexBinding(int vertexIndex) const = 0;
    virtual bool isValidBinding(int vertexIndex) const = 0;
};

// ✅ DataFlowController 인터페이스 (데이터 흐름 관리)
class IDataFlowController {
public:
    virtual ~IDataFlowController() = default;
    
               // 데이터 흐름 제어
           virtual MStatus initializeDataFlow() = 0;
           virtual MStatus processDataFlow() = 0;
           virtual MStatus validateDataFlow() = 0;
           
           // 데이터 전송 및 동기화
           virtual MStatus synchronizeRepositories() = 0;
           virtual MStatus transferDataBetweenServices() = 0;
           
           // 데이터 흐름 상태 관리
           virtual bool isDataFlowValid() const = 0;
           virtual MStatus getDataFlowStatus() const = 0;
           
           // 에러 처리 및 복구
           virtual MStatus handleDataFlowError(const MStatus& error) = 0;
           virtual MStatus recoverDataFlow() = 0;
           
           // 추가 데이터 흐름 제어 메서드들
           virtual MStatus optimizeDataFlow() = 0;
           virtual MStatus monitorDataFlowPerformance() = 0;
           virtual MStatus cleanupDataFlow() = 0;
};

#endif // OFFSETCURVETYPES_H
