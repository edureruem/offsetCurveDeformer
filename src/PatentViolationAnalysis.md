# 특허 US8400455B2 위반 사항 분석

## 🚨 현재 구현의 주요 특허 위반 사항들

### 1. 실제 곡선 데이터 저장 (특허 위반!)

**특허 원칙**: "without actually creating offset curves"
**현재 구현**: 실제 곡선 데이터를 메모리에 저장

```cpp
// ❌ 특허 위반: offsetCurveData.h에서
class offsetCurveData {
private:
    MPointArray mBindCVs;         // 바인드 포즈의 CV 위치 저장
    MPointArray mCurrentCVs;      // 현재 CV 위치 저장
    MMatrixArray mBindMatrices;   // 바인드 포즈 행렬 저장
    MDoubleArray mOrientations;   // CV 방향값 저장
    // ... 더 많은 곡선 데이터 저장
};
```

**문제점**:
- 메모리에 실제 곡선 정보를 캐싱
- `cacheBindPoseData()`, `computeOrientations()` 등으로 곡선 데이터 저장
- 특허의 "실제로 곡선을 생성하지 않는다" 원칙 위반

### 2. Maya 곡선 객체에 과도한 의존 (특허 위반!)

```cpp
// ❌ 특허 위반: offsetCurveData.cpp에서
void offsetCurveData::initialize(const MDagPath& curvePath) {
    MFnNurbsCurve fnCurve(mCurvePath);  // 실제 곡선 객체 생성
    fnCurve.getCVs(mCurrentCVs);        // CV 데이터 추출해서 저장
    mLength = fnCurve.length();         // 길이 계산해서 저장
    // ... 더 많은 곡선 정보 추출 및 저장
}
```

**문제점**:
- 43개의 `MFnNurbsCurve` 사용 지점 발견
- 곡선 정보를 미리 계산해서 저장하는 방식
- 실시간 계산이 아닌 캐싱 방식 사용

### 3. 잘못된 오프셋 프리미티브 구현

**특허 원칙**: 각 모델 포인트별로 개별 오프셋 프리미티브
**현재 구현**: 복잡한 구조체로 과도한 정보 저장

```cpp
// ❌ 현재 구현: 너무 복잡하고 불필요한 데이터 포함
struct PatentCompliantOffsetPrimitive {
    int primitiveType;
    int influenceCurveIndex;
    double bindParamU;
    MPoint bindInfluencePoint;
    MVector bindOffsetVector;
    MVector bindTangent, bindNormal, bindBinormal;
    double bindCurvature;
    double weight;
    MMatrix localToInfluenceTransform;  // 불필요한 매트릭스
    // ... 너무 많은 데이터
};
```

### 4. 변형 계산 알고리즘 오류

**특허 원칙**: 
1. 현재 영향 곡선에서 프레넷 프레임 계산
2. 바인드 시점의 오프셋 벡터를 현재 프레넷 프레임에 적용

**현재 구현**: 복잡한 변환 매트릭스와 캐시된 데이터 사용

## ✅ 특허 준수를 위한 올바른 구현

### 1. 오프셋 프리미티브 (특허 준수)

```cpp
// ✅ 특허 준수: 최소한의 수학적 파라미터만
struct PatentCorrectOffsetPrimitive {
    int influenceCurveIndex;      // 영향 곡선 인덱스 (경로 참조만)
    double bindParamU;            // 바인드 시점의 곡선 파라미터
    MVector bindOffsetVector;     // 바인드 시점의 오프셋 벡터 (로컬)
    double weight;                // 영향 가중치
    
    // 이게 전부! 다른 데이터는 실시간 계산
};
```

### 2. 실시간 계산 함수 (특허 준수)

```cpp
// ✅ 특허 준수: 실시간으로만 계산, 저장 안 함
MStatus calculateCurrentFrenetFrame(const MDagPath& curvePath, 
                                   double paramU,
                                   MVector& tangent,
                                   MVector& normal, 
                                   MVector& binormal) {
    // Maya 곡선에서 즉석으로 계산, 저장하지 않음
    MFnNurbsCurve fnCurve(curvePath);
    fnCurve.getTangent(paramU, tangent);
    // normal, binormal 즉석 계산
    // 결과를 메모리에 저장하지 않음!
}
```

### 3. 올바른 변형 계산 (특허 알고리즘)

```cpp
// ✅ 특허 준수: 올바른 변형 계산
MPoint deformVertex(const PatentCorrectOffsetPrimitive& offsetPrimitive,
                   const std::vector<MDagPath>& influenceCurves) {
    // 1. 현재 영향 곡선에서 프레넷 프레임 계산 (실시간)
    MVector currentTangent, currentNormal, currentBinormal;
    calculateCurrentFrenetFrame(influenceCurves[offsetPrimitive.influenceCurveIndex],
                               offsetPrimitive.bindParamU,
                               currentTangent, currentNormal, currentBinormal);
    
    // 2. 현재 영향 곡선 상의 점 계산 (실시간)
    MPoint currentInfluencePoint;
    MFnNurbsCurve fnCurve(influenceCurves[offsetPrimitive.influenceCurveIndex]);
    fnCurve.getPointAtParam(offsetPrimitive.bindParamU, currentInfluencePoint);
    
    // 3. 바인드 시점의 오프셋 벡터를 현재 프레넷 프레임에 적용
    MVector worldOffsetVector = 
        offsetPrimitive.bindOffsetVector.x * currentTangent +
        offsetPrimitive.bindOffsetVector.y * currentNormal +
        offsetPrimitive.bindOffsetVector.z * currentBinormal;
    
    // 4. 새로운 정점 위치 = 현재 영향점 + 변환된 오프셋 벡터
    return currentInfluencePoint + worldOffsetVector * offsetPrimitive.weight;
}
```

## 📊 특허 준수도 점수

- **현재 구현**: 30/100 (주요 원칙 위반)
- **필요한 개선**: 70% 리팩토링 필요

### 핵심 개선 사항:
1. `offsetCurveData` 클래스 제거 또는 대폭 단순화
2. 모든 곡선 데이터 캐싱 제거
3. 실시간 계산 함수로 전환
4. 오프셋 프리미티브 구조 단순화
5. 특허 알고리즘에 맞는 변형 계산 구현
