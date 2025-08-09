# 특허 US8400455B2 수학적 공식 분석

## 🧮 특허의 핵심 수학적 알고리즘

### 1. 바인딩 페이즈 (Binding Phase) 수학

#### 1.1 가장 가까운 점 찾기
```
주어진: 모델 포인트 P_model, 영향 곡선 C(u)
목표: 가장 가까운 곡선 상의 점과 파라미터 찾기

최소화 함수: min |P_model - C(u)|²
결과: u_bind, P_influence = C(u_bind)
```

#### 1.2 오프셋 벡터 계산 (로컬 좌표계)
```
P_influence = C(u_bind)  // 영향 곡선 상의 가장 가까운 점
T = C'(u_bind)           // 탄젠트 벡터 (정규화)
N = 계산된 노말 벡터      // 프레넷 프레임의 노말
B = T × N               // 바이노말 벡터

// 월드 좌표의 오프셋 벡터
offset_world = P_model - P_influence

// 로컬 좌표계로 변환 (이것이 핵심!)
offset_local.x = offset_world · T  // 탄젠트 방향 성분
offset_local.y = offset_world · N  // 노말 방향 성분  
offset_local.z = offset_world · B  // 바이노말 방향 성분
```

#### 1.3 오프셋 프리미티브 저장
```cpp
// 특허에서 저장하는 것 (최소한):
struct PatentOffsetPrimitive {
    int curve_index;           // 영향 곡선 인덱스
    double u_bind;             // 바인드 시점의 곡선 파라미터
    MVector offset_local;      // 로컬 좌표계의 오프셋 벡터
    double weight;             // 영향 가중치
};
```

### 2. 변형 페이즈 (Deformation Phase) 수학

#### 2.1 현재 프레넷 프레임 계산
```
// 애니메이션으로 곡선이 변형된 후
P_current = C_current(u_bind)     // 현재 영향 곡선 상의 점
T_current = C'_current(u_bind)    // 현재 탄젠트 벡터
N_current = 현재 노말 벡터         // 현재 프레넷 프레임의 노말
B_current = T_current × N_current // 현재 바이노말 벡터
```

#### 2.2 변형된 모델 포인트 계산 (특허의 핵심!)
```
// 바인드 시점의 로컬 오프셋을 현재 프레넷 프레임에 적용
offset_world_current = 
    offset_local.x * T_current +
    offset_local.y * N_current +
    offset_local.z * B_current

// 새로운 모델 포인트 위치
P_model_new = P_current + offset_world_current * weight
```

### 3. 프레넷 프레임 계산 (Frenet Frame)

#### 3.1 탄젠트 벡터
```
T(u) = normalize(C'(u))  // 곡선의 1차 미분을 정규화
```

#### 3.2 노말 벡터 (여러 방법)
```
방법 1 - 곡률 기반:
N(u) = normalize(T'(u))  // 탄젠트의 미분 (곡률 방향)

방법 2 - 최소 회전 (특허에서 선호):
N(u) = 이전 프레임으로부터 최소 회전으로 계산
```

#### 3.3 바이노말 벡터
```
B(u) = T(u) × N(u)  // 외적으로 계산
```

### 4. 가중치 계산

#### 4.1 거리 기반 가중치
```
distance = |P_model - P_influence|
weight = 1.0 / (1.0 + distance / falloff_radius)
```

#### 4.2 다중 곡선 영향시 정규화
```
total_weight = Σ weight_i
normalized_weight_i = weight_i / total_weight
```

## 🎯 특허 알고리즘의 핵심 장점

### 1. 메모리 효율성
- 실제 오프셋 곡선을 생성하지 않음
- 수학적 파라미터만 저장 (4개 값만!)
- 곡선 데이터 캐싱 불필요

### 2. 정확한 변형
- 프레넷 프레임 기반으로 로컬 좌표계 유지
- 곡선의 회전, 스케일, 비틀림에 정확히 반응
- 볼륨 보존 효과 자동 달성

### 3. 실시간 처리
- 애니메이션 시에만 계산 수행
- 바인딩 데이터는 변경 없음
- GPU 병렬화 가능

## ❌ 현재 구현과의 차이점

### 현재 구현의 문제:
1. **과도한 데이터 저장**: CV 배열, 행렬 배열 등
2. **복잡한 변환**: 불필요한 매트릭스 연산
3. **캐싱 방식**: 실시간 계산 대신 미리 계산해서 저장
4. **메모리 낭비**: 곡선 데이터를 중복 저장

### 특허 준수 구현:
1. **최소 데이터**: 4개 값만 저장 (curve_index, u_bind, offset_local, weight)
2. **단순 계산**: 벡터 내적과 외적만 사용
3. **실시간 방식**: 필요할 때만 Maya 곡선에서 계산
4. **메모리 효율**: 수학적 파라미터만 저장
