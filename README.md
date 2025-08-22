# OffsetCurveDeformer (OCD)

Maya용 Offset Curve Deformation 플러그인으로, [US8400455B2 특허](https://patents.google.com/patent/US8400455B2/en)의 기술을 구현한 고급 스키닝 시스템입니다.

## 주요 특징

### 🎯 특허 기반 기술
- **Offset Curve Deformation**: 각 모델 포인트에 대해 영향 곡선으로부터 오프셋된 별도의 곡선 생성
- **Non-local Influence**: 포인트 기반이 아닌 곡선 기반의 영향 메커니즘
- **Volume Preservation**: 기존 스키닝 기법의 볼륨 손실, "candy wrapper" 핀칭, 자체 교차 등의 아티팩트 최소화

### 🚀 성능 최적화
- **CPU 멀티스레딩**: MThreadPool을 활용한 병렬 처리
- **GPU 가속**: Maya 2016+ 버전에서 OpenCL 기반 GPU 가속 지원
- **AVX 최적화**: Advanced Vector Extensions를 활용한 벡터 연산 최적화

### 🎨 곡선 타입 지원
- **B-spline**: 일반적인 지오메트리(어깨, 가슴, 목 등)에 적합
- **Arc-segment**: 특수한 형태(팔꿈치, 손가락 관절 등)에 최적화

## 빌드 방법

### Windows
```bash
# Maya 버전별 자동 빌드
build.bat

# 수동 빌드 (특정 Maya 버전)
mkdir build.2025
cd build.2025
cmake -A x64 -T v141 -DMAYA_VERSION=2025 ../
cmake --build . --target install --config Release
```

### Linux/macOS
```bash
mkdir build
cd build
cmake -DMAYA_VERSION=2025 ../
make install
```

## 사용법

### Python 명령어
```python
import maya.cmds as cmds

# 영향 곡선과 변형될 지오메트리 선택
curves = cmds.ls(sl=True, type='transform')[:-1]  # 마지막은 변형될 지오메트리
geometry = cmds.ls(sl=True, type='transform')[-1]

# OCD 디포머 생성
ocd_node = cmds.offsetCurve(curves, geometry, 
                           offsetDistance=1.0, 
                           falloffRadius=2.0, 
                           curveType=0,  # 0: B-spline, 1: Arc-segment
                           name='myOCD')
```

### MEL 명령어
```mel
// 영향 곡선과 변형될 지오메트리 선택
select -r curve1 curve2 geometry1;

// OCD 디포머 생성
offsetCurve -offsetDistance 1.0 -falloffRadius 2.0 -curveType 0 -name "myOCD";
```

## 파라미터 설명

| 파라미터 | 설명 | 기본값 | 범위 |
|---------|------|--------|------|
| `offsetDistance` | 오프셋 곡선까지의 거리 | 1.0 | 0.001 ~ 100.0 |
| `falloffRadius` | 영향 감쇠 반경 | 2.0 | 0.1 ~ 100.0 |
| `curveType` | 곡선 타입 (0: B-spline, 1: Arc-segment) | 0 | 0 ~ 1 |
| `numTasks` | CPU 멀티스레딩 태스크 수 | 32 | 1 ~ 64 |
| `envelope` | 디포머 강도 | 1.0 | 0.0 ~ 1.0 |

## 아키텍처

### 핵심 컴포넌트
- **OffsetCurveDeformer**: 메인 디포머 노드 (MPxDeformerNode 상속)
- **OffsetCurveCmd**: Maya 명령어 인터페이스 (MPxCommand 상속)
- **OffsetCurveAlgorithm**: OCD 알고리즘 구현
- **OffsetCurveGPU**: GPU 가속 구현 (MPxGPUDeformer 상속)

### 데이터 흐름
1. **바인딩 단계**: 각 모델 포인트에 대해 오프셋 곡선 생성
2. **변형 단계**: 스켈레톤 애니메이션에 따른 오프셋 곡선 변형
3. **렌더링**: 변형된 오프셋 곡선을 기반으로 최종 지오메트리 계산

## 특허 기술 상세

### Offset Curve Deformation
특허의 핵심 아이디어는 각 모델 포인트에 대해 영향 곡선으로부터 오프셋된 별도의 곡선을 생성하는 것입니다:

```
기존 방식: "point on a stick" - 점 기반 영향
OCD 방식: 각 포인트마다 개별 오프셋 곡선 - 곡선 기반 영향
```

### 볼륨 보존 메커니즘
- **Bend 내부**: 자체 교차를 피하기 위해 포인트가 밖으로 미끄러짐
- **Bend 외부**: 스트레칭을 줄이기 위해 포인트가 안으로 미끄러짐

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 특허 기술을 구현한 것입니다. 상업적 사용 시 해당 특허의 라이선스를 확인하시기 바랍니다.

## 기여

이 프로젝트는 cvWrap의 패턴을 기반으로 하며, Maya API의 모범 사례를 따릅니다. 버그 리포트, 기능 제안, 코드 기여를 환영합니다.

## 참고 자료

- [US8400455B2 특허](https://patents.google.com/patent/US8400455B2/en)
- [cvWrap 프로젝트](https://github.com/chadmv/cvwrap)
- [Maya API 문서](https://help.autodesk.com/view/MAYAUL/2025/ENU/Maya-API-Documentation/)
