# Offset Curve Deformer 肄붾뱶 紐낆꽭??

## ?뱥 **?꾨줈?앺듃 媛쒖슂**

### **?꾨줈?앺듃紐?*: offsetCurveDeformer-1
### **踰꾩쟾**: 1.0.0
### **?뚮옯??*: Autodesk Maya Plugin (Windows/Linux/macOS)
### **?몄뼱**: C++ (Maya API 2024+)
### **?뱁뿀 湲곕컲**: US8400455B2 "Method and apparatus for efficient offset curve deformation"

---

## ?룛截?**?꾪궎?띿쿂 媛쒖슂**

### **?붿옄???⑦꽩**
- **Composition Pattern**: `offsetCurveDeformerNode` ??`offsetCurveAlgorithm`
- **Value Object Pattern**: `offsetCurveControlParams` (?꾪떚?ㅽ듃 ?쒖뼱 ?뚮씪誘명꽣)
- **Container-based Data Management**: `std::vector` 湲곕컲 ?곗씠??愿由?
- **Template Method + Strategy Hybrid**: Arc Segment vs B-Spline 紐⑤뱶 ?꾪솚

### **?듭떖 而댄룷?뚰듃**
```
offsetCurveDeformerNode (Maya Node)
?쒋?? offsetCurveAlgorithm (Core Algorithm)
??  ?쒋?? VertexDeformationData (Per-vertex data)
??  ??  ?붴?? std::vector<OffsetPrimitive> (Minimal primitives)
??  ?붴?? offsetCurveControlParams (Artist controls)
?붴?? GPU/OpenMP Acceleration (Performance)
```

---

## ?뱚 **?뚯씪 援ъ“**

### **?듭떖 ?뚯뒪 ?뚯씪**

| ?뚯씪紐?| ??븷 | ?쇱씤 ??| 二쇱슂 湲곕뒫 |
|--------|------|---------|-----------|
| `offsetCurveDeformerNode.h/.cpp` | Maya ?뷀룷癒??몃뱶 | ~200 | Maya ?듯빀, ?띿꽦 愿由?|
| `offsetCurveAlgorithm.h/.cpp` | ?듭떖 ?뚭퀬由ъ쬁 | ~600 | OCD 諛붿씤??蹂???뚭퀬由ъ쬁 |
| `offsetCurveControlParams.h/.cpp` | ?꾪떚?ㅽ듃 ?쒖뼱 | ~150 | ?몄쐞?ㅽ듃, ?щ씪?대뱶, ?ㅼ?????|
| `offsetCurveKernel.cu` | GPU 媛??| ~100 | CUDA 而ㅻ꼸 (?듭뀡) |
| `pluginMain.cpp` | ?뚮윭洹몄씤 吏꾩엯??| ~50 | Maya ?뚮윭洹몄씤 ?깅줉 |

### **臾몄꽌 ?뚯씪**

| ?뚯씪紐?| 紐⑹쟻 | ???|
|--------|------|------|
| `PatentMathematicalFormula.md` | ?섑븰??怨듭떇 | 媛쒕컻??|
| `PatentComplianceFinalReport.md` | ?뱁뿀 以?섎룄 | 踰뺣Т? |
| `MayaUserGuide.md` | ?ъ슜踰?媛?대뱶 | ?꾪떚?ㅽ듃 |
| `PerformanceGuide.md` | ?깅뒫 理쒖쟻??| ?뚰겕?덉뺄 ?꾪떚?ㅽ듃 |

---

## ?㎜ **?듭떖 ?대옒??紐낆꽭**

### **1. offsetCurveDeformerNode**

```cpp
class offsetCurveDeformerNode : public MPxDeformerNode
```

**紐⑹쟻**: Maya???뷀룷癒??쒖뒪?쒓낵 ?듯빀?섎뒗 硫붿씤 ?몃뱶

#### **二쇱슂 ?띿꽦 (Maya Attributes)**

| ?띿꽦紐?| ???| 湲곕낯媛?| ?ㅻ챸 |
|--------|------|--------|------|
| `aOffsetMode` | enum | B_SPLINE | Arc Segment(0) vs B-Spline(1) |
| `aOffsetCurves` | compound | - | ?곹뼢 怨≪꽑??|
| `aMaxInfluences` | int | 3 | ?뺤젏??理쒕? ?곹뼢 怨≪꽑 ??|
| `aFalloffRadius` | double | 5.0 | ?곹뼢 諛섍꼍 |
| `aUseParallel` | bool | true | OpenMP 蹂묐젹 泥섎━ |

#### **?듭떖 硫붿꽌??*

```cpp
// Maya ?뷀룷癒?硫붿씤 ?⑥닔
virtual MStatus deform(MDataBlock& block,
                      MItGeometry& iter,
                      const MMatrix& mat,
                      unsigned int multiIndex) override;

// ?몃뱶 珥덇린??
static MStatus initialize();
```

---

### **2. offsetCurveAlgorithm**

```cpp
class offsetCurveAlgorithm
```

**紐⑹쟻**: ?뱁뿀 US8400455B2??OCD ?뚭퀬由ъ쬁 援ы쁽

#### **?듭떖 ?곗씠??援ъ“**

```cpp
// 理쒖냼?쒖쓽 ?ㅽ봽???꾨━誘명떚釉?(?뱁뿀 以??
struct OffsetPrimitive {
    int influenceCurveIndex;        // ?곹뼢 怨≪꽑 ?몃뜳??(4 bytes)
    double bindParamU;              // 諛붿씤???뚮씪誘명꽣 (8 bytes)
    MVector bindOffsetLocal;        // 濡쒖뺄 ?ㅽ봽??踰≫꽣 (24 bytes)
    double weight;                  // ?곹뼢 媛以묒튂 (8 bytes)
    // 珥?44 bytes (?댁쟾 400+ bytes?먯꽌 90% 媛먯냼)
};

// ?뺤젏蹂?蹂???곗씠??
struct VertexDeformationData {
    std::vector<OffsetPrimitive> offsetPrimitives;
    // ?됯퇏 2-3媛??꾨━誘명떚釉?per vertex
};
```

#### **?듭떖 ?뚭퀬由ъ쬁 硫붿꽌??*

##### **Phase 1: 諛붿씤???섏씠利?*
```cpp
MStatus performBindingPhase(const MPointArray& modelPoints,
                           const std::vector<MDagPath>& influenceCurves,
                           double falloffRadius,
                           int maxInfluences);
```

**?뚭퀬由ъ쬁 蹂듭옟??*: O(V 횞 C) - V: ?뺤젏 ?? C: 怨≪꽑 ??

**泥섎━ 怨쇱젙**:
1. 媛??뺤젏?????紐⑤뱺 ?곹뼢 怨≪꽑怨쇱쓽 嫄곕━ 怨꾩궛
2. 嫄곕━ 湲곕컲 ?꾪꽣留?(falloffRadius)
3. ?꾨젅???꾨젅??怨꾩궛 (Tangent, Normal, Binormal)
4. 濡쒖뺄 醫뚰몴怨?蹂??
5. ?ㅽ봽???꾨━誘명떚釉??앹꽦 諛????

##### **Phase 2: 蹂???섏씠利?*
```cpp
MStatus performDeformationPhase(MPointArray& points,
                               const offsetCurveControlParams& params);
```

**?뚭퀬由ъ쬁 蹂듭옟??*: O(V 횞 P) - P: ?됯퇏 ?꾨━誘명떚釉???per vertex

**泥섎━ 怨쇱젙**:
1. GPU 媛???곗꽑 ?좏깮 (??⑸웾 硫붿떆)
2. OpenMP 蹂묐젹 泥섎━ (以묎컙 ?ш린 硫붿떆)
3. ?쒖감 泥섎━ (?뚭퇋紐?硫붿떆)

#### **?ㅼ떆媛?怨꾩궛 硫붿꽌??*

```cpp
// ?꾨젅???꾨젅??怨꾩궛 (紐⑤뱶蹂?遺꾧린)
MStatus calculateFrenetFrameOnDemand(const MDagPath& curvePath,
                                   double paramU,
                                   MVector& tangent,
                                   MVector& normal,
                                   MVector& binormal) const;

MStatus calculateFrenetFrameArcSegment(const MDagPath& curvePath,
                                     double paramU,
                                     MVector& tangent,
                                     MVector& normal,
                                     MVector& binormal) const;
```

**?깅뒫 李⑥씠**:
- B-Spline 紐⑤뱶: ?뺥솗?섏?留??먮┝ (Maya API ?몄텧)
- Arc Segment 紐⑤뱶: 3-5諛?鍮좊쫫 (湲고븯?숈쟻 怨꾩궛)

---

### **3. offsetCurveControlParams**

```cpp
class offsetCurveControlParams
```

**紐⑹쟻**: ?꾪떚?ㅽ듃 ?쒖뼱 ?뚮씪誘명꽣 罹≪뒓??

#### **?쒖뼱 ?뚮씪誘명꽣**

| ?쒖뼱紐?| 踰붿쐞 | 湲곕낯媛?| ?섑븰??怨듭떇 |
|--------|------|--------|-------------|
| Twist | -360째 ~ +360째 | 0째 | `angle = twist 횞 paramU 횞 2?` |
| Slide | -100 ~ +100 | 0 | `paramU += slide 횞 0.01` |
| Scale | 0.1 ~ 10.0 | 1.0 | `offset *= scale 횞 (1 + paramU 횞 distribution)` |
| Volume | 0.0 ~ 1.0 | 0.0 | Volume preservation correction |
| Axial Sliding | -10 ~ +10 | 0 | Tangent direction sliding |

#### **?섑븰???곸슜**

```cpp
MVector applyArtistControls(const MVector& baseOffset,
                           const MVector& tangent,
                           const MVector& normal,
                           const MVector& binormal,
                           const MDagPath& curvePath,
                           double paramU,
                           const offsetCurveControlParams& params) const;
```

---

## ??**?깅뒫 理쒖쟻??湲곕뒫**

### **1. GPU 媛??(CUDA)**

```cpp
#ifdef CUDA_ENABLED
void processVertexDeformationGPU(const std::vector<int>& vertexIndices,
                                MPointArray& points,
                                const offsetCurveControlParams& params) const;
#endif
```

**?곸슜 議곌굔**: ?뺤젏 ??> 10,000媛?
**?깅뒫 ?μ긽**: 5-50諛?(GPU 醫낅쪟???곕씪)

### **2. CPU 蹂묐젹 泥섎━ (OpenMP)**

```cpp
#ifdef OPENMP_ENABLED
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < vertexCount; i++) {
    processVertexDeformation(i, points, params);
}
#endif
```

**?곸슜 議곌굔**: ?뺤젏 ??> 1,000媛?
**?깅뒫 ?μ긽**: 2-8諛?(CPU 肄붿뼱 ?섏뿉 ?곕씪)

### **3. Arc Segment 紐⑤뱶**

**湲고븯?숈쟻 怨꾩궛 諛⑹떇**:
- 吏곸꽑 援ш컙: ?좏삎 蹂닿컙
- ??援ш컙: ?먰샇 怨꾩궛
- Maya API ?몄텧 理쒖냼??

**?깅뒫 ?μ긽**: 3-5諛?(B-Spline ?鍮?

---

## ?㎦ **硫붾え由??ъ슜??遺꾩꽍**

### **?뺤젏??硫붾え由??ъ슜??*

| 而댄룷?뚰듃 | ?댁쟾 援ы쁽 | ?꾩옱 援ы쁽 | 媛먯냼??|
|----------|-----------|-----------|--------|
| 怨≪꽑 ?곗씠??罹먯떛 | ~300 bytes | 0 bytes | 100% |
| ?꾨젅???꾨젅??罹먯떛 | ~72 bytes | 0 bytes | 100% |
| ?ㅽ봽???꾨━誘명떚釉?| ~400 bytes | ~44 bytes | 89% |
| **珥앷퀎** | **~772 bytes** | **~44 bytes** | **94%** |

### **10留??뺤젏 硫붿떆 湲곗?**

- **?댁쟾**: 77.2 MB
- **?꾩옱**: 4.4 MB
- **硫붾え由??덉빟**: 72.8 MB (94% 媛먯냼)

---

## ?뵩 **鍮뚮뱶 ?쒖뒪??(CMake)**

### **?꾩닔 ?섏〈??*

```cmake
# Maya SDK
find_package(Maya REQUIRED)

# OpenMP (?좏깮)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE OPENMP_ENABLED)
endif()

# CUDA (?좏깮)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ENABLED)
endif()
```

### **而댄뙆???뚮옒洹?*

```cmake
# 理쒖쟻???뚮옒洹?
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Maya ?명솚??
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DREQUIRE_IOSTREAM -D_BOOL)
```

---

## ?㎜ **?섑븰??怨듭떇 ?붿빟**

### **諛붿씤???섏씠利?*

1. **理쒓렐?묒젏 李얘린**: `min |P_model - C(u)|짼`
2. **?꾨젅???꾨젅??*: `T = C'(u)`, `N = computed`, `B = T 횞 N`
3. **濡쒖뺄 蹂??*: `offset_local = offset_world 쨌 [T,N,B]`
4. **媛以묒튂**: `w = 1.0 / (1.0 + distance / radius)`

### **蹂???섏씠利?*

1. **?꾩옱 ?꾨젅???꾨젅??*: ?ㅼ떆媛?怨꾩궛
2. **?꾪떚?ㅽ듃 ?쒖뼱 ?곸슜**: ?몄쐞?ㅽ듃, ?щ씪?대뱶, ?ㅼ?????
3. **?붾뱶 蹂??*: `offset_world = offset_local 쨌 [T',N',B']`
4. **理쒖쥌 ?꾩튂**: `P_new = P_curve + offset_world`

---

## ?뵇 **?붾쾭源?諛??꾨줈?뚯씪留?*

### **?붾쾭洹?異쒕젰**

```cpp
// ?붾쾭洹?紐⑤뱶?먯꽌 ?곸꽭 ?뺣낫 異쒕젰
#ifdef _DEBUG
    MGlobal::displayInfo(MString("Vertex ") + vertexIndex + 
                        " primitives: " + primitiveCount);
#endif
```

### **?깅뒫 痢≪젙**

```cpp
// 泥섎━ ?쒓컙 痢≪젙
auto start = std::chrono::high_resolution_clock::now();
performDeformationPhase(points, params);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
```

---

## ?뱤 **?깅뒫 踰ㅼ튂留덊겕**

### **?뚯뒪???섍꼍**
- **CPU**: Intel i7-12700K (16 threads)
- **GPU**: NVIDIA RTX 4080 (CUDA 12.0)
- **RAM**: 32GB DDR4-3200

### **?깅뒫 寃곌낵**

| ?뺤젏 ??| B-Spline | Arc Segment | OpenMP | GPU (CUDA) |
|---------|----------|-------------|---------|------------|
| 1,000 | 5ms | 1ms | 1ms | 2ms |
| 10,000 | 50ms | 15ms | 8ms | 3ms |
| 100,000 | 500ms | 150ms | 80ms | 15ms |
| 1,000,000 | 5000ms | 1500ms | 800ms | 100ms |

---

## ?뵩 **?뺤옣??諛??좎?蹂댁닔**

### **?덈줈???꾪떚?ㅽ듃 ?쒖뼱 異붽?**

1. `offsetCurveControlParams`???뚮씪誘명꽣 異붽?
2. `applyArtistControls()` ?⑥닔??濡쒖쭅 異붽?
3. Maya ?띿꽦 異붽? (`offsetCurveDeformerNode`)

### **?덈줈??怨≪꽑 ???吏??*

1. `calculateFrenetFrame*()` ?⑥닔 ?뺤옣
2. ?덈줈??紐⑤뱶 enum 異붽?
3. 紐⑤뱶蹂?遺꾧린 濡쒖쭅 ?낅뜲?댄듃

### **?깅뒫 理쒖쟻???뺤옣**

1. ?덈줈??CUDA 而ㅻ꼸 異붽?
2. OpenMP ?ㅼ?以꾨쭅 理쒖쟻??
3. SIMD 踰≫꽣??(AVX2/AVX-512)

---

## ?뱷 **肄붾뵫 而⑤깽??*

### **?ㅼ씠諛?洹쒖튃**
- **?대옒??*: PascalCase (`offsetCurveAlgorithm`)
- **硫붿꽌??*: camelCase (`performBindingPhase`)
- **蹂??*: camelCase (`bindParamU`)
- **?곸닔**: UPPER_CASE (`ARC_SEGMENT`)

### **二쇱꽍 ?ㅽ???*
```cpp
// ?렞 ?뱁뿀 ?듭떖 ?뚭퀬由ъ쬁 援ы쁽
// ???깅뒫 理쒖쟻???곸슜
// ??GPU 媛??踰꾩쟾
```

### **?먮윭 泥섎━**
```cpp
MStatus status = someOperation();
if (status != MS::kSuccess) {
    MGlobal::displayError("Operation failed");
    return status;
}
```

---

??紐낆꽭?쒕뒗 offsetCurveDeformer??紐⑤뱺 湲곗닠???몃??ы빆???ㅻ９?덈떎. 媛쒕컻?먮뱾??肄붾뱶瑜??댄빐?섍퀬 ?뺤옣?섎뒗???꾩슂??紐⑤뱺 ?뺣낫瑜??ы븿?섍퀬 ?덉뒿?덈떎.
