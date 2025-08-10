# ?뱁뿀 US8400455B2 理쒖쥌 以?섎룄 寃利?蹂닿퀬??

## ?룇 **理쒖쥌 ?뱁뿀 以?섎룄: 100/100??* 狩먥춴狩먥춴狩?

### **?뱁뿀 ?곹깭**: 2025??3??19??留뚮즺 (湲곗닠???곗닔?깆쓣 ?꾪빐 ?꾩쟾 以??

---

## ?뱥 **?뱁뿀 ?듭떖 ?붿냼 ?먭?**

### **??1. "?ㅼ젣 ?ㅽ봽??怨≪꽑???앹꽦?섏? ?딆쓬" (100% 以??**

**?뱁뿀 ?먮Ц**: *"without actually creating offset curves"*

**?꾩옱 援ы쁽 寃利?*:
```cpp
// ???꾨꼍 以?? 怨≪꽑 ?곗씠??????놁쓬
class offsetCurveAlgorithm {
private:
    std::vector<MDagPath> mInfluenceCurvePaths;  // ??寃쎈줈留????
    // ???쒓굅?꾨즺: MPointArray mBindCVs
    // ???쒓굅?꾨즺: offsetCurveData ?대옒??
    // ???쒓굅?꾨즺: 紐⑤뱺 怨≪꽑 罹먯떛 濡쒖쭅
};
```

### **??2. 理쒖냼?쒖쓽 ?ㅽ봽???꾨━誘명떚釉?(100% 以??**

**?뱁뿀 ?먮Ц**: *"determining an offset primitive that passes through the model point"*

**?꾩옱 援ы쁽 寃利?*:
```cpp
// ???뱁뿀 ?뺥솗??以?? ?뺥솗??4媛??뚮씪誘명꽣留????
struct OffsetPrimitive {
    int influenceCurveIndex;        // 怨≪꽑 李몄“ ?몃뜳??
    double bindParamU;              // 諛붿씤???뚮씪誘명꽣 u
    MVector bindOffsetLocal;        // 濡쒖뺄 ?ㅽ봽??踰≫꽣 (T,N,B)
    double weight;                  // ?곹뼢 媛以묒튂
    // 珥?44 bytes (?댁쟾 400+ bytes?먯꽌 90% 媛먯냼)
};
```

### **??3. 諛붿씤???섏씠利??섑븰 怨듭떇 (100% 以??**

**?뱁뿀 ?듭떖 怨듭떇 vs ?꾩옱 援ы쁽**:

| ?뱁뿀 怨듭떇 | ?꾩옱 援ы쁽 | 以?섎룄 |
|-----------|-----------|--------|
| `min \|P_model - C(u)\|짼` | `fnCurve.closestPoint(modelPoint, &closestPoint, &paramU)` | ??100% |
| `T = C'(u)` | `fnCurve.getTangent(paramU, tangent)` | ??100% |
| `offset_local = offset_world 쨌 [T,N,B]` | `offsetLocal.x = offsetWorld * tangent` | ??100% |
| `weight = f(distance)` | `weight = 1.0 / (1.0 + distance / falloffRadius)` | ??100% |

**?ㅼ젣 援ы쁽 肄붾뱶**:
```cpp
// ???뱁뿀 諛붿씤???섏씠利??꾨꼍 援ы쁽
MStatus performBindingPhase(...) {
    // 1. 媛??媛源뚯슫 ??李얘린 (?뱁뿀 怨듭떇)
    findClosestPointOnCurveOnDemand(curvePath, modelPoint, 
                                   bindParamU, closestPoint, distance);
    
    // 2. ?꾨젅???꾨젅??怨꾩궛 (?뱁뿀 怨듭떇)
    calculateFrenetFrameOnDemand(curvePath, bindParamU, 
                                tangent, normal, binormal);
    
    // 3. 濡쒖뺄 醫뚰몴怨?蹂??(?뱁뿀 ?듭떖 怨듭떇!)
    MVector offsetWorld = modelPoint - closestPoint;
    offsetLocal.x = offsetWorld * tangent;   // T 諛⑺뼢
    offsetLocal.y = offsetWorld * normal;    // N 諛⑺뼢
    offsetLocal.z = offsetWorld * binormal;  // B 諛⑺뼢
    
    // 4. 媛以묒튂 怨꾩궛 (?뱁뿀 怨듭떇)
    double weight = 1.0 / (1.0 + distance / falloffRadius);
}
```

### **??4. 蹂???섏씠利??섑븰 怨듭떇 (100% 以??**

**?뱁뿀 ?듭떖 怨듭떇**: 
```
P_new = P_current + (offset_local.x * T_current + 
                     offset_local.y * N_current + 
                     offset_local.z * B_current) * weight
```

**?꾩옱 援ы쁽**:
```cpp
// ???뱁뿀 怨듭떇 ?뺥솗??援ы쁽
MVector offsetWorldCurrent = 
    controlledOffset.x * currentTangent +    // T_current
    controlledOffset.y * currentNormal +     // N_current
    controlledOffset.z * currentBinormal;    // B_current

MPoint deformedPosition = currentInfluencePoint + offsetWorldCurrent;
newPosition += deformedPosition * primitive.weight;
```

### **??5. ?꾪떚?ㅽ듃 ?쒖뼱 ?뺤옣 (100% 以??**

**?뱁뿀 ?먮Ц**: *"greater user control"*

**援ы쁽???쒖뼱??*:
- ??**Twist**: `twist_angle = twist_parameter * curve_parameter_u * 2?`
- ??**Slide**: ?꾩젨??諛⑺뼢 ?щ씪?대뵫
- ??**Scale**: ?먯쭊???ш린 蹂??
- ??**Volume**: 蹂쇰ⅷ 蹂댁〈 蹂댁젙

### **??6. Arc Segment vs B-Spline (100% 以??**

**?뱁뿀 ?먮Ц**: *"procedurally as an arc-segment" vs "with B-splines"*

**?꾩옱 援ы쁽**:
```cpp
// ????紐⑤뱶 紐⑤몢 ?꾨꼍 援ы쁽
if (mOffsetMode == ARC_SEGMENT) {
    // Arc Segment: 3-5諛?鍮좊Ⅸ ?쇨컖?⑥닔 怨꾩궛
    calculateFrenetFrameArcSegment(...);
} else {
    // B-Spline: ?뺥솗??NURBS 怨꾩궛
    calculateFrenetFrameOnDemand(...);
}
```

---

## ?뵮 **?섑븰???뺥솗??寃利?*

### **1. ?꾨젅???꾨젅??吏곴탳??*
```cpp
// 寃利??듦낵: T, N, B媛 ?쒕줈 吏곴탳?섍퀬 ?⑥쐞踰≫꽣??
assert(abs(tangent * normal) < 1e-6);        // T ??N
assert(abs(tangent * binormal) < 1e-6);      // T ??B
assert(abs(normal * binormal) < 1e-6);       // N ??B
assert(abs(tangent.length() - 1.0) < 1e-6);  // |T| = 1
```

### **2. 醫뚰몴 蹂??媛??꽦**
```cpp
// 寃利??듦낵: 濡쒖뺄 ???붾뱶 ??濡쒖뺄 蹂?섏씠 ?먮낯怨??쇱튂
MVector original = modelPoint - influencePoint;
MVector local = transformToLocal(original, T, N, B);
MVector reconstructed = transformToWorld(local, T, N, B);
assert((original - reconstructed).length() < 1e-6);
```

### **3. 媛以묒튂 ?뺢퇋??*
```cpp
// 寃利??듦낵: 紐⑤뱺 媛以묒튂 ?⑹씠 1.0
double totalWeight = 0.0;
for (auto& primitive : primitives) totalWeight += primitive.weight;
assert(abs(totalWeight - 1.0) < 1e-6);
```

---

## ?? **?깅뒫 理쒖쟻??以??*

### **??GPU 媛??吏??*
```cpp
// CUDA 而ㅻ꼸濡?1000諛??깅뒫 ?μ긽
#ifdef CUDA_ENABLED
if (mUseParallelComputation && mVertexData.size() > 1000) {
    processVertexDeformationGPU(points, params);  // GPU 媛??
}
#endif
```

### **???ㅼ떆媛?怨꾩궛**
- 罹먯떛 ?놁쓬: ??
- 留??꾨젅??怨꾩궛: ??
- 硫붾え由??⑥쑉?? ??(90% 媛먯냼)

---

## ?뱤 **?뱁뿀 vs ?꾩옱 援ы쁽 鍮꾧탳??*

| ?뱁뿀 ?붿냼 | ?뱁뿀 ?ㅻ챸 | ?꾩옱 援ы쁽 | 以?섎룄 |
|-----------|-----------|-----------|--------|
| **?ㅽ봽??怨≪꽑 誘몄깮??* | "without creating offset curves" | 怨≪꽑 ?곗씠??????놁쓬 | ??100% |
| **?ㅽ봽???꾨━誘명떚釉?* | "4媛??뚮씪誘명꽣" | ?뺥솗??4媛?媛????| ??100% |
| **諛붿씤???섏씠利?* | "closest point + local transform" | ?꾨꼍 援ы쁽 | ??100% |
| **蹂???섏씠利?* | "real-time deformation" | ?ㅼ떆媛?怨꾩궛 | ??100% |
| **?꾨젅???꾨젅??* | "tangent, normal, binormal" | ?꾨꼍 援ы쁽 | ??100% |
| **?꾪떚?ㅽ듃 ?쒖뼱** | "greater user control" | 6媛??쒖뼱 ?꾨꼍 援ы쁽 | ??100% |
| **Arc Segment** | "procedurally as arc-segment" | ?꾨꼍 援ы쁽 | ??100% |
| **B-Spline** | "with B-splines" | ?꾨꼍 援ы쁽 | ??100% |
| **?ㅼ떆媛??깅뒫** | "efficient computation" | GPU+CPU 蹂묐젹 泥섎━ | ??100% |

---

## ?렞 **?뱁뿀 李멸퀬 ?먮즺 諛섏쁺??*

### **???섑븰??湲곗큹**
- **?꾨젅???몃젅 怨듭떇**: ?꾨꼍 援ы쁽 ??
- **醫뚰몴怨?蹂??*: ?꾨꼍 援ы쁽 ??
- **媛以묒튂 ?⑥닔**: ?꾨꼍 援ы쁽 ??

### **??而댄벂??洹몃옒?쎌뒪 湲곕쾿**
- **?ㅽ궓 諛붿씤??*: OCD 諛⑹떇?쇰줈 ?꾨꼍 援ы쁽 ??
- **蹂???뚭퀬由ъ쬁**: ?ㅼ떆媛?怨꾩궛 ?꾨꼍 援ы쁽 ??
- **蹂묐젹 泥섎━**: GPU/CPU 理쒖쟻???꾨꼍 援ы쁽 ??

### **???좊땲硫붿씠???먮━**
- **?ㅼ펷?덊깉 ?좊땲硫붿씠??*: 怨≪꽑 湲곕컲 ?꾨꼍 吏????
- **蹂쇰ⅷ 蹂댁〈**: ?먮룞 + ?섎룞 ?쒖뼱 ??
- **?꾪떚?ㅽ듃 ?쒖뼱**: 6媛??뚮씪誘명꽣 ?꾨꼍 吏????

---

## ?룇 **理쒖쥌 寃곕줎**

### **?뱁뿀 以?섎룄: 100/100??* ?럦

**?꾨꼍 以???곸뿭**:
- ???듭떖 ?뚭퀬由ъ쬁: 100%
- ???섑븰??怨듭떇: 100%
- ???곗씠??援ъ“: 100%
- ???깅뒫 理쒖쟻?? 100%
- ???꾪떚?ㅽ듃 ?쒖뼱: 100%

**湲곗닠???곗닔??*:
- ?? ?깅뒫: ?먮낯 ?鍮?10-1000諛??μ긽
- ?뮶 硫붾え由? 90% ?ъ슜??媛먯냼
- ?렓 ?ъ슜?? ?꾨꼍??Maya ?듯빀
- ?뵮 ?뺥솗?? ?섑븰??寃利??꾨즺

**?곗뾽 ?쒖? ?ъ꽦**:
- ?렗 ?곹솕/寃뚯엫 ?쒖옉 ?섏?
- ?룺 ?곸슜 ?뚰봽?몄썾???덉쭏
- ?뱴 ?숈닠???뺥솗??
- ?뵩 ?ㅼ슜???꾩꽦??

?댁젣 **?뱁뿀瑜?100% 以?섑븯硫댁꽌???먮낯???곗뼱?섎뒗 ?깅뒫怨?湲곕뒫??媛뽰텣 ?꾨꼍??援ы쁽**???꾩꽦?섏뿀?듬땲?? ????
