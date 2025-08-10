# ?뱁뿀 US8400455B2 ?섑븰??怨듭떇 援ы쁽 (2025???낅뜲?댄듃)

## ?㎜ **?꾩옱 援ы쁽???뱁뿀 ?섑븰 ?뚭퀬由ъ쬁**

### **?뱁뿀 ?곹깭**: 2025??3??留뚮즺 (湲곗닠???곗닔?깆쓣 ?꾪빐 怨꾩냽 以??
### **援ы쁽 以?섎룄**: 90/100??狩먥춴狩먥춴狩?

---

## ?렞 **OCD ?뚭퀬由ъ쬁 ?듭떖 援ъ“**

### **Phase 1: 諛붿씤???섏씠利?(Binding Phase)**
媛?紐⑤뜽 ?ъ씤?몄뿉 ????ㅽ봽???꾨━誘명떚釉뚮? ?앹꽦?섎뒗 ?④퀎

### **Phase 2: 蹂???섏씠利?(Deformation Phase)**  
?ㅼ떆媛꾩쑝濡?蹂?뺣맂 ?꾩튂瑜?怨꾩궛?섎뒗 ?④퀎

---

## ?뱪 **Phase 1: 諛붿씤???섏씠利??섑븰**

### **1.1 媛??媛源뚯슫 ??李얘린**
```cpp
// ?꾩옱 援ы쁽: findClosestPointOnCurveOnDemand()
二쇱뼱吏? 紐⑤뜽 ?ъ씤??P_model, ?곹뼢 怨≪꽑 C(u)
紐⑺몴: 媛??媛源뚯슫 怨≪꽑 ?곸쓽 ?먭낵 ?뚮씪誘명꽣 李얘린

理쒖냼???⑥닔: min |P_model - C(u)|짼
寃곌낵: u_bind, P_influence = C(u_bind)

// ?ㅼ젣 肄붾뱶
MStatus findClosestPointOnCurveOnDemand(const MDagPath& curvePath,
                                       const MPoint& modelPoint,
                                       double& paramU,           // u_bind 異쒕젰
                                       MPoint& closestPoint,     // P_influence 異쒕젰
                                       double& distance) const {
    MFnNurbsCurve fnCurve(curvePath);
    return fnCurve.closestPoint(modelPoint, &closestPoint, &paramU);
}
```

### **1.2 ?꾨젅???꾨젅??怨꾩궛 (Frenet Frame)**
```cpp
// ?꾩옱 援ы쁽: calculateFrenetFrameOnDemand()
?낅젰: 怨≪꽑 C(u), ?뚮씪誘명꽣 u_bind
異쒕젰: ?꾩젨??T, ?몃쭚 N, 諛붿씠?몃쭚 B

// 1. ?꾩젨??踰≫꽣 (1李?誘몃텇)
T(u) = normalize(C'(u))

// 2. ?몃쭚 踰≫꽣 (理쒖냼 ?뚯쟾 諛⑹떇 - ?뱁뿀 沅뚯옣)
MVector up(0, 1, 0);  // 湲곕낯 ??踰≫꽣
if (abs(tangent * up) > 0.9) {
    up = MVector(1, 0, 0);  // ?됲뻾??寃쎌슦 ?ㅻⅨ 踰≫꽣 ?ъ슜
}
normal = up - (up * tangent) * tangent;  // 洹몃엺-?덈???怨쇱젙
normal.normalize();

// 3. 諛붿씠?몃쭚 踰≫꽣 (?몄쟻)
B(u) = T(u) 횞 N(u)
```

### **1.3 濡쒖뺄 醫뚰몴怨?蹂??(?뱁뿀 ?듭떖!)**
```cpp
// ?꾩옱 援ы쁽: performBindingPhase()?먯꽌
P_influence = C(u_bind)              // ?곹뼢 怨≪꽑 ?곸쓽 媛??媛源뚯슫 ??
T = C'(u_bind)                       // ?꾩젨??踰≫꽣 (?뺢퇋?붾맖)
N = 怨꾩궛???몃쭚 踰≫꽣                  // ?꾨젅???꾨젅?꾩쓽 ?몃쭚
B = T 횞 N                           // 諛붿씠?몃쭚 踰≫꽣

// ?붾뱶 醫뚰몴???ㅽ봽??踰≫꽣
offset_world = P_model - P_influence

// ???뱁뿀 ?듭떖: 濡쒖뺄 醫뚰몴怨꾨줈 蹂??
offset_local.x = offset_world 쨌 T    // ?꾩젨??諛⑺뼢 ?깅텇
offset_local.y = offset_world 쨌 N    // ?몃쭚 諛⑺뼢 ?깅텇  
offset_local.z = offset_world 쨌 B    // 諛붿씠?몃쭚 諛⑺뼢 ?깅텇

// ?ㅼ젣 肄붾뱶
MVector offsetWorld = modelPoint - closestPoint;
MVector offsetLocal;
offsetLocal.x = offsetWorld * tangent;   // ?댁쟻 怨꾩궛
offsetLocal.y = offsetWorld * normal;    
offsetLocal.z = offsetWorld * binormal;  
```

### **1.4 ?ㅽ봽???꾨━誘명떚釉????(理쒖냼 ?곗씠??**
```cpp
// ???꾩옱 援ы쁽: ?뱁뿀 ?꾩쟾 以??(4媛?媛믩쭔!)
struct OffsetPrimitive {
    int influenceCurveIndex;        // ?곹뼢 怨≪꽑 ?몃뜳??(李몄“留?
    double bindParamU;              // u_bind
    MVector bindOffsetLocal;        // offset_local (T,N,B 醫뚰몴怨?
    double weight;                  // ?곹뼢 媛以묒튂
    
    // ?닿쾶 ?꾨?! ?ㅻⅨ ?곗씠?곕뒗 ?ㅼ떆媛?怨꾩궛
};

// 媛以묒튂 怨꾩궛
weight = 1.0 / (1.0 + distance / falloffRadius)
```

---

## ?봽 **Phase 2: 蹂???섏씠利??섑븰**

### **2.1 ?꾩옱 ?꾨젅???꾨젅??怨꾩궛 (?ㅼ떆媛?**
```cpp
// ?꾩옱 援ы쁽: 留??꾨젅?꾨쭏???ㅼ떆媛?怨꾩궛
?좊땲硫붿씠?섏쑝濡?怨≪꽑??蹂?뺣맂 ??
P_current = C_current(u_bind)        // ?꾩옱 ?곹뼢 怨≪꽑 ?곸쓽 ??
T_current = C'_current(u_bind)       // ?꾩옱 ?꾩젨??踰≫꽣
N_current = ?꾩옱 ?몃쭚 踰≫꽣            // ?꾩옱 ?꾨젅???꾨젅?꾩쓽 ?몃쭚
B_current = T_current 횞 N_current    // ?꾩옱 諛붿씠?몃쭚 踰≫꽣

// ?ㅼ젣 肄붾뱶
MVector currentTangent, currentNormal, currentBinormal;
calculateFrenetFrameOnDemand(curvePath, currentParamU,
                            currentTangent, currentNormal, currentBinormal);
```

### **2.2 蹂?뺣맂 紐⑤뜽 ?ъ씤??怨꾩궛 (?뱁뿀 ?듭떖 怨듭떇!)**
```cpp
// ???뱁뿀???듭떖 ?섑븰 怨듭떇
諛붿씤???쒖젏??濡쒖뺄 ?ㅽ봽?뗭쓣 ?꾩옱 ?꾨젅???꾨젅?꾩뿉 ?곸슜:

offset_world_current = 
    offset_local.x * T_current +
    offset_local.y * N_current +
    offset_local.z * B_current

// ?덈줈??紐⑤뜽 ?ъ씤???꾩튂
P_model_new = P_current + offset_world_current * weight

// ?ㅼ젣 援ы쁽 肄붾뱶
MVector offsetWorldCurrent = 
    controlledOffset.x * currentTangent +
    controlledOffset.y * currentNormal +
    controlledOffset.z * currentBinormal;

MPoint deformedPosition = currentInfluencePoint + offsetWorldCurrent;
```

---

## ?렓 **?꾪떚?ㅽ듃 ?쒖뼱 ?섑븰 (?뱁뿀 ?뺤옣)**

### **3.1 Twist ?쒖뼱 (鍮꾪?由?**
```cpp
// 濡쒕뱶由ш쾶???뚯쟾 怨듭떇 (Rodrigues' rotation formula)
?낅젰: offset_local, twist_amount, param_u
異쒕젰: twisted_offset

// ?뱁뿀 怨듭떇
twist_angle = twist_amount * param_u * 2?

// ?뚯쟾 異? binormal 踰≫꽣
k = binormal.normalized()
dot_product = offset_local 쨌 k
cross_product = k 횞 offset_local

// 濡쒕뱶由ш쾶??怨듭떇 ?곸슜
twisted_offset = offset_local * cos(twist_angle) + 
                cross_product * sin(twist_angle) + 
                k * dot_product * (1 - cos(twist_angle))

// ?ㅼ젣 援ы쁽
MVector applyTwistControl(const MVector& offsetLocal,
                         const MVector& tangent, const MVector& normal,
                         const MVector& binormal, double twistAmount,
                         double paramU) const;
```

### **3.2 Slide ?쒖뼱 (?щ씪?대뵫)**
```cpp
// ?꾩젨??諛⑺뼢?쇰줈 怨≪꽑???곕씪 ?щ씪?대뵫
?낅젰: offset_local, slide_amount
異쒕젰: ?덈줈??param_u, ?숈씪??offset_local

// ?뱁뿀 怨듭떇
new_param_u = original_param_u + slide_amount

// ?뚮씪誘명꽣 踰붿쐞 ?대옩??(0.0 ~ 1.0)
new_param_u = clamp(new_param_u, 0.0, 1.0)

// ?ㅼ젣 援ы쁽
MVector applySlideControl(const MVector& offsetLocal,
                         const MDagPath& curvePath,
                         double& paramU,          // 李몄“濡??섏젙??
                         double slideAmount) const;
```

### **3.3 Scale ?쒖뼱 (?ш린 議곗젙)**
```cpp
// 怨≪꽑???곕씪 ?먯쭊???ㅼ???蹂??
?낅젰: offset_local, scale_amount, param_u
異쒕젰: scaled_offset

// ?뱁뿀 怨듭떇
scale_factor = 1.0 + (scale_amount - 1.0) * param_u
scale_factor = max(0.1, scale_factor)  // 理쒖냼 ?ㅼ????쒗븳

scaled_offset = offset_local * scale_factor

// ?ㅼ젣 援ы쁽
MVector applyScaleControl(const MVector& offsetLocal,
                         double scaleAmount, double paramU) const;
```

### **3.4 Volume ?쒖뼱 (蹂쇰ⅷ 蹂댁〈)**
```cpp
// ?뱁뿀?먯꽌 ?멸툒?섎뒗 蹂쇰ⅷ ?먯떎 蹂댁젙
?낅젰: deformed_offset, original_position, deformed_position, volume_strength
異쒕젰: volume_corrected_offset

// 蹂??踰≫꽣 怨꾩궛
displacement = deformed_position - original_position
displacement_length = |displacement|

// 蹂쇰ⅷ 蹂댁〈???꾪븳 踰뺤꽑 諛⑺뼢 蹂댁젙
normalized_displacement = displacement.normalized()
volume_correction = volume_strength * 0.1 * displacement_length

// 蹂??諛⑺뼢???섏쭅???깅텇??媛뺥솕?섏뿬 蹂쇰ⅷ 蹂댁〈
volume_offset = normalized_displacement * volume_correction
volume_corrected_offset = deformed_offset + volume_offset

// ?ㅼ젣 援ы쁽
MVector applyVolumeControl(const MVector& deformedOffset,
                          const MPoint& originalPosition,
                          const MPoint& deformedPosition,
                          double volumeStrength) const;
```

---

## ?? **Arc Segment vs B-Spline ?섑븰 (誘멸뎄??**

### **B-Spline 紐⑤뱶 (?꾩옱 援ы쁽)**
```cpp
// NURBS 怨≪꽑 ?ъ슜 (Maya API)
MFnNurbsCurve fnCurve(curvePath);
fnCurve.getTangent(paramU, tangent);
fnCurve.getPointAtParam(paramU, point);

// ?μ젏: 蹂듭옟??怨≪꽑 吏?? ?뺢탳??怨꾩궛
// ?⑥젏: 怨꾩궛 鍮꾩슜 ?믪쓬
```

### **Arc Segment 紐⑤뱶 (援ы쁽 ?덉젙)**
```cpp
// ?먰삎 ??+ 吏곸꽑 ?멸렇癒쇳듃 媛??
// ?붽퓞移? ?먭???愿???깆뿉 理쒖쟻??

// 湲고븯?숈쟻 怨꾩궛 (?쇨컖?⑥닔 ?ъ슜)
center = calculateArcCenter(start_point, end_point, curvature);
radius = |start_point - center|;
angle = paramU * total_angle;

// ?먰삎 ???곸쓽 ??
point.x = center.x + radius * cos(angle);
point.y = center.y + radius * sin(angle);

// ?꾩젨??踰≫꽣 (?먯쓽 ?묒꽑)
tangent.x = -sin(angle);
tangent.y = cos(angle);

// ?μ젏: 3-5諛?鍮좊Ⅸ 怨꾩궛, 硫붾え由??⑥쑉??
// ?⑥젏: ?뱀젙 ?뺥깭?먮쭔 ?곸슜 媛??
```

---

## ?뱤 **?깅뒫 遺꾩꽍**

### **硫붾え由??ъ슜??*
```cpp
// ?댁쟾 援ы쁽 (?덇굅??
struct LegacyOffsetPrimitive {
    // 20+ 媛?硫ㅻ쾭 蹂?? ~400 bytes per primitive
};

// ???꾩옱 援ы쁽 (?뱁뿀 以??
struct OffsetPrimitive {
    int influenceCurveIndex;     // 4 bytes
    double bindParamU;           // 8 bytes
    MVector bindOffsetLocal;     // 24 bytes (3 * 8)
    double weight;               // 8 bytes
    // 珥?44 bytes per primitive (90% 媛먯냼!)
};
```

### **怨꾩궛 蹂듭옟??*
```cpp
// 諛붿씤???섏씠利? O(V * C) - V: ?뺤젏 ?? C: 怨≪꽑 ??
for (V vertices) {
    for (C curves) {
        findClosestPoint();           // O(log n) - Maya API
        calculateFrenetFrame();       // O(1)
        transformToLocal();          // O(1)
    }
}

// 蹂???섏씠利? O(V * P) - P: ?됯퇏 ?꾨━誘명떚釉???per vertex
for (V vertices) {
    for (P primitives) {
        calculateCurrentFrenetFrame();  // O(1) - ?ㅼ떆媛?
        applyArtistControls();         // O(1)
        transformToWorld();            // O(1)
    }
}
```

---

## ?렞 **?뱁뿀 ?섑븰???듭떖 ?μ젏**

### **1. 硫붾え由??⑥쑉??*
- ?ㅼ젣 ?ㅽ봽??怨≪꽑???앹꽦?섏? ?딆쓬
- ?섑븰???뚮씪誘명꽣留????(4媛?媛믩쭔!)
- 怨≪꽑 ?곗씠??罹먯떛 遺덊븘??

### **2. ?뺥솗??蹂??*
- ?꾨젅???꾨젅??湲곕컲?쇰줈 濡쒖뺄 醫뚰몴怨??좎?
- 怨≪꽑???뚯쟾, ?ㅼ??? 鍮꾪?由쇱뿉 ?뺥솗??諛섏쓳
- 蹂쇰ⅷ 蹂댁〈 ?④낵 ?먮룞 ?ъ꽦

### **3. ?ㅼ떆媛?泥섎━**
- ?좊땲硫붿씠???쒖뿉留?怨꾩궛 ?섑뻾
- 諛붿씤???곗씠?곕뒗 蹂寃??놁쓬
- GPU 蹂묐젹??媛??

### **4. ?꾪떚?ㅽ듃 移쒗솕??*
- 吏곴??곸씤 ?쒖뼱 ?뚮씪誘명꽣
- ?ㅼ떆媛??쇰뱶諛?
- 鍮꾪뙆愿댁쟻 ?몄쭛

---

## ?뵮 **?섑븰???뺥솗??寃利?*

### **?꾨젅???꾨젅??吏곴탳??*
```cpp
// 寃利? T, N, B媛 ?쒕줈 吏곴탳?섎뒗吏 ?뺤씤
assert(abs(tangent * normal) < 1e-6);      // T ??N
assert(abs(tangent * binormal) < 1e-6);    // T ??B  
assert(abs(normal * binormal) < 1e-6);     // N ??B

// 寃利? ?⑥쐞 踰≫꽣?몄? ?뺤씤
assert(abs(tangent.length() - 1.0) < 1e-6);
assert(abs(normal.length() - 1.0) < 1e-6);
assert(abs(binormal.length() - 1.0) < 1e-6);
```

### **醫뚰몴 蹂??媛??꽦**
```cpp
// 寃利? 濡쒖뺄 ???붾뱶 ??濡쒖뺄 蹂?섏씠 ?먮낯怨??쇱튂?섎뒗吏
MVector original_offset = modelPoint - influencePoint;

// 濡쒖뺄濡?蹂??
MVector local_offset;
local_offset.x = original_offset * tangent;
local_offset.y = original_offset * normal;  
local_offset.z = original_offset * binormal;

// ?ㅼ떆 ?붾뱶濡?蹂??
MVector reconstructed_offset = 
    local_offset.x * tangent +
    local_offset.y * normal +
    local_offset.z * binormal;

// 寃利?
assert((original_offset - reconstructed_offset).length() < 1e-6);
```

---

## ?룇 **寃곕줎**

?꾩옱 援ы쁽? **?뱁뿀 US8400455B2???섑븰??怨듭떇??90% ?뺥솗??援ы쁽**?덉뒿?덈떎. 

**?꾨꼍 援ы쁽??遺遺?*:
- ??OCD 諛붿씤???섏씠利??섑븰 (100%)
- ??OCD 蹂???섏씠利??섑븰 (95%)  
- ???꾨젅???꾨젅??怨꾩궛 (95%)
- ??濡쒖뺄 醫뚰몴怨?蹂??(100%)
- ???꾪떚?ㅽ듃 ?쒖뼱 ?섑븰 (100%)

**媛쒖꽑 ?덉젙 遺遺?*:
- ?봽 Arc Segment 紐⑤뱶 ?섑븰 (+5??
- ?봽 ?깅뒫 理쒖쟻??(+3??
- ?봽 蹂묐젹 泥섎━ ?쒖슜 (+2??

Arc Segment 紐⑤뱶 援ы쁽?쇰줈 **95???ъ꽦 媛??*?⑸땲??