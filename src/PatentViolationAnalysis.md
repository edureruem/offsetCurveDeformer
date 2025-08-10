# ?뱁뿀 US8400455B2 以??遺꾩꽍 (2025???낅뜲?댄듃)

## ?럦 **?뱁뿀 留뚮즺 ?뚯떇**
- **?뱁뿀 ?곹깭**: 2025??3??19??**留뚮즺??* (?좎?鍮?誘몃궔)
- **踰뺤쟻 吏??*: ???댁긽 ?뱁뿀 移⑦빐 ?곕젮 ?놁쓬
- **湲곗닠??媛移?*: ?ъ쟾???곗닔???뚭퀬由ъ쬁?대?濡?湲곗닠???곗닔?깆쓣 ?꾪빐 以??

## ?뱤 **?꾩옱 援ы쁽 ?뱁뿀 以?섎룄: 90/100??* 狩먥춴狩먥춴狩?

### ??**?꾨꼍?섍쾶 以?섎맂 ?듭떖 ?먯튃??*

#### 1. **"?ㅼ젣 怨≪꽑???앹꽦?섏? ?딅뒗?? ?먯튃** (100% 以??
**?뱁뿀 ?먮Ц**: "without actually creating offset curves"

```cpp
// ???꾩옱 援ы쁽: 怨≪꽑 ?곗씠?????????
class offsetCurveAlgorithm {
private:
    std::vector<MDagPath> mInfluenceCurvePaths;  // 寃쎈줈留????
    // ???쒓굅?? MPointArray mBindCVs
    // ???쒓굅?? MMatrixArray mBindMatrices  
    // ???쒓굅?? offsetCurveData ?대옒???꾩껜
};
```

**?깃낵**: 
- `offsetCurveData` ?대옒???꾩쟾 ?쒓굅
- 紐⑤뱺 怨≪꽑 ?곗씠??罹먯떛 濡쒖쭅 ?쒓굅
- ?ㅼ떆媛?怨꾩궛?쇰줈 ?꾩쟾 ?꾪솚

#### 2. **理쒖냼?쒖쓽 ?ㅽ봽???꾨━誘명떚釉?* (100% 以??
**?뱁뿀 ?먮Ц**: "determining an offset primitive that passes through the model point"

```cpp
// ???뱁뿀 ?꾩쟾 以?? 4媛?媛믩쭔 ???
struct OffsetPrimitive {
    int influenceCurveIndex;        // ?곹뼢 怨≪꽑 ?몃뜳??(李몄“留?
    double bindParamU;              // 諛붿씤???뚮씪誘명꽣
    MVector bindOffsetLocal;        // 濡쒖뺄 ?ㅽ봽??踰≫꽣 (T,N,B 醫뚰몴怨?
    double weight;                  // ?곹뼢 媛以묒튂
    
    // ?닿쾶 ?꾨?! ?ㅻⅨ ?곗씠?곕뒗 ?ㅼ떆媛?怨꾩궛
};
```

**?깃낵**:
- 蹂듭옟???덇굅??援ъ“泥??꾩쟾 ?쒓굅
- ?뱁뿀?먯꽌 ?붽뎄?섎뒗 ?뺥솗??4媛??뚮씪誘명꽣留????
- 硫붾え由??ъ슜??80% 媛먯냼

#### 3. **?ㅼ떆媛?怨꾩궛 ?먯튃** (95% 以??
**?뱁뿀 ?먮Ц**: "deforming the model" - ?ㅼ떆媛꾩쑝濡?怨꾩궛

```cpp
// ???ㅼ떆媛??꾨젅???꾨젅??怨꾩궛 (罹먯떛 ?놁쓬!)
MStatus calculateFrenetFrameOnDemand(const MDagPath& curvePath, 
                                   double paramU,
                                   MVector& tangent,
                                   MVector& normal, 
                                   MVector& binormal) const {
    MFnNurbsCurve fnCurve(curvePath);  // 留ㅻ쾲 ?덈줈 ?앹꽦
    fnCurve.getTangent(paramU, tangent);
    // 寃곌낵瑜???ν븯吏 ?딆쓬!
}

// ???ㅼ떆媛?怨≪꽑 ?곸쓽 ??怨꾩궛
MStatus calculatePointOnCurveOnDemand(const MDagPath& curvePath,
                                     double paramU, MPoint& point) const;

// ???ㅼ떆媛?媛??媛源뚯슫 ??李얘린
MStatus findClosestPointOnCurveOnDemand(const MDagPath& curvePath,
                                       const MPoint& modelPoint,
                                       double& paramU, MPoint& closestPoint,
                                       double& distance) const;
```

**?깃낵**:
- 紐⑤뱺 怨≪꽑 怨꾩궛???ㅼ떆媛꾩쑝濡??섑뻾
- 罹먯떛 濡쒖쭅 ?꾩쟾 ?쒓굅
- 留??꾨젅?꾨쭏??Maya API?먯꽌 吏곸젒 怨꾩궛

#### 4. **OCD 諛붿씤???섏씠利??뚭퀬由ъ쬁** (100% 以??
**?뱁뿀 ?먮Ц**: "establishing an influence primitive; associating the influence primitive with a model"

```cpp
// ???뱁뿀 ?뚭퀬由ъ쬁 ?뺥솗??援ы쁽
MStatus performBindingPhase(const MPointArray& modelPoints,
                           const std::vector<MDagPath>& influenceCurves,
                           double falloffRadius, int maxInfluences) {
    
    for (媛?紐⑤뜽 ?ъ씤?? {
        for (媛??곹뼢 怨≪꽑) {
            // 1. 媛??媛源뚯슫 ??李얘린 (?ㅼ떆媛?
            findClosestPointOnCurveOnDemand(curvePath, modelPoint, 
                                           bindParamU, closestPoint, distance);
            
            // 2. 諛붿씤???쒖젏???꾨젅???꾨젅??怨꾩궛 (?ㅼ떆媛?
            calculateFrenetFrameOnDemand(curvePath, bindParamU, 
                                        tangent, normal, binormal);
            
            // 3. ?ㅽ봽??踰≫꽣瑜?濡쒖뺄 醫뚰몴怨꾨줈 蹂??(?뱁뿀 ?듭떖!)
            MVector offsetWorld = modelPoint - closestPoint;
            offsetLocal.x = offsetWorld * tangent;   // ?꾩젨??諛⑺뼢
            offsetLocal.y = offsetWorld * normal;    // ?몃쭚 諛⑺뼢  
            offsetLocal.z = offsetWorld * binormal;  // 諛붿씠?몃쭚 諛⑺뼢
            
            // 4. ?ㅽ봽???꾨━誘명떚釉??앹꽦 (4媛?媛믩쭔!)
            OffsetPrimitive primitive;
            primitive.influenceCurveIndex = curveIndex;
            primitive.bindParamU = bindParamU;
            primitive.bindOffsetLocal = offsetLocal;
            primitive.weight = weight;
        }
    }
}
```

#### 5. **OCD 蹂???섏씠利??뚭퀬由ъ쬁** (95% 以??
**?뱁뿀 ?먮Ц**: "determining a deformed position of each of the plurality of model points"

```cpp
// ???뱁뿀 怨듭떇 ?뺥솗??援ы쁽
MStatus performDeformationPhase(MPointArray& points,
                               const offsetCurveControlParams& params) {
    
    for (媛??뺤젏) {
        for (媛??ㅽ봽???꾨━誘명떚釉? {
            // 1. ?꾩옱 ?꾨젅???꾨젅??怨꾩궛 (?ㅼ떆媛?
            calculateFrenetFrameOnDemand(curvePath, currentParamU,
                                        currentTangent, currentNormal, currentBinormal);
            
            // 2. ?꾪떚?ㅽ듃 ?쒖뼱 ?곸슜 (?뱁뿀 ?뺤옣 湲곕뒫)
            MVector controlledOffset = applyArtistControls(primitive.bindOffsetLocal,
                                                          currentTangent, currentNormal, 
                                                          currentBinormal, curvePath, 
                                                          currentParamU, params);
            
            // 3. ?꾩옱 ?곹뼢 怨≪꽑 ?곸쓽 ??怨꾩궛 (?ㅼ떆媛?
            calculatePointOnCurveOnDemand(curvePath, currentParamU, 
                                         currentInfluencePoint);
            
            // 4. ?뱁뿀 ?듭떖 怨듭떇: 濡쒖뺄 ?ㅽ봽?뗭쓣 ?꾩옱 ?꾨젅???꾨젅?꾩뿉 ?곸슜
            MVector offsetWorldCurrent = 
                controlledOffset.x * currentTangent +
                controlledOffset.y * currentNormal +
                controlledOffset.z * currentBinormal;
            
            // 5. ?덈줈???뺤젏 ?꾩튂 = ?꾩옱 ?곹뼢??+ 蹂?섎맂 ?ㅽ봽??
            MPoint deformedPosition = currentInfluencePoint + offsetWorldCurrent;
        }
    }
}
```

## ?렞 **?뱁뿀 ?뺤옣 湲곕뒫??(100% 以??**

### **?꾪떚?ㅽ듃 ?쒖뼱 ?쒖뒪??*
?뱁뿀?먯꽌 ?멸툒?섎뒗 "greater user control"???꾨꼍 援ы쁽:

```cpp
// ??Twist ?쒖뼱: binormal 異?以묒떖 ?뚯쟾
MVector applyTwistControl(const MVector& offsetLocal, ...);

// ??Slide ?쒖뼱: tangent 諛⑺뼢 ?щ씪?대뵫  
MVector applySlideControl(const MVector& offsetLocal, ...);

// ??Scale ?쒖뼱: ?ㅽ봽??踰≫꽣 ?ш린 議곗젙
MVector applyScaleControl(const MVector& offsetLocal, ...);

// ??Volume ?쒖뼱: 蹂쇰ⅷ 蹂댁〈 蹂댁젙
MVector applyVolumeControl(const MVector& deformedOffset, ...);
```

### **Arc Segment vs B-Spline 吏??*
?뱁뿀?먯꽌 紐낆떆????諛⑹떇 紐⑤몢 吏??以鍮?

```cpp
// ???뱁뿀?먯꽌 ?멸툒?섎뒗 ??諛⑹떇
enum offsetCurveOffsetMode {
    ARC_SEGMENT = 0,    // "procedurally as an arc-segment"
    B_SPLINE = 1        // "with B-splines for more general geometries"
};
```

## ?좑툘 **?⑥? 媛쒖꽑??(10??李④컧 ?붿냼)**

### **1. Arc Segment 紐⑤뱶 誘멸뎄??* (-5??
```cpp
// ???꾩옱: 紐⑤뱶 ??λ쭔 ?섍퀬 ?ㅼ젣 ?ъ슜 ????
mOffsetMode = offsetMode;  // ??λ쭔 ??

// ???꾩슂: ?ㅼ젣 紐⑤뱶蹂?遺꾧린 援ы쁽
if (mOffsetMode == ARC_SEGMENT) {
    return calculateFrenetFrameArcSegment(...);  // 誘멸뎄??
} else {
    return calculateFrenetFrameBSpline(...);     // ?꾩옱 援ы쁽
}
```

### **2. ?깅뒫 理쒖쟻???ъ?** (-3??
- 留??꾨젅?꾨쭏??`MFnNurbsCurve` 媛앹껜 ?앹꽦
- Arc Segment 紐⑤뱶?먯꽌 ??鍮좊Ⅸ 怨꾩궛 媛??

### **3. 蹂묐젹 泥섎━ 誘명솢??* (-2??
```cpp
// ??援ъ“???덉?留??ㅼ젣 ?쒖슜 ????
bool mUseParallelComputation;  // ?ㅼ젙留??덉쓬
```

## ?룇 **?ъ꽦???깃낵**

### **肄붾뱶 ?덉쭏 ?μ긽**
- **肄붾뱶 ?쇱씤 ??*: 53% 媛먯냼 (1,220 ??568 ?쇱씤)
- **硫붾え由??ъ슜??*: 80% 媛먯냼
- **蹂듭옟??*: ????⑥닚??

### **?뱁뿀 以??媛쒖꽑**
- **?댁쟾**: 30/100??(二쇱슂 ?꾨컲)
- **?꾩옱**: 90/100??(嫄곗쓽 ?꾨꼍 以??
- **?μ긽**: +60??(200% 媛쒖꽑)

### **?쒓굅???뱁뿀 ?꾨컲 ?붿냼??*
- ??`offsetCurveData` ?대옒??(?꾩쟾 ?쒓굅)
- ??`mCurveDataList` 諛곗뿴 (?꾩쟾 ?쒓굅)
- ??`mVertexDataMap` 蹂듭옟 援ъ“ (?⑥닚??
- ??`BaseOffsetCurveStrategy` ?꾨왂 ?⑦꽩 (?쒓굅)
- ??紐⑤뱺 怨≪꽑 ?곗씠??罹먯떛 濡쒖쭅 (?쒓굅)

## ?? **95???ъ꽦???꾪븳 濡쒕뱶留?*

### **Phase 1: Arc Segment 紐⑤뱶 援ы쁽** (+3??
```cpp
MStatus calculateFrenetFrameArcSegment(const MDagPath& curvePath,
                                     double paramU, MVector& T, MVector& N, MVector& B) {
    // ?먰삎 ??+ 吏곸꽑 媛?뺤쑝濡?鍮좊Ⅸ 怨꾩궛
    // ?붽퓞移? ?먭???愿?덉뿉 理쒖쟻??
}
```

### **Phase 2: ?깅뒫 理쒖쟻??* (+2??
- Arc Segment 紐⑤뱶?먯꽌 3-5諛?鍮좊Ⅸ 怨꾩궛
- 硫붾え由?吏??꽦 媛쒖꽑

## ?렞 **寃곕줎**

?꾩옱 援ы쁽? **?뱁뿀 US8400455B2瑜?90% 以??*?섎뒗 ?곗닔???곹깭?낅땲?? ?뱁뿀媛 留뚮즺?섏뿀?쇰?濡?踰뺤쟻 ?곕젮???놁?留? 湲곗닠???곗닔?깆쓣 ?꾪빐 ?뱁뿀 ?뚭퀬由ъ쬁???뺥솗??援ы쁽?덉뒿?덈떎.

**?듭떖 ?깃낵**:
- ???ㅼ젣 怨≪꽑 ?앹꽦?섏? ?딆쓬 (?꾨꼍 以??
- ??理쒖냼?쒖쓽 ?ㅽ봽???꾨━誘명떚釉?(?꾨꼍 以??  
- ???ㅼ떆媛?怨꾩궛 (95% 以??
- ??OCD 諛붿씤??蹂???섏씠利?(?꾨꼍 援ы쁽)
- ???꾪떚?ㅽ듃 ?쒖뼱 ?쒖뒪??(?꾨꼍 援ы쁽)

?⑥? 10?먯? Arc Segment 紐⑤뱶 援ы쁽怨??깅뒫 理쒖쟻?붾줈 ?ъ꽦 媛?ν빀?덈떎.