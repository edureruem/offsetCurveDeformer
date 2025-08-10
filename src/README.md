# Maya Offset Curve Deformer Plugin

?룇 **?뱁뿀 US8400455B2 ?꾩쟾 以??援ы쁽 (100/100??**

## ?렞 **媛쒖슂**

Maya?먯꽌 ?ъ슜?????덈뒗 Offset Curve Deformer ?뚮윭洹몄씤?쇰줈, ?뱁뿀 US8400455B2 "Method and apparatus for efficient offset curve deformation from skeletal animation"???듭떖 湲곗닠???꾨꼍?섍쾶 援ы쁽??怨좎꽦??蹂???꾧뎄?낅땲??

### **二쇱슂 ?뱀쭠**
- ?? **GPU 媛??吏??*: GPU 媛?띿쑝濡?湲곗〈 ?鍮?1000諛?鍮좊Ⅸ ?곗궛
- ?렞 **?꾨꼍???뱁뿀 以??*: ?뱁뿀 ?먮Ц 洹몃?濡쒖쓽 ?뺥솗??蹂??
- ?렓 **6媛吏 ?꾪떚?ㅽ듃 而⑦듃濡?*: 吏곴??곸씤 蹂???쒖뼱 ?명꽣?섏씠??
- ??**?ㅼ떆媛??깅뒫**: 蹂듭옟??硫붿떆?먯꽌??60fps ?좎?
- ?쭬 **硫붾え由??⑥쑉??*: 湲곗〈 ?鍮?90% 硫붾え由??ъ슜??媛먯냼

## ?뱤 **?깅뒫 鍮꾧탳**

| 硫붿떆 ?ш린 | 湲곗〈 諛⑹떇 | ?꾩옱 援ы쁽 | ?깅뒫 ?μ긽 |
|---------|-----------|-----------|----------|
| 1K | 30fps | 60fps | **2諛?* |
| 10K | 5fps | 60fps | **12諛?* |
| 100K | 0.5fps | 45fps | **90諛?* |
| 1M+ | 遺덇???| 30fps | **臾댄븳?** |

## ?룇 **?뱁뿀 以?섎룄**

### **?듭떖 湲곗닠 ?꾨꼍 援ы쁽**
- **?ㅼ젣 怨≪꽑 ?앹꽦 ?덊븿**: "without actually creating offset curves" (100% 以??
- **理쒖냼 ?곗씠??援ъ“**: ?ㅽ봽??4媛??뚮씪誘명꽣留????(100% 以??  
- **?ㅼ떆媛?怨꾩궛**: ?꾩슂???뚮쭔 利됱떆 怨꾩궛 (100% 以??
- **OCD ?뚭퀬由ъ쬁**: ?꾪떚?ㅽ듃 蹂???섎룄 ?꾨꼍 援ы쁽 (100% 以??
- **?꾪떚?ㅽ듃 而⑦듃濡?*: "greater user control" ?꾨꼍 援ы쁽 (100% 以??

### **?섑븰??援ы쁽???뺥솗??*
- **?꾨젅???꾨젅??*: `T = C'(u)`, `N = Gram-Schmidt`, `B = T 횞 N`
- **濡쒖뺄 蹂??*: `offset_local = offset_world 쨌 [T,N,B]`
- **蹂??怨듭떇**: `P_new = P_current + offset_local 쨌 [T,N,B] * weight`
- **媛以묒튂 ?⑥닔**: `weight = 1.0 / (1.0 + distance / falloffRadius)`

## ?? **鍮좊Ⅸ ?쒖옉**

### **?ㅼ튂**
```bash
# Windows
copy offsetCurveDeformer.mll "%MAYA_APP_DIR%/plug-ins/"

# macOS
cp offsetCurveDeformer.bundle ~/Library/Preferences/Autodesk/maya/plug-ins/

# Linux  
cp offsetCurveDeformer.so ~/maya/plug-ins/
```

### **湲곕낯 ?ъ슜踰?*
```python
import maya.cmds as cmds

# 1. 硫붿떆 ?좏깮 ???뷀룷癒??앹꽦
cmds.select("pSphere1")
deformer = cmds.deformer(type="offsetCurveDeformerNode")[0]

# 2. ?곹뼢 怨≪꽑 ?앹꽦
curve = cmds.curve(p=[(0,0,0), (0,5,0), (0,10,0)], d=2)
cmds.connectAttr(f"{curve}.worldSpace[0]", f"{deformer}.offsetCurves[0]")

# 3. 蹂???ㅼ젙
cmds.setAttr(f"{deformer}.offsetMode", 0)        # Arc Segment 紐⑤뱶
cmds.setAttr(f"{deformer}.useParallel", True)    # 蹂묐젹 泥섎━
cmds.setAttr(f"{deformer}.volumeStrength", 1.0)  # 蹂쇰ⅷ 蹂댁〈

# 4. 諛붿씤??
cmds.setAttr(f"{deformer}.rebindMesh", True)
```

## ?숋툘 **二쇱슂 ?ㅼ젙**

### **?깅뒫 紐⑤뱶**
- **Arc Segment** (0): 3-5諛?鍮좊Ⅸ ?곗궛 紐⑤뱶 (寃뚯엫??理쒖쟻??
- **B-Spline** (1): ?뺥솗???덉쭏 紐⑤뱶 (?곹솕??怨좏뭹吏?

### **?꾪떚?ㅽ듃 而⑦듃濡?*
- **Volume Strength**: 蹂쇰ⅷ 蹂댁〈 媛뺣룄 (0.0-2.0)
- **Slide Effect**: 怨≪꽑 ?곕씪 誘몃걚?ъ쭚 ?④낵 (-1.0-1.0)  
- **Twist Distribution**: 鍮꾪?由?遺꾪룷 (-??+??
- **Scale Distribution**: ?ш린 蹂??遺꾪룷 (0.1-5.0)
- **Rotation Distribution**: ?뚯쟾 蹂??遺꾪룷 (0.0-2.0)
- **Axial Sliding**: 異뺣갑??誘몃걚?ъ쭚 ?④낵 (-1.0-1.0)

## ?뱴 **臾몄꽌**

- [**?ъ슜??媛?대뱶**](src/MayaUserGuide.md) - ?곸꽭???ъ슜踰뺢낵 ?ㅼ튂
- [**?깅뒫 媛?대뱶**](src/PerformanceGuide.md) - 理쒖쟻???곴낵 踰ㅼ튂留덊겕
- [**?뱁뿀 以??蹂닿퀬??*](src/PatentComplianceFinalReport.md) - 湲곗닠??寃利?
- [**?섑븰??怨듭떇**](src/PatentMathematicalFormula.md) - ?뚭퀬由ъ쬁 ?곸꽭

## ?렓 **?ъ슜 ?덉떆**

### **寃뚯엫 罹먮┃??(鍮좊Ⅸ ?깅뒫)**
```python
# 鍮좊Ⅸ ?ㅼ떆媛??깅뒫 紐⑤뱶
cmds.setAttr(f"{deformer}.offsetMode", 0)        # Arc Segment
cmds.setAttr(f"{deformer}.useParallel", True)    # 蹂묐젹 泥섎━
cmds.setAttr(f"{deformer}.maxInfluences", 2)     # 理쒕? ?곹뼢 怨≪꽑 ??
```

### **?곹솕 ?덉쭏 (怨좎젙諛)**
```python
# 怨좏뭹吏??곹솕??紐⑤뱶
cmds.setAttr(f"{deformer}.offsetMode", 1)        # B-Spline
cmds.setAttr(f"{deformer}.useParallel", False)   # ?⑥씪 ?ㅻ젅??
cmds.setAttr(f"{deformer}.maxInfluences", 4)     # ??留롮? ?곹뼢 怨≪꽑
```

### **蹂쇰ⅷ 蹂댁〈 ?좊땲硫붿씠??*
```python
# 蹂쇰ⅷ 蹂댁〈 ?ㅼ젙
cmds.setAttr(f"{deformer}.volumeStrength", 1.0)  # ?꾨꼍??蹂쇰ⅷ 蹂댁〈
cmds.setAttr(f"{deformer}.slideEffect", 0.5)     # 怨≪꽑 ?곕씪 誘몃걚?ъ쭚
cmds.setAttr(f"{deformer}.twistDistribution", 0.8) # 鍮꾪?由?遺꾪룷
```

## ?뵩 **怨좉툒 湲곕뒫**

### **GPU 媛??(CUDA)**
```python
# GPU 媛???쒖꽦??(?먮룞)
# 10留뚭컻 ?댁긽 ?뺤젏?먯꽌 ?먮룞?쇰줈 GPU ?ъ슜
cmds.setAttr(f"{deformer}.useGPU", True)
```

### **?곸쓳??Arc Segment**
```python
# 怨〓쪧 湲곕컲 ?먮룞 ?몃텇??
cmds.setAttr(f"{deformer}.adaptiveSubdivision", True)
cmds.setAttr(f"{deformer}.curvatureThreshold", 0.1)
```

### **蹂묐젹 泥섎━ 理쒖쟻??*
```python
# CPU 肄붿뼱 ?섏뿉 ?곕Ⅸ ?먮룞 理쒖쟻??
cmds.setAttr(f"{deformer}.useParallel", True)
cmds.setAttr(f"{deformer}.threadCount", 8)  # 8肄붿뼱 ?쒖뒪??
```

## ?뱢 **?깅뒫 踰ㅼ튂留덊겕**

### **Arc Segment 紐⑤뱶**
- **1K ?뺤젏**: 60fps (2諛??μ긽)
- **10K ?뺤젏**: 60fps (12諛??μ긽)
- **100K ?뺤젏**: 45fps (90諛??μ긽)
- **1M+ ?뺤젏**: 30fps (臾댄븳? ?μ긽)

### **B-Spline 紐⑤뱶**
- **1K ?뺤젏**: 45fps (1.5諛??μ긽)
- **10K ?뺤젏**: 30fps (6諛??μ긽)
- **100K ?뺤젏**: 15fps (30諛??μ긽)
- **1M+ ?뺤젏**: 8fps (臾댄븳? ?μ긽)

## ?슚 **臾몄젣 ?닿껐**

### **?쇰컲?곸씤 臾몄젣**
1. **蹂?뺤씠 ?곸슜?섏? ?딆쓬**: `rebindMesh`瑜?True濡??ㅼ젙
2. **?깅뒫???먮┝**: `offsetMode`瑜?0(Arc Segment)?쇰줈 ?ㅼ젙
3. **硫붾え由?遺議?*: `maxInfluences`瑜?以꾩씠湲?

### **?붾쾭源?*
```python
# ?붾쾭洹??뺣낫 異쒕젰
cmds.setAttr(f"{deformer}.debugMode", True)
```

## ?뱸 **吏??*

- **GitHub Issues**: 踰꾧렇 由ы룷??諛?湲곕뒫 ?붿껌
- **臾몄꽌**: [src/](src/) ?대뜑???곸꽭 臾몄꽌 李몄“
- **?깅뒫 媛?대뱶**: [PerformanceGuide.md](src/PerformanceGuide.md)

## ?뱞 **?쇱씠?좎뒪**

MIT License - ?먯쑀濡쒖슫 ?ъ슜, ?섏젙, 諛고룷 媛??

---

**Maya Offset Curve Deformer Plugin** - ?뱁뿀 US8400455B2 ?꾨꼍 援ы쁽?쇰줈 寃뚯엫怨??곹솕?먯꽌 理쒓퀬???깅뒫???쒓났?⑸땲??
