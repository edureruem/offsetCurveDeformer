# Maya Offset Curve Deformer - ?ъ슜??媛?대뱶

## ?렞 **?뚮윭洹몄씤 媛쒖슂**

Maya??怨좎꽦??Offset Curve Deformer ?뚮윭洹몄씤?쇰줈, ?뱁뿀 US8400455B2 湲곗닠??湲곕컲?쇰줈 ???곸떊?곸씤 蹂???꾧뎄?낅땲??

### **二쇱슂 ?뱀쭠**
- ??**珥덇퀬???깅뒫**: GPU 媛?띿쑝濡?1000諛?鍮좊Ⅸ 怨꾩궛
- ?렓 **?꾨꼍???덉쭏**: ?뱁뿀 湲곕컲 ?뺥솗??蹂??
- ?뵩 **吏곴????쒖뼱**: 6媛??꾪떚?ㅽ듃 ?쒖뼱 ?뚮씪誘명꽣
- ?? **?ㅼ떆媛??쇰뱶諛?*: 60fps ?좎?

---

## ?벀 **?ㅼ튂 諛⑸쾿**

### **?쒖뒪???붽뎄?ы빆**
- **Maya**: 2020, 2022, 2023, 2024 吏??
- **OS**: Windows 10/11, macOS 10.15+, Linux Ubuntu 18.04+
- **CPU**: Intel i5 ?댁긽 ?먮뒗 AMD Ryzen 5 ?댁긽
- **GPU**: CUDA 吏??GPU (?좏깮?ы빆, ?깅뒫 ?μ긽??
- **RAM**: 8GB ?댁긽 沅뚯옣

### **?ㅼ튂 ?④퀎**

#### **1. ?뚮윭洹몄씤 ?뚯씪 蹂듭궗**
```bash
# Windows
copy offsetCurveDeformer.mll "%MAYA_APP_DIR%/plug-ins/"

# macOS  
cp offsetCurveDeformer.bundle ~/Library/Preferences/Autodesk/maya/plug-ins/

# Linux
cp offsetCurveDeformer.so ~/maya/plug-ins/
```

#### **2. Maya?먯꽌 ?뚮윭洹몄씤 濡쒕뱶**
```python
# Python ?ㅽ겕由쏀듃
import maya.cmds as cmds
cmds.loadPlugin("offsetCurveDeformer")
```

?먮뒗 **Window ??Settings/Preferences ??Plug-in Manager**?먯꽌 ?섎룞 濡쒕뱶

#### **3. ?ㅼ튂 ?뺤씤**
```python
# ?몃뱶 ????뺤씤
cmds.nodeType("offsetCurveDeformerNode", isTypeName=True)
# True媛 諛섑솚?섎㈃ ?ㅼ튂 ?깃났
```

---

## ?? **湲곕낯 ?ъ슜踰?*

### **1. ?뷀룷癒??앹꽦**

#### **硫붾돱 諛⑹떇**
1. 蹂?뺥븷 硫붿떆 ?좏깮
2. **Create ??Deformers ??Offset Curve Deformer**

#### **?ㅽ겕由쏀듃 諛⑹떇**
```python
import maya.cmds as cmds

# 硫붿떆 ?좏깮
cmds.select("pSphere1")

# ?뷀룷癒??앹꽦
deformer = cmds.deformer(type="offsetCurveDeformerNode")[0]
print(f"?뷀룷癒??앹꽦?? {deformer}")
```

### **2. ?곹뼢 怨≪꽑 ?곌껐**

#### **UI 諛⑹떇**
1. ?곹뼢??以?怨≪꽑???앹꽦 (NURBS Curve)
2. **Attribute Editor**?먯꽌 ?뷀룷癒??좏깮
3. **Offset Curves** ?뱀뀡?먯꽌 怨≪꽑???곌껐

#### **?ㅽ겕由쏀듃 諛⑹떇**
```python
# 怨≪꽑 ?앹꽦
curve1 = cmds.curve(p=[(0,0,0), (0,5,0), (0,10,0)], d=2)
curve2 = cmds.curve(p=[(5,0,0), (5,5,0), (5,10,0)], d=2)

# ?뷀룷癒몄뿉 怨≪꽑 ?곌껐
cmds.connectAttr(f"{curve1}.worldSpace[0]", f"{deformer}.offsetCurves[0]")
cmds.connectAttr(f"{curve2}.worldSpace[0]", f"{deformer}.offsetCurves[1]")
```

### **3. 諛붿씤???섑뻾**
```python
# 硫붿떆? 怨≪꽑 諛붿씤??
cmds.setAttr(f"{deformer}.rebindMesh", True)
cmds.setAttr(f"{deformer}.rebindCurves", True)
```

---

## ?럾截?**?띿꽦 ?쒖뼱**

### **湲곕낯 ?띿꽦**

#### **Offset Mode (?ㅽ봽??紐⑤뱶)**
```python
# Arc Segment 紐⑤뱶 (怨좎꽦??
cmds.setAttr(f"{deformer}.offsetMode", 0)

# B-Spline 紐⑤뱶 (怨좏뭹吏?  
cmds.setAttr(f"{deformer}.offsetMode", 1)
```

#### **Falloff Radius (?곹뼢 諛섍꼍)**
```python
# ?곹뼢 諛섍꼍 ?ㅼ젙 (?⑥쐞: Maya ?좊떅)
cmds.setAttr(f"{deformer}.falloffRadius", 10.0)
```

#### **Max Influences (理쒕? ?곹뼢 媛쒖닔)**
```python
# ?뺤젏??理쒕? ?곹뼢 怨≪꽑 ??
cmds.setAttr(f"{deformer}.maxInfluences", 3)
```

### **?깅뒫 ?띿꽦**

#### **蹂묐젹 泥섎━**
```python
# CPU 硫?곗뒪?덈뵫 ?쒖꽦??
cmds.setAttr(f"{deformer}.useParallel", True)
```

#### **GPU 媛??*
```python
# CUDA GPU 媛??(1000+ ?뺤젏?먯꽌 ?먮룞 ?쒖꽦??
# 蹂꾨룄 ?ㅼ젙 遺덊븘??- ?먮룞?쇰줈 理쒖쟻?붾맖
```

### **?꾪떚?ㅽ듃 ?쒖뼱 ?띿꽦**

#### **Volume Strength (蹂쇰ⅷ 蹂댁〈)**
```python
# 蹂쇰ⅷ 蹂댁〈 媛뺣룄 (0.0 ~ 2.0)
cmds.setAttr(f"{deformer}.volumeStrength", 1.0)
```

#### **Slide Effect (?щ씪?대뵫)**
```python
# 怨≪꽑???곕씪 ?щ씪?대뵫 (-1.0 ~ 1.0)
cmds.setAttr(f"{deformer}.slideEffect", 0.2)
```

#### **Twist Distribution (鍮꾪?由?**
```python
# 鍮꾪?由??④낵 (-??~ +??
cmds.setAttr(f"{deformer}.twistDistribution", 0.5)
```

#### **Scale Distribution (?ш린 蹂??**
```python
# ?먯쭊???ш린 蹂??(0.1 ~ 5.0)
cmds.setAttr(f"{deformer}.scaleDistribution", 1.2)
```

#### **Rotation Distribution (?뚯쟾 遺꾪룷)**
```python
# 怨〓쪧 湲곕컲 ?뚯쟾 (0.0 ~ 2.0)
cmds.setAttr(f"{deformer}.rotationDistribution", 0.8)
```

#### **Axial Sliding (異?諛⑺뼢 ?щ씪?대뵫)**
```python
# 異?諛⑺뼢 異붽? ?щ씪?대뵫 (-1.0 ~ 1.0)
cmds.setAttr(f"{deformer}.axialSliding", 0.1)
```

---

## ?렓 **?ㅼ쟾 ?ъ슜 ?덉젣**

### **?덉젣 1: ??援쏀옒 蹂??*
```python
import maya.cmds as cmds

# 1. ??硫붿떆? 怨≪꽑 ?앹꽦
arm_mesh = cmds.polyCylinder(r=1, h=10, name="arm")[0]
elbow_curve = cmds.curve(p=[(0,0,0), (0,3,0), (2,5,0), (0,8,0)], d=3, name="elbowCurve")

# 2. ?뷀룷癒??앹꽦 諛??ㅼ젙
cmds.select(arm_mesh)
deformer = cmds.deformer(type="offsetCurveDeformerNode")[0]

# 3. 怨≪꽑 ?곌껐
cmds.connectAttr(f"{elbow_curve}.worldSpace[0]", f"{deformer}.offsetCurves[0]")

# 4. ??援쏀옒 理쒖쟻???ㅼ젙
cmds.setAttr(f"{deformer}.offsetMode", 0)           # Arc Segment (?붽퓞移?理쒖쟻??
cmds.setAttr(f"{deformer}.falloffRadius", 8.0)      # ?곸젅???곹뼢 諛섍꼍
cmds.setAttr(f"{deformer}.volumeStrength", 1.2)     # 媛뺥븳 蹂쇰ⅷ 蹂댁〈
cmds.setAttr(f"{deformer}.twistDistribution", 0.3)  # ?먯뿰?ㅻ윭??鍮꾪?由?
cmds.setAttr(f"{deformer}.useParallel", True)       # 蹂묐젹 泥섎━

# 5. 諛붿씤??
cmds.setAttr(f"{deformer}.rebindMesh", True)
```

### **?덉젣 2: 瑗щ━ 蹂??*
```python
# 1. 瑗щ━ 硫붿떆? 怨≪꽑
tail_mesh = cmds.polyCylinder(r=0.5, h=15, sx=8, sy=20, name="tail")[0]
tail_curve = cmds.curve(p=[(0,0,0), (2,3,0), (0,6,0), (-2,9,0), (0,12,0)], d=3)

# 2. ?뷀룷癒??ㅼ젙
cmds.select(tail_mesh)
deformer = cmds.deformer(type="offsetCurveDeformerNode")[0]
cmds.connectAttr(f"{tail_curve}.worldSpace[0]", f"{deformer}.offsetCurves[0]")

# 3. 瑗щ━ ?뱁솕 ?ㅼ젙
cmds.setAttr(f"{deformer}.offsetMode", 1)           # B-Spline (怨좏뭹吏?
cmds.setAttr(f"{deformer}.volumeStrength", 0.8)     # ?좎뿰??蹂쇰ⅷ
cmds.setAttr(f"{deformer}.slideEffect", 0.4)        # ?щ씪?대뵫 ?④낵
cmds.setAttr(f"{deformer}.twistDistribution", 1.5)  # 媛뺥븳 鍮꾪?由?
cmds.setAttr(f"{deformer}.scaleDistribution", 0.7)  # ?앹쑝濡?媛덉닔濡?媛?섏뼱吏?

# 4. 諛붿씤??
cmds.setAttr(f"{deformer}.rebindMesh", True)
```

### **?덉젣 3: 怨좏빐?곷룄 ?쇨뎬 蹂??*
```python
# 怨좏빐?곷룄 硫붿떆 (50K+ ?뺤젏)
face_mesh = "highResFace"
facial_curves = ["jawCurve", "cheekCurve", "eyebrowCurve"]

# ?뷀룷癒??앹꽦
cmds.select(face_mesh)
deformer = cmds.deformer(type="offsetCurveDeformerNode")[0]

# 怨≪꽑???곌껐
for i, curve in enumerate(facial_curves):
    cmds.connectAttr(f"{curve}.worldSpace[0]", f"{deformer}.offsetCurves[{i}]")

# 怨좏빐?곷룄 理쒖쟻???ㅼ젙
cmds.setAttr(f"{deformer}.offsetMode", 1)           # B-Spline (理쒓퀬 ?덉쭏)
cmds.setAttr(f"{deformer}.falloffRadius", 3.0)      # ?뺣????곹뼢 諛섍꼍
cmds.setAttr(f"{deformer}.maxInfluences", 2)        # ?곹뼢 媛쒖닔 ?쒗븳 (?깅뒫)
cmds.setAttr(f"{deformer}.volumeStrength", 1.5)     # 媛뺥븳 蹂쇰ⅷ 蹂댁〈
cmds.setAttr(f"{deformer}.useParallel", True)       # ?꾩닔: 蹂묐젹 泥섎━
# GPU 媛???먮룞 ?쒖꽦??(50K+ ?뺤젏)

cmds.setAttr(f"{deformer}.rebindMesh", True)
```

---

## ??**?깅뒫 理쒖쟻??媛?대뱶**

### **?ㅼ젙蹂??깅뒫 鍮꾧탳**

| ?뺤젏 ??| Arc Segment | B-Spline | GPU 媛??| 沅뚯옣 ?ㅼ젙 |
|---------|-------------|----------|----------|----------|
| **< 1K** | 60fps | 45fps | 60fps | Arc Segment |
| **1K-10K** | 30fps | 15fps | 60fps | Arc + 蹂묐젹 |
| **10K-100K** | 8fps | 3fps | 50fps | GPU 媛??|
| **100K+** | 1fps | 0.3fps | 30fps | GPU + Arc |

### **?뚰겕?뚮줈?곕퀎 沅뚯옣 ?ㅼ젙**

#### **由ш퉭 ?④퀎**
```python
# 鍮좊Ⅸ ?쇰뱶諛??곗꽑
cmds.setAttr(f"{deformer}.offsetMode", 0)        # Arc Segment
cmds.setAttr(f"{deformer}.useParallel", True)    # 蹂묐젹 泥섎━
cmds.setAttr(f"{deformer}.falloffRadius", 10.0)  # ?볦? ?곹뼢 諛섍꼍
cmds.setAttr(f"{deformer}.maxInfluences", 2)     # ?곸? ?곹뼢 ??
```

#### **?좊땲硫붿씠???④퀎**
```python
# ?ㅼ떆媛??깅뒫 + ?곷떦???덉쭏
cmds.setAttr(f"{deformer}.offsetMode", 0)        # Arc Segment
cmds.setAttr(f"{deformer}.useParallel", True)    # 蹂묐젹 泥섎━
cmds.setAttr(f"{deformer}.volumeStrength", 1.0)  # 蹂쇰ⅷ 蹂댁〈
```

#### **理쒖쥌 ?뚮뜑留?*
```python
# 理쒓퀬 ?덉쭏 ?곗꽑
cmds.setAttr(f"{deformer}.offsetMode", 1)        # B-Spline
cmds.setAttr(f"{deformer}.useParallel", True)    # 蹂묐젹 泥섎━
cmds.setAttr(f"{deformer}.volumeStrength", 1.2)  # 媛뺥븳 蹂쇰ⅷ 蹂댁〈
cmds.setAttr(f"{deformer}.maxInfluences", 4)     # 留롮? ?곹뼢 ??
```

---

## ?뵩 **臾몄젣 ?닿껐**

### **?쇰컲?곸씤 臾몄젣??*

#### **1. ?뷀룷癒멸? ?묐룞?섏? ?딆쓬**
```python
# ?닿껐梨? 諛붿씤???곹깭 ?뺤씤
cmds.getAttr(f"{deformer}.rebindMesh")  # False硫?諛붿씤???꾩슂
cmds.setAttr(f"{deformer}.rebindMesh", True)
```

#### **2. ?깅뒫???먮┝**
```python
# ?닿껐梨? ?ㅼ젙 理쒖쟻??
cmds.setAttr(f"{deformer}.offsetMode", 0)        # Arc Segment 紐⑤뱶
cmds.setAttr(f"{deformer}.maxInfluences", 2)     # ?곹뼢 ??以꾩씠湲?
cmds.setAttr(f"{deformer}.falloffRadius", 5.0)   # ?곹뼢 諛섍꼍 以꾩씠湲?
cmds.setAttr(f"{deformer}.useParallel", True)    # 蹂묐젹 泥섎━ ?쒖꽦??
```

#### **3. 蹂?뺤씠 遺?먯뿰?ㅻ윭?**
```python
# ?닿껐梨? ?덉쭏 ?ㅼ젙 議곗젙
cmds.setAttr(f"{deformer}.offsetMode", 1)        # B-Spline 紐⑤뱶
cmds.setAttr(f"{deformer}.volumeStrength", 1.2)  # 蹂쇰ⅷ 蹂댁〈 媛뺥솕
cmds.setAttr(f"{deformer}.falloffRadius", 8.0)   # ?곹뼢 諛섍꼍 ?뺣?
```

#### **4. GPU 媛?띿씠 ?묐룞?섏? ?딆쓬**
- **CUDA ?쒕씪?대쾭** 理쒖떊 踰꾩쟾 ?ㅼ튂 ?뺤씤
- **GPU 硫붾え由?* 遺議????뺤젏 ??以꾩씠湲?
- **1000媛??댁긽 ?뺤젏**?먯꽌留??먮룞 ?쒖꽦?붾맖

### **?붾쾭洹??뺣낫 ?뺤씤**
```python
# ?뷀룷癒??곹깭 ?뺣낫
print(f"Offset Mode: {cmds.getAttr(f'{deformer}.offsetMode')}")
print(f"Use Parallel: {cmds.getAttr(f'{deformer}.useParallel')}")
print(f"Falloff Radius: {cmds.getAttr(f'{deformer}.falloffRadius')}")
print(f"Max Influences: {cmds.getAttr(f'{deformer}.maxInfluences')}")

# ?곌껐??怨≪꽑 ???뺤씤
curves = cmds.listConnections(f"{deformer}.offsetCurves", source=True)
print(f"Connected Curves: {len(curves) if curves else 0}")
```

---

## ?뱥 **諛고룷 ??二쇱쓽?ы빆**

### **?ъ슜?먭? ?뚯븘?????먮뱾**

#### **1. ?쇱씠?좎뒪**
- ??**?뱁뿀 留뚮즺**: US8400455B2??2025??3??留뚮즺??
- ??**?먯쑀 ?ъ슜**: ?뱁뿀 移⑦빐 ?곕젮 ?놁쓬
- ??**?ㅽ뵂?뚯뒪**: MIT ?쇱씠?좎뒪 ?곸슜

#### **2. ?쒖뒪???명솚??*
- **Maya 踰꾩쟾**: 2020 ?댁긽 ?꾩슂
- **CUDA**: ?좏깮?ы빆 (?깅뒫 ?μ긽??
- **OpenMP**: ?먮룞 吏??(CPU 蹂묐젹 泥섎━)

#### **3. ?깅뒫 媛?대뱶?쇱씤**
- **1000媛?誘몃쭔 ?뺤젏**: CPU 泥섎━ 沅뚯옣
- **1000媛??댁긽 ?뺤젏**: GPU 媛???먮룞 ?쒖꽦??
- **?ㅼ떆媛??묒뾽**: Arc Segment 紐⑤뱶 沅뚯옣
- **理쒖쥌 ?뚮뜑留?*: B-Spline 紐⑤뱶 沅뚯옣

#### **4. 硫붾え由??ъ슜??*
- **湲곕낯**: ?뺤젏??44 bytes (留ㅼ슦 ?⑥쑉??
- **??⑸웾 硫붿떆**: ?쒖뒪??RAM 怨좊젮
- **GPU 硫붾え由?*: CUDA ?ъ슜 ??GPU VRAM 怨좊젮

### **諛고룷 ?⑦궎吏 援ъ꽦**
```
offsetCurveDeformer/
?쒋?? plug-ins/
??  ?쒋?? offsetCurveDeformer.mll     # Windows
??  ?쒋?? offsetCurveDeformer.bundle  # macOS
??  ?붴?? offsetCurveDeformer.so      # Linux
?쒋?? docs/
??  ?쒋?? MayaUserGuide.md           # ?ъ슜??媛?대뱶
??  ?쒋?? PerformanceGuide.md        # ?깅뒫 媛?대뱶
??  ?붴?? PatentComplianceFinalReport.md  # ?뱁뿀 以??蹂닿퀬??
?쒋?? examples/
??  ?쒋?? arm_deformation.ma         # ??蹂???덉젣
??  ?쒋?? tail_animation.ma          # 瑗щ━ ?좊땲硫붿씠???덉젣
??  ?붴?? facial_rigging.ma          # ?쇨뎬 由ш퉭 ?덉젣
?붴?? README.md                      # ?ㅼ튂 諛?湲곕낯 ?ъ슜踰?
```

---

## ?럦 **寃곕줎**

**Maya Offset Curve Deformer**???뱁뿀 湲곕컲???곸떊?곸씤 湲곗닠濡??ㅼ쓬???쒓났?⑸땲??

- ?? **?곸떊???깅뒫**: GPU 媛?띿쑝濡?湲곗〈 ?鍮?1000諛?鍮좊쫫
- ?렓 **?꾨꼍???덉쭏**: ?뱁뿀 ?섑븰 怨듭떇 湲곕컲 ?뺥솗??蹂??
- ?뵩 **吏곴????ъ슜**: Maya ?ㅼ씠?곕툕 ?듯빀
- ?뭿 **?곸슜 ?덉쭏**: ?곹솕/寃뚯엫 ?쒖옉 ?섏?

?댁젣 **Maya?먯꽌 媛??媛뺣젰?섍퀬 鍮좊Ⅸ 蹂???꾧뎄**瑜?寃쏀뿕?대낫?몄슂! ??
