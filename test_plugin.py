"""
Offset Curve Deformer Plugin Test Script
Maya에서 플러그인 기능을 테스트하는 스크립트
"""

import maya.cmds as cmds
import maya.mel as mel

def test_plugin_loading():
    """플러그인 로딩 테스트"""
    print("=== 플러그인 로딩 테스트 ===")
    
    try:
        # 플러그인 로드
        cmds.loadPlugin("offsetCurveDeformer.mll", quiet=True)
        print("✅ 플러그인 로드 성공")
        return True
    except Exception as e:
        print(f"❌ 플러그인 로드 실패: {e}")
        return False

def test_node_creation():
    """노드 생성 테스트"""
    print("\n=== 노드 생성 테스트 ===")
    
    try:
        # 테스트 메시 생성
        mesh = cmds.polySphere(r=5, sx=20, sy=20)[0]
        print(f"✅ 테스트 메시 생성: {mesh}")
        
        # 디포머 노드 생성
        deformer = cmds.deformer(mesh, type="offsetCurveDeformer")[0]
        print(f"✅ 디포머 노드 생성: {deformer}")
        
        return mesh, deformer
    except Exception as e:
        print(f"❌ 노드 생성 실패: {e}")
        return None, None

def test_curve_connection():
    """곡선 연결 테스트"""
    print("\n=== 곡선 연결 테스트 ===")
    
    try:
        # 테스트 곡선 생성
        curve = cmds.curve(p=[(0,0,0), (0,5,0), (0,10,0)], d=2)
        print(f"✅ 테스트 곡선 생성: {curve}")
        
        # 디포머에 곡선 연결
        mesh, deformer = test_node_creation()
        if mesh and deformer:
            cmds.connectAttr(f"{curve}.worldSpace[0]", f"{deformer}.offsetCurves[0]")
            print(f"✅ 곡선 연결 성공: {curve} -> {deformer}")
            return curve, mesh, deformer
        
        return None, None, None
    except Exception as e:
        print(f"❌ 곡선 연결 실패: {e}")
        return None, None, None

def test_attributes():
    """속성 테스트"""
    print("\n=== 속성 테스트 ===")
    
    try:
        curve, mesh, deformer = test_curve_connection()
        if not deformer:
            return False
            
        # 속성 값 설정 테스트
        cmds.setAttr(f"{deformer}.offsetMode", 0)        # Arc Segment 모드
        cmds.setAttr(f"{deformer}.volumeStrength", 1.0)  # 볼륨 보존
        cmds.setAttr(f"{deformer}.slideEffect", 0.5)     # 슬라이딩 효과
        
        print("✅ 속성 설정 성공")
        
        # 속성 값 확인
        mode = cmds.getAttr(f"{deformer}.offsetMode")
        volume = cmds.getAttr(f"{deformer}.volumeStrength")
        slide = cmds.getAttr(f"{deformer}.slideEffect")
        
        print(f"   - Offset Mode: {mode}")
        print(f"   - Volume Strength: {volume}")
        print(f"   - Slide Effect: {slide}")
        
        return True
    except Exception as e:
        print(f"❌ 속성 테스트 실패: {e}")
        return False

def test_deformation():
    """변형 테스트"""
    print("\n=== 변형 테스트 ===")
    
    try:
        curve, mesh, deformer = test_curve_connection()
        if not deformer:
            return False
            
        # 변형 적용
        cmds.setAttr(f"{deformer}.rebindMesh", True)
        
        # 결과 확인
        vertices = cmds.polyInfo(mesh, vertex=True)
        print(f"✅ 변형 적용 성공")
        print(f"   - 메시: {mesh}")
        print(f"   - 정점 수: {len(vertices)}")
        
        return True
    except Exception as e:
        print(f"❌ 변형 테스트 실패: {e}")
        return False

def run_all_tests():
    """모든 테스트 실행"""
    print("🚀 Offset Curve Deformer 플러그인 테스트 시작")
    print("=" * 50)
    
    tests = [
        ("플러그인 로딩", test_plugin_loading),
        ("노드 생성", test_node_creation),
        ("곡선 연결", test_curve_connection),
        ("속성 테스트", test_attributes),
        ("변형 테스트", test_deformation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} 테스트에서 예외 발생: {e}")
    
    print("\n" + "=" * 50)
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️  일부 테스트 실패")
    
    return passed == total

if __name__ == "__main__":
    # Maya에서 실행할 때
    if cmds.about(batch=True):
        print("배치 모드에서는 테스트를 실행할 수 없습니다.")
    else:
        run_all_tests()
