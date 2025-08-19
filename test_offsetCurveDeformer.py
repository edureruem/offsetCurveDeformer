"""
offsetCurveDeformer 노드 테스트 스크립트
Maya 2020에서 실행하여 노드 기능을 테스트합니다.
"""

import maya.cmds as cmds
import maya.mel as mel

def test_offsetCurveDeformer():
    """offsetCurveDeformer 노드의 기본 기능을 테스트합니다."""
    
    print("=== offsetCurveDeformer 노드 테스트 시작 ===")
    
    try:
        # 1. 테스트용 커브 생성
        print("1. 테스트용 NURBS 커브 생성...")
        test_curve = cmds.circle(radius=5, sections=8, name="testCurve")[0]
        print(f"   생성된 커브: {test_curve}")
        
        # 2. offsetCurveDeformer 노드 생성
        print("2. offsetCurveDeformer 노드 생성...")
        deformer_node = cmds.deformer(test_curve, type="offsetCurveDeformerPatent")[0]
        print(f"   생성된 디포머: {deformer_node}")
        
        # 3. 노드 속성 확인
        print("3. 노드 속성 확인...")
        attributes = cmds.listAttr(deformer_node, keyable=True)
        print(f"   키프레임 가능한 속성들: {attributes}")
        
        # 4. 기본 속성 값 설정
        print("4. 기본 속성 값 설정...")
        cmds.setAttr(f"{deformer_node}.offsetMode", 0)  # Normal 모드
        cmds.setAttr(f"{deformer_node}.implementationType", 0)  # B-Spline
        cmds.setAttr(f"{deformer_node}.falloffRadius", 2.0)
        cmds.setAttr(f"{deformer_node}.normalOffset", 1.0)
        cmds.setAttr(f"{deformer_node}.debugDisplay", True)
        
        print("   기본 속성 설정 완료")
        
        # 5. 영향 커브 생성 및 연결
        print("5. 영향 커브 생성 및 연결...")
        influence_curve = cmds.circle(radius=3, sections=6, name="influenceCurve")[0]
        cmds.connectAttr(f"{influence_curve}.worldSpace[0]", f"{deformer_node}.influenceCurve[0]")
        print(f"   영향 커브 연결: {influence_curve} -> {deformer_node}")
        
        # 6. 변형 테스트
        print("6. 변형 테스트...")
        # 영향 커브를 이동시켜 변형 효과 확인
        cmds.move(2, 0, 0, influence_curve, relative=True)
        print("   영향 커브 이동 완료")
        
        # 7. 속성 애니메이션 테스트
        print("7. 속성 애니메이션 테스트...")
        cmds.setKeyframe(f"{deformer_node}.normalOffset", time=1, value=0.0)
        cmds.setKeyframe(f"{deformer_node}.normalOffset", time=24, value=2.0)
        print("   normalOffset 키프레임 설정 완료 (프레임 1-24)")
        
        # 8. 결과 확인
        print("8. 결과 확인...")
        print(f"   테스트 커브: {test_curve}")
        print(f"   디포머 노드: {deformer_node}")
        print(f"   영향 커브: {influence_curve}")
        
        print("\n=== 테스트 완료 ===")
        print("Maya 뷰포트에서 변형 효과를 확인하세요.")
        print("타임라인을 1-24 프레임으로 이동하여 애니메이션을 확인하세요.")
        
        return True
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        return False

def test_advanced_features():
    """고급 기능을 테스트합니다."""
    
    print("\n=== 고급 기능 테스트 시작 ===")
    
    try:
        # 기존 노드 찾기
        deformer_nodes = cmds.ls(type="offsetCurveDeformerPatent")
        if not deformer_nodes:
            print("offsetCurveDeformer 노드를 먼저 생성해주세요.")
            return False
            
        deformer_node = deformer_nodes[0]
        print(f"테스트 대상 노드: {deformer_node}")
        
        # 1. 다양한 오프셋 모드 테스트
        print("1. 다양한 오프셋 모드 테스트...")
        modes = ["Normal", "Tangent", "BiNormal"]
        for i, mode in enumerate(modes):
            cmds.setAttr(f"{deformer_node}.offsetMode", i)
            print(f"   {mode} 모드 설정 완료")
            
        # 2. 구현 타입 변경 테스트
        print("2. 구현 타입 변경 테스트...")
        cmds.setAttr(f"{deformer_node}.implementationType", 1)  # Arc Segment
        print("   Arc Segment 모드로 변경")
        
        # 3. 고급 제어 속성 테스트
        print("3. 고급 제어 속성 테스트...")
        cmds.setAttr(f"{deformer_node}.volumePreservation", 0.8)
        cmds.setAttr(f"{deformer_node}.selfIntersectionPrevention", False)
        cmds.setAttr(f"{deformer_node}.poseSpaceBlending", 0.5)
        print("   고급 속성 설정 완료")
        
        print("\n=== 고급 기능 테스트 완료 ===")
        return True
        
    except Exception as e:
        print(f"고급 기능 테스트 중 오류 발생: {str(e)}")
        return False

def cleanup_test():
    """테스트 데이터를 정리합니다."""
    
    print("\n=== 테스트 데이터 정리 ===")
    
    try:
        # 테스트로 생성된 오브젝트들 삭제
        test_objects = ["testCurve", "influenceCurve"]
        for obj in test_objects:
            if cmds.objExists(obj):
                cmds.delete(obj)
                print(f"   {obj} 삭제 완료")
        
        # 디포머 노드들 삭제
        deformer_nodes = cmds.ls(type="offsetCurveDeformerPatent")
        for node in deformer_nodes:
            cmds.delete(node)
            print(f"   {node} 삭제 완료")
            
        print("   정리 완료")
        
    except Exception as e:
        print(f"정리 중 오류 발생: {str(e)}")

# 메인 실행
if __name__ == "__main__":
    print("offsetCurveDeformer 노드 테스트 스크립트")
    print("=" * 50)
    
    # 기본 테스트 실행
    if test_offsetCurveDeformer():
        # 고급 기능 테스트 실행
        test_advanced_features()
    
    print("\n테스트 완료! Maya 뷰포트에서 결과를 확인하세요.")
    print("정리가 필요하면 cleanup_test() 함수를 실행하세요.")
