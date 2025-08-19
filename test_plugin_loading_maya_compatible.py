"""
offsetCurveDeformer 플러그인 로드 테스트 스크립트 (Maya Python 2.7 호환)
Maya 2020에서 플러그인 로드 상태를 확인하고 기본 기능을 테스트합니다.
"""

import maya.cmds as cmds
import maya.mel as mel

def check_plugin_status():
    """플러그인 로드 상태를 확인합니다."""
    
    print("=== offsetCurveDeformer 플러그인 상태 확인 ===")
    
    # 1. 플러그인 매니저에서 상태 확인
    print("1. 플러그인 매니저 상태 확인...")
    try:
        # 플러그인 매니저 열기
        cmds.pluginManager(query=True, listPlugins=True)
        print("   플러그인 매니저 접근 가능")
    except Exception as e:
        print("   플러그인 매니저 접근 실패: {}".format(str(e)))
    
    # 2. 로드된 플러그인 목록 확인
    print("2. 로드된 플러그인 목록 확인...")
    try:
        loaded_plugins = cmds.pluginInfo(query=True, listLoaded=True)
        if loaded_plugins:
            print("   로드된 플러그인 수: {}".format(len(loaded_plugins)))
            for plugin in loaded_plugins:
                if "offsetCurve" in plugin.lower():
                    print("   발견된 관련 플러그인: {}".format(plugin))
        else:
            print("   로드된 플러그인이 없습니다.")
    except Exception as e:
        print("   플러그인 목록 확인 실패: {}".format(str(e)))
    
    # 3. 특정 플러그인 정보 확인
    print("3. offsetCurveDeformer 플러그인 정보 확인...")
    try:
        plugin_info = cmds.pluginInfo("offsetCurveDeformer.mll", query=True, version=True)
        print("   플러그인 버전: {}".format(plugin_info))
    except Exception as e:
        print("   플러그인 정보 확인 실패: {}".format(str(e)))
    
    # 4. 노드 타입 확인
    print("4. 노드 타입 확인...")
    try:
        node_types = cmds.ls(nodeTypes=True)
        offset_nodes = [node for node in node_types if "offsetCurve" in node.lower()]
        if offset_nodes:
            print("   발견된 관련 노드 타입: {}".format(offset_nodes))
        else:
            print("   관련 노드 타입을 찾을 수 없습니다.")
    except Exception as e:
        print("   노드 타입 확인 실패: {}".format(str(e)))

def load_plugin():
    """플러그인을 로드합니다."""
    
    print("\n=== 플러그인 로드 시도 ===")
    
    try:
        # 플러그인 로드
        print("1. 플러그인 로드 시도...")
        cmds.loadPlugin("offsetCurveDeformer.mll")
        print("   플러그인 로드 성공!")
        
        # 로드 후 상태 재확인
        print("2. 로드 후 상태 확인...")
        check_plugin_status()
        
        return True
        
    except Exception as e:
        print("   플러그인 로드 실패: {}".format(str(e)))
        return False

def test_basic_node_creation():
    """기본 노드 생성을 테스트합니다."""
    
    print("\n=== 기본 노드 생성 테스트 ===")
    
    try:
        # 1. 테스트 커브 생성
        print("1. 테스트 커브 생성...")
        test_curve = cmds.circle(radius=3, sections=6, name="testCurve")[0]
        print("   생성된 커브: {}".format(test_curve))
        
        # 2. 노드 타입 확인
        print("2. 사용 가능한 디포머 타입 확인...")
        deformer_types = cmds.deformer(query=True, type=True)
        offset_deformers = [d for d in deformer_types if "offsetCurve" in d.lower()]
        print("   발견된 offsetCurve 디포머: {}".format(offset_deformers))
        
        # 3. 노드 생성 시도
        if offset_deformers:
            print("3. offsetCurveDeformer 노드 생성 시도...")
            try:
                deformer_node = cmds.deformer(test_curve, type=offset_deformers[0])[0]
                print("   디포머 노드 생성 성공: {}".format(deformer_node))
                
                # 4. 노드 속성 확인
                print("4. 노드 속성 확인...")
                attributes = cmds.listAttr(deformer_node, keyable=True)
                if attributes:
                    print("   키프레임 가능한 속성들: {}".format(attributes))
                else:
                    print("   키프레임 가능한 속성이 없습니다.")
                
                return True
                
            except Exception as e:
                print("   디포머 노드 생성 실패: {}".format(str(e)))
                return False
        else:
            print("   offsetCurve 디포머 타입을 찾을 수 없습니다.")
            return False
            
    except Exception as e:
        print("   기본 테스트 실패: {}".format(str(e)))
        return False

def test_deformation_workflow():
    """변형 워크플로우를 테스트합니다."""
    
    print("\n=== 변형 워크플로우 테스트 ===")
    
    try:
        # 1. 기존 노드 찾기
        deformer_nodes = cmds.ls(type="offsetCurveDeformerPatent")
        if not deformer_nodes:
            print("offsetCurveDeformer 노드를 먼저 생성해주세요.")
            return False
            
        deformer_node = deformer_nodes[0]
        print("테스트 대상 노드: {}".format(deformer_node))
        
        # 2. 영향 커브 생성 및 연결
        print("2. 영향 커브 생성 및 연결...")
        influence_curve = cmds.circle(radius=3, sections=6, name="influenceCurve")[0]
        cmds.connectAttr("{}worldSpace[0]".format(influence_curve), "{}influenceCurve[0]".format(deformer_node))
        print("   영향 커브 연결: {} -> {}".format(influence_curve, deformer_node))
        
        # 3. 기본 속성 설정
        print("3. 기본 속성 설정...")
        cmds.setAttr("{}offsetMode".format(deformer_node), 0)  # Normal 모드
        cmds.setAttr("{}implementationType".format(deformer_node), 0)  # B-Spline
        cmds.setAttr("{}falloffRadius".format(deformer_node), 2.0)
        cmds.setAttr("{}normalOffset".format(deformer_node), 1.0)
        cmds.setAttr("{}debugDisplay".format(deformer_node), True)
        print("   기본 속성 설정 완료")
        
        # 4. 변형 테스트
        print("4. 변형 테스트...")
        cmds.move(2, 0, 0, influence_curve, relative=True)
        print("   영향 커브 이동 완료")
        
        # 5. 속성 애니메이션 테스트
        print("5. 속성 애니메이션 테스트...")
        cmds.setKeyframe("{}normalOffset".format(deformer_node), time=1, value=0.0)
        cmds.setKeyframe("{}normalOffset".format(deformer_node), time=24, value=2.0)
        print("   normalOffset 키프레임 설정 완료 (프레임 1-24)")
        
        print("   변형 워크플로우 테스트 완료")
        return True
        
    except Exception as e:
        print("   변형 워크플로우 테스트 실패: {}".format(str(e)))
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
        print("테스트 대상 노드: {}".format(deformer_node))
        
        # 1. 다양한 오프셋 모드 테스트
        print("1. 다양한 오프셋 모드 테스트...")
        modes = ["Normal", "Tangent", "BiNormal"]
        for i, mode in enumerate(modes):
            cmds.setAttr("{}offsetMode".format(deformer_node), i)
            print("   {} 모드 설정 완료".format(mode))
            
        # 2. 구현 타입 변경 테스트
        print("2. 구현 타입 변경 테스트...")
        cmds.setAttr("{}implementationType".format(deformer_node), 1)  # Arc Segment
        print("   Arc Segment 모드로 변경")
        
        # 3. 고급 제어 속성 테스트
        print("3. 고급 제어 속성 테스트...")
        cmds.setAttr("{}volumePreservation".format(deformer_node), 0.8)
        cmds.setAttr("{}selfIntersectionPrevention".format(deformer_node), False)
        cmds.setAttr("{}poseSpaceBlending".format(deformer_node), 0.5)
        print("   고급 속성 설정 완료")
        
        print("=== 고급 기능 테스트 완료 ===")
        return True
        
    except Exception as e:
        print("고급 기능 테스트 중 오류 발생: {}".format(str(e)))
        return False

def cleanup_test():
    """테스트 데이터를 정리합니다."""
    
    print("\n=== 테스트 데이터 정리 ===")
    
    try:
        # 테스트 커브 삭제
        test_objects = ["testCurve", "influenceCurve"]
        for obj in test_objects:
            if cmds.objExists(obj):
                cmds.delete(obj)
                print("   {} 삭제 완료".format(obj))
        
        # 디포머 노드들 삭제
        deformer_nodes = cmds.ls(type="offsetCurveDeformerPatent")
        for node in deformer_nodes:
            cmds.delete(node)
            print("   {} 삭제 완료".format(node))
            
        print("   정리 완료")
        
    except Exception as e:
        print("   정리 중 오류 발생: {}".format(str(e)))

# 메인 실행
if __name__ == "__main__":
    print("offsetCurveDeformer 플러그인 로드 테스트 (Maya Python 2.7 호환)")
    print("=" * 70)
    
    # 1. 현재 상태 확인
    check_plugin_status()
    
    # 2. 플러그인 로드 시도
    if load_plugin():
        # 3. 기본 노드 생성 테스트
        if test_basic_node_creation():
            # 4. 변형 워크플로우 테스트
            test_deformation_workflow()
            # 5. 고급 기능 테스트
            test_advanced_features()
    
    print("\n=== 테스트 완료 ===")
    print("정리가 필요하면 cleanup_test() 함수를 실행하세요.")
