"""
offsetCurveDeformer 플러그인 로드 테스트 스크립트
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
        print(f"   플러그인 매니저 접근 실패: {str(e)}")
    
    # 2. 로드된 플러그인 목록 확인
    print("2. 로드된 플러그인 목록 확인...")
    try:
        loaded_plugins = cmds.pluginInfo(query=True, listLoaded=True)
        if loaded_plugins:
            print(f"   로드된 플러그인 수: {len(loaded_plugins)}")
            for plugin in loaded_plugins:
                if "offsetCurve" in plugin.lower():
                    print(f"   발견된 관련 플러그인: {plugin}")
        else:
            print("   로드된 플러그인이 없습니다.")
    except Exception as e:
        print(f"   플러그인 목록 확인 실패: {str(e)}")
    
    # 3. 특정 플러그인 정보 확인
    print("3. offsetCurveDeformer 플러그인 정보 확인...")
    try:
        plugin_info = cmds.pluginInfo("offsetCurveDeformer.mll", query=True, version=True)
        print(f"   플러그인 버전: {plugin_info}")
    except Exception as e:
        print(f"   플러그인 정보 확인 실패: {str(e)}")
    
    # 4. 노드 타입 확인
    print("4. 노드 타입 확인...")
    try:
        node_types = cmds.ls(nodeTypes=True)
        offset_nodes = [node for node in node_types if "offsetCurve" in node.lower()]
        if offset_nodes:
            print(f"   발견된 관련 노드 타입: {offset_nodes}")
        else:
            print("   관련 노드 타입을 찾을 수 없습니다.")
    except Exception as e:
        print(f"   노드 타입 확인 실패: {str(e)}")

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
        print(f"   플러그인 로드 실패: {str(e)}")
        return False

def test_basic_node_creation():
    """기본 노드 생성을 테스트합니다."""
    
    print("\n=== 기본 노드 생성 테스트 ===")
    
    try:
        # 1. 테스트 커브 생성
        print("1. 테스트 커브 생성...")
        test_curve = cmds.circle(radius=3, sections=6, name="testCurve")[0]
        print(f"   생성된 커브: {test_curve}")
        
        # 2. 노드 타입 확인
        print("2. 사용 가능한 디포머 타입 확인...")
        deformer_types = cmds.deformer(query=True, type=True)
        offset_deformers = [d for d in deformer_types if "offsetCurve" in d.lower()]
        print(f"   발견된 offsetCurve 디포머: {offset_deformers}")
        
        # 3. 노드 생성 시도
        if offset_deformers:
            print("3. offsetCurveDeformer 노드 생성 시도...")
            try:
                deformer_node = cmds.deformer(test_curve, type=offset_deformers[0])[0]
                print(f"   디포머 노드 생성 성공: {deformer_node}")
                
                # 4. 노드 속성 확인
                print("4. 노드 속성 확인...")
                attributes = cmds.listAttr(deformer_node, keyable=True)
                if attributes:
                    print(f"   키프레임 가능한 속성들: {attributes}")
                else:
                    print("   키프레임 가능한 속성이 없습니다.")
                
                return True
                
            except Exception as e:
                print(f"   디포머 노드 생성 실패: {str(e)}")
                return False
        else:
            print("   offsetCurve 디포머 타입을 찾을 수 없습니다.")
            return False
            
    except Exception as e:
        print(f"   기본 테스트 실패: {str(e)}")
        return False

def cleanup_test():
    """테스트 데이터를 정리합니다."""
    
    print("\n=== 테스트 데이터 정리 ===")
    
    try:
        # 테스트 커브 삭제
        if cmds.objExists("testCurve"):
            cmds.delete("testCurve")
            print("   testCurve 삭제 완료")
        
        # 디포머 노드들 삭제
        deformer_nodes = cmds.ls(type="offsetCurveDeformerPatent")
        for node in deformer_nodes:
            cmds.delete(node)
            print(f"   {node} 삭제 완료")
            
        print("   정리 완료")
        
    except Exception as e:
        print(f"   정리 중 오류 발생: {str(e)}")

# 메인 실행
if __name__ == "__main__":
    print("offsetCurveDeformer 플러그인 로드 테스트")
    print("=" * 60)
    
    # 1. 현재 상태 확인
    check_plugin_status()
    
    # 2. 플러그인 로드 시도
    if load_plugin():
        # 3. 기본 노드 생성 테스트
        test_basic_node_creation()
    
    print("\n=== 테스트 완료 ===")
    print("정리가 필요하면 cleanup_test() 함수를 실행하세요.")
