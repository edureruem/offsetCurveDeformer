#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maya Offset Curve Deformer Plugin Test Script
Maya에서 offsetCurveDeformer 플러그인을 테스트하는 스크립트
"""

import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMaya as om

def test_plugin_loading():
    """플러그인 로딩 테스트"""
    print("=== 플러그인 로딩 테스트 ===")
    
    try:
        # 플러그인 로드
        result = cmds.loadPlugin("offsetCurveDeformer.mll", quiet=True)
        if result:
            print(f"플러그인 로딩 성공: {result}")
            return True
        else:
            print("플러그인 로딩 실패")
            return False
    except Exception as e:
        print(f"플러그인 로딩 오류: {e}")
        return False

def test_node_creation():
    """노드 생성 테스트"""
    print("\n=== 노드 생성 테스트 ===")
    
    try:
        # offsetCurveDeformer 노드 생성
        node_name = cmds.createNode("offsetCurveDeformer")
        if node_name:
            print(f"노드 생성 성공: {node_name}")
            
            # 노드 타입 확인
            node_type = cmds.nodeType(node_name)
            print(f"노드 타입: {node_type}")
            
            # 노드 삭제
            cmds.delete(node_name)
            print("노드 삭제 성공")
            return True
        else:
            print("노드 생성 실패")
            return False
    except Exception as e:
        print(f"노드 생성 오류: {e}")
        return False

def test_attributes():
    """속성 테스트"""
    print("\n=== 속성 테스트 ===")
    
    try:
        # 노드 생성
        node_name = cmds.createNode("offsetCurveDeformer")
        
        # 기본 속성들 확인
        attributes = [
            "falloffRadius",
            "deformationStrength", 
            "volumePreservation",
            "slide",
            "rotation",
            "scale",
            "twist",
            "axialSliding",
            "poseBlending"
        ]
        
        print("기본 속성 확인:")
        for attr in attributes:
            full_attr = f"{node_name}.{attr}"
            if cmds.objExists(full_attr):
                value = cmds.getAttr(full_attr)
                print(f"  - {attr}: {value}")
            else:
                print(f"  - {attr}: 존재하지 않음")
        
        # 노드 삭제
        cmds.delete(node_name)
        return True
        
    except Exception as e:
        print(f"속성 테스트 오류: {e}")
        return False

def test_deformation():
    """변형 테스트"""
    print("\n=== 변형 테스트 ===")
    
    try:
        # 테스트 메시 생성
        mesh_name = cmds.polyCube(name="testMesh", width=2, height=2, depth=2)[0]
        
        # 테스트 곡선 생성
        curve_name = cmds.curve(name="testCurve", 
                              p=[(0,0,0), (1,1,0), (2,0,0), (3,1,0)], 
                              degree=3)
        
        # offsetCurveDeformer 노드 생성
        deformer_name = cmds.createNode("offsetCurveDeformer")
        
        # 연결
        cmds.connectAttr(f"{mesh_name}.outMesh", f"{deformer_name}.inputMesh")
        cmds.connectAttr(f"{curve_name}.worldSpace[0]", f"{deformer_name}.inputCurve")
        
        print("변형 설정 완료:")
        print(f"  - 메시: {mesh_name}")
        print(f"  - 곡선: {curve_name}")
        print(f"  - 디포머: {deformer_name}")
        
        # 정리
        cmds.delete(mesh_name, curve_name, deformer_name)
        return True
        
    except Exception as e:
        print(f"변형 테스트 오류: {e}")
        return False

def test_performance():
    """성능 테스트"""
    print("\n=== 성능 테스트 ===")
    
    try:
        import time
        
        # 대용량 메시 생성 (100x100x100 = 1,000,000 버텍스)
        start_time = time.time()
        mesh_name = cmds.polyPlane(name="performanceMesh", 
                                 width=10, height=10, 
                                 subdivisionsWidth=100, subdivisionsHeight=100)[0]
        creation_time = time.time() - start_time
        
        print(f"대용량 메시 생성: {creation_time:.3f}초")
        
        # 디포머 적용
        start_time = time.time()
        deformer_name = cmds.deformer(mesh_name, type="offsetCurveDeformer")[0]
        deformation_time = time.time() - start_time
        
        print(f"디포머 적용: {deformation_time:.3f}초")
        
        # 정리
        cmds.delete(mesh_name)
        
        return True
        
    except Exception as e:
        print(f"성능 테스트 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("Maya Offset Curve Deformer Plugin 테스트 시작")
    print("=" * 50)
    
    tests = [
        ("플러그인 로딩", test_plugin_loading),
        ("노드 생성", test_node_creation),
        ("속성 테스트", test_attributes),
        ("변형 테스트", test_deformation),
        ("성능 테스트", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"{test_name} 테스트에서 예외 발생: {e}")
            print()
    
    print("=" * 50)
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("모든 테스트 통과! 플러그인이 정상적으로 작동합니다.")
    else:
        print("일부 테스트 실패. 플러그인에 문제가 있을 수 있습니다.")
    
    return passed == total

if __name__ == "__main__":
    main()
