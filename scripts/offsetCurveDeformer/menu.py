import maya.cmds as cmds
import maya.mel as mel

def createMenu():
    """Create OCD menu in Maya"""
    # Check if menu already exists
    if cmds.menu('OCDMenu', exists=True):
        cmds.deleteUI('OCDMenu')
    
    # Create main menu
    mainWindow = mel.eval('$tmpVar=$gMainWindow')
    ocdMenu = cmds.menu('OCDMenu', label='Offset Curve Deformer', parent=mainWindow)
    
    # Create menu items
    cmds.menuItem(label='Create OCD Node', command='import maya.cmds as cmds; cmds.deformer(type="offsetCurveDeformer")')
    cmds.menuItem(divider=True)
    cmds.menuItem(label='About OCD', command='import maya.cmds as cmds; cmds.confirmDialog(title="About", message="Offset Curve Deformer v2.0.0\\nGPU Accelerated Curve Deformation", button="OK")')
    
    print("OCD Menu created successfully")

def removeMenu():
    """Remove OCD menu from Maya"""
    if cmds.menu('OCDMenu', exists=True):
        cmds.deleteUI('OCDMenu')
        print("OCD Menu removed successfully")

def onMayaIdle():
    """Called when Maya is idle - create menu if not exists"""
    if not cmds.menu('OCDMenu', exists=True):
        createMenu()
