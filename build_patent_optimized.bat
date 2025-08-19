@echo off
echo ========================================
echo US8400455 Patent - OCD Build Script
echo Sony's Revolutionary Skinning Technique
echo ========================================
echo.

REM Check if Maya is installed
set MAYA_PATH=""
if exist "C:\Program Files\Autodesk\Maya2024" (
    set MAYA_PATH="C:\Program Files\Autodesk\Maya2024"
) else if exist "C:\Program Files\Autodesk\Maya2023" (
    set MAYA_PATH="C:\Program Files\Autodesk\Maya2023"
) else if exist "C:\Program Files\Autodesk\Maya2022" (
    set MAYA_PATH="C:\Program Files\Autodesk\Maya2022"
) else if exist "C:\Program Files\Autodesk\Maya2020" (
    set MAYA_PATH="C:\Program Files\Autodesk\Maya2020"
) else (
    echo ERROR: Maya installation not found!
    echo Please install Maya 2020 or later.
    pause
    exit /b 1
)

echo Found Maya at: %MAYA_PATH%
echo.

REM Set environment variables for patent-optimized build
set MAYA_INCLUDE=%MAYA_PATH%\include
set MAYA_LIB=%MAYA_PATH%\lib
set MAYA_BIN=%MAYA_PATH%\bin

REM Set compiler flags for patent technology optimization
set CXX_FLAGS=/std:c++17 /O2 /DNDEBUG /DWIN32 /D_WINDOWS /D_USRDLL /DMAYA_PLUGIN /DUS8400455_PATENT
set CXX_FLAGS=%CXX_FLAGS% /D_CRT_SECURE_NO_WARNINGS /D_SCL_SECURE_NO_WARNINGS
set CXX_FLAGS=%CXX_FLAGS% /DMAYA_EXPORT /DMAYA_API_EXPORT /DMAYA_DEPRECATED

REM Set linker flags
set LINK_FLAGS=/DLL /SUBSYSTEM:WINDOWS /MACHINE:X64
set LINK_FLAGS=%LINK_FLAGS% /LIBPATH:%MAYA_LIB% /LIBPATH:%MAYA_LIB%\x64

echo Building US8400455 Patent-based OCD Plugin...
echo Compiler Flags: %CXX_FLAGS%
echo.

REM Create build directory
if not exist "build.patent" mkdir build.patent
cd build.patent

REM Configure CMake with patent optimizations
echo Configuring CMake with patent optimizations...
cmake .. -G "Visual Studio 16 2019" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DMAYA_VERSION=2020 ^
    -DUS8400455_PATENT=ON ^
    -DPATENT_OPTIMIZATION=ON ^
    -DCMAKE_CXX_FLAGS="%CXX_FLAGS%" ^
    -DCMAKE_EXE_LINKER_FLAGS="%LINK_FLAGS%" ^
    -DCMAKE_SHARED_LINKER_FLAGS="%LINK_FLAGS%"

if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo Building patent-optimized plugin...
cmake --build . --config Release --target offsetCurveDeformer

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESSFUL!
echo ========================================
echo.
echo Patent-optimized plugin built successfully!
echo.

REM Copy plugin to Maya plugins directory
echo Installing plugin to Maya...
if exist "src\Release\offsetCurveDeformer.mll" (
    copy "src\Release\offsetCurveDeformer.mll" "%MAYA_BIN%\plug-ins\"
    echo Plugin installed to: %MAYA_BIN%\plug-ins\
) else (
    echo WARNING: Plugin file not found!
)

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo To use the US8400455 patent technology:
echo 1. Restart Maya
echo 2. Load the plugin: Plug-in Manager
echo 3. Run: deformer -type offsetCurveDeformerPatent
echo 4. Test with the provided Python script: test_ocd_patent.py
echo.
echo Patent Features Available:
echo • 2-Phase Binding + Deformation
echo • Virtual Offset Curves (no storage overhead)
echo • Frenet Frame-based deformation (T, N, B)
echo • B-Spline + Arc Segment implementations
echo • Advanced artist controls: Twist, Slide, Scale, Rotation
echo • Volume preservation & self-intersection prevention
echo • Pose space deformation support
echo.

pause
