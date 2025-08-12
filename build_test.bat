@echo off
echo ========================================
echo Offset Curve Deformer Build Test
echo ========================================

REM 기존 빌드 폴더 정리
if exist "build.clean" (
    echo Cleaning previous build...
    rmdir /s /q "build.clean"
)

REM 새 빌드 폴더 생성
mkdir "build.clean"
cd "build.clean"

REM CMake 설정
echo Configuring with CMake...
cmake .. -G "Visual Studio 16 2019" -A x64 -DMAYA_VERSION=2020

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM 빌드 실행
echo Building project...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Plugin location: build.clean\src\Release\offsetCurveDeformer.mll
echo.
echo To install to Maya:
echo 1. Copy offsetCurveDeformer.mll to your Maya plug-ins folder
echo 2. Restart Maya
echo 3. Load the plugin: Plug-in Manager
echo.
pause
