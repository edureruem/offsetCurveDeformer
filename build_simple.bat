@echo off
echo ========================================
echo Offset Curve Deformer Simple Build
echo ========================================

REM 기존 빌드 폴더 정리
if exist "build.2020" (
    echo Cleaning previous build...
    rmdir /s /q "build.2020"
)

REM 새 빌드 폴더 생성
mkdir build.2020
cd build.2020

REM CMake 설정 (drawCurveContext와 동일한 방식)
echo Configuring with CMake...
cmake -A x64 -T v143 -DMAYA_VERSION=2020 ../

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM 빌드 실행
echo Building project...
cmake --build . --target install --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Plugin location: build.2020/src/Release/offsetCurveDeformer.mll
echo.
pause
