@echo off
FOR %%G IN (2019, 2020, 2022, 2023) DO (call :subroutine "%%G")
GOTO :eof

:subroutine
set builddir=build.%%G
if not exist %builddir% goto BUILDENV
del %builddir% /S /Q
:BUILDENV
mkdir %builddir%
cd %builddir%
if %%G LSS "2020" (
    cmake -A x64 -T v140 -DMAYA_VERSION=%%G ../
) ELSE (
    cmake -A x64 -T v141 -DMAYA_VERSION=%%G ../
)
cmake --build . --target install --config Release
cd ..
goto :eof
