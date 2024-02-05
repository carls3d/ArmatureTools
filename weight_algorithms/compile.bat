
@echo off
python setup.py build_ext --inplace
@REM python setup.py build_ext --build-lib C:/Users/Carl/Documents/GitHub/ArmatureTools/cython_testing/builds
if %ERRORLEVEL% == 0 (
    echo Compilation successful.
    @REM python annotate.py
) else (
    echo Compilation failed.
    pause
)
@REM pause