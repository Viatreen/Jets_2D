rmdir /Q /S build
mkdir build
mkdir Saves
cd build
if not defined DevEnvDir (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
cmake -GNinja -DCMAKE_BUILD_TYPE="Debug" -DARCH_61=1 -S .. -B .
ninja && Jets_2D.exe
cd ..

@REM "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" build\Jets_2D.vcxproj && build\Debug\Jets_2D.exe