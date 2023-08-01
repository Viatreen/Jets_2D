cd build
if not defined DevEnvDir (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
ninja && cd .. && "build\Jets_2D.exe"
cd ..