# Jets_2D
This application simulates and trains 2-dimensional drones to compete with each other. Points are based on airtime, striking the opponent with a bulllet, and avoiding enemy bullets. Several thousand iterations of drone matches are run in parallel with the best drones kept and copied. The copies are mutated slightly.

# Setup Instruction
An nVidia graphics card is required to run this application. 

nVidia GPU architecture target is set in CMakeLists.txt in the project root directory. 
Currently compiling for Pascal (GTX 10-series), Volta (Expensive deep learning cards), and Turing (RTX 20-series) architectures. 
This application includes the grid sync feature. This feature is only by supported by Pascal cards and newer so you must have a 10-series card or newer.
If you have a newer card than Turing, lookup the architecture of your nVidia card and append to arguments in this line: set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 62 70 72 75). 

Visual Studio Code (vscode) is the recommended development environment. Ensure the following extensions are installed:
C/C++: Microsoft
CMake Tools: Microsoft
NSight Visual Studio Code Edition: NVIDIA

Recommended:
GitLens: Eric Amodio
Github Pull Requests and Issues: GitHub

All Systems:
Install Git
Install CMake version 3.17 or higher (In the terminal, check with "cmake --version")
Make sure CUDA Toolkit is installed. For Windows, you must have Visual Studio installed before installing CUDA. The Visual Studio compiler, MSVC, is the compiler recommended for this project.

For Windows, to setup a project in vscode:
(Visual Studio is no longer supported currently. Can be re-enabled by changing the paths to the font and shader file)
1) Open vscode. From the welcome page, select "Clone a repository"
2) Use the git clone address: https://github.com/Viatreen/Jets_2D.git and enter this URL at the top of vscode
3) Now choose a directory. This directory will receive the repo folder (i.e. Jets_2D). This folder will be the root directory of the project.
4) vscode will prompt for a compiler to select. Select "Visual Studio ... release - amd64"
5) The OUTPUT screen at the bottom of vscode will display the cmake process status
6) Build and run the project by clicking the play button on the top of the screen or pressing F5

For Linux (This has only been tested on Ubuntu 16.04 and 18.04):
1) Ensure CMake 3.17 or higher is installed (cmake --version)
2) Run ./1_Linux_Cmake_Then_Build_Then_Run.sh
