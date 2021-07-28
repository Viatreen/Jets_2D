# Jets_2D
This application simulates and trains 2-dimensional drones to compete with each other. Points are based on airtime, striking the opponent with a bulllet, and avoiding enemy bullets. Several thousand iterations of drone matches are run in parallel with the best drones kept and copied. The copies are mutated slightly.

# Setup Instruction
An nVidia graphics card is required to run this application. 

nVidia GPU architecture target is set in CMakeLists.txt in the project root directory. 
Currently compiling for Pascal (GTX 10-series), Volta (Expensive deep learning cards), and Turing (RTX 20-series) architectures. 
This application includes the grid sync feature. This feature is only by supported by Pascal cards and newer so you must have a 10-series card or newer.
If you have a newer card than Turing, lookup the architecture of your nVidia card and append to arguments in this line: set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 62 70 72 75). 

All Systems:
Make sure CUDA Toolkit is installed. For Windows, you must have Visual Studio installed before installing CUDA.

For Windows, to setup a project in Visual Studio:
1) Open Visual Studio. From the welcome page, select "Clone a repository"
2) Use the git clone address: https://github.com/Viatreen/Jets_2D.git
3) Now choose a directory. Create a directory for the repo with the name you want for the repo (eg. Jets_2D). It will be the root directory of the project.
4) Visual Studio will clone and setup the project based on the CMakeLists.txt in the project root directory. This may take some time (~5 minutes)
5) Wait for CMake cache generation to be completed
6) Build and run the project by clicking the play button on the top of the screen or pressing F5

For Linux (This has only been tested on Ubuntu 16.04 and 18.04):
1) vscode is recommended as the code editor (Not required). It's also recommended to install the CUDA extension by nVidia
2) Ensure CMake 3.17 or higher is installed (cmake --version)
3) Run ./1_Linux_Cmake_Then_Build_Then_Run.sh
