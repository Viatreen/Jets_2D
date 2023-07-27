mkdir -p build
rm -f build/Jets_2D
cmake -GNinja -DCMAKE_BUILD_TYPE="Debug" -DARCH_87=1 -S . -B build/
cd build/
mkdir -p Saves/
ninja
cd ..
