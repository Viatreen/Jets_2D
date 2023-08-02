mkdir -p Saves/
mkdir -p build
rm -f build/Jets_2D
cd build/
cmake -GNinja -DCMAKE_BUILD_TYPE="Debug" -DARCH_87=1 -S .. -B .
ninja
cd ..
