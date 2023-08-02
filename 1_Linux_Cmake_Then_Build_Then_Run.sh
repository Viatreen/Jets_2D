rm -rf build
mkdir -p Saves/
mkdir -p build
rm -f build/Jets_2D
cd build/
cmake -GNinja -DCMAKE_BUILD_TYPE="Release" -DARCH_61=1 -S .. -B .
ninja && cd .. && build/Jets_2D
