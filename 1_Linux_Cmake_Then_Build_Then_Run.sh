rm -rf build
mkdir -p Saves/
mkdir -p build
rm -f build/Jets_2D
cd build/
cmake -DCMAKE_BUILD_TYPE="Release" -DARCH_61=1 -S .. -B .
make && cd .. && build/Jets_2D
