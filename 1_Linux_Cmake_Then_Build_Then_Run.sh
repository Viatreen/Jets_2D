mkdir -p build
rm -f build/Jets_2D
cmake -GNinja -DCMAKE_BUILD_TYPE="Release" -DARCH_87=1 -S . -B build/
cd build/
mkdir -p Saves/
ninja && ./Jets_2D
cd ..