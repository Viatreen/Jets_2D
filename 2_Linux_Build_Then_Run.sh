mkdir -p build
rm -f build/Jets_2D
cd build/
mkdir -p Saves/
ninja && ./Jets_2D
cd ..