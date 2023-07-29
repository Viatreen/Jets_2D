mkdir -p Saves/
mkdir -p build
rm -f build/Jets_2D
cd build/
ninja && cd .. && build/Jets_2D
cd ..