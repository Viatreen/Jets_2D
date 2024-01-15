mkdir -p Saves/
mkdir -p build
rm -f build/Jets_2D
cd build/
make && cd .. && build/Jets_2D
cd ..
