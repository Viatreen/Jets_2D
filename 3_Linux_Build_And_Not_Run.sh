mkdir -p build
rm -f build/Controls.app
cmake -S . -B build/
cd build/
mkdir -p Saves/
make
cd ..
