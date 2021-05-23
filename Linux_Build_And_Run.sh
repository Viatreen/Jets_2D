mkdir -p build
cmake -S . -B build/
cd build/
mkdir -p Saves/
make
./Controls.app