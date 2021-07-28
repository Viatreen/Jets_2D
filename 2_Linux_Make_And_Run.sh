mkdir -p build
rm -f build/Controls.app
cd build/
mkdir -p Saves/
make
cd ..
./build/Controls.app