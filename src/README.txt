rmdir /s /q build
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/Users/robor/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build