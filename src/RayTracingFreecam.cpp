#include <iostream>
#include "display.h"

int launchDisplay(); // forward declare

int main() {
    std::cout << "CPU start\n";
    return launchDisplay();
    std::cout << "CPU end\n";
    return 0;
}
