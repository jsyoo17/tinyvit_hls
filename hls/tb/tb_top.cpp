#include <cstdio>
#include "top.hpp"

int main() {
    float in[1024];
    float out[1024];

    for (int i = 0; i < 1024; ++i) {
        in[i] = i;
        out[i] = 0;
    }

    my_top(in, out);

    int errors = 0;
    for (int i = 0; i < 1024; ++i) {
        if (out[i] != in[i]) {
            errors++;
        }
    }

    if (errors == 0) {
        std::printf("TEST PASSED\n");
        return 0;
    } else {
        std::printf("TEST FAILED: %d errors\n", errors);
        return 1;
    }
}

