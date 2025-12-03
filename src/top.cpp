#include "top.hpp"

void my_top(const float in[1024], float out[1024]) {
#pragma HLS interface m_axi port=in  offset=slave bundle=gmem0
#pragma HLS interface m_axi port=out offset=slave bundle=gmem1
#pragma HLS interface s_axilite port=return

    for (int i = 0; i < 1024; ++i) {
    #pragma HLS pipeline II=1
        out[i] = in[i]; // placeholder
    }
}

