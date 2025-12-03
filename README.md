# TinyViT HLS (Vitis HLS 2022.2)
Target Model: TinyViT 5M 
Future goals
1. software verification of TinyViT model
2. hardware implementation of float32
3. software verification of fixed-point
4. hardware implementation of fixed-point
5. hardware implementation modifications
6. apply farther quantizations

# Steps to edit files
1. create a new file (e.g. new src/function.cpp)
2. add it to vitis_hls IDE by
    - project -> Add Files
    - browse to the new file location
    - remember to select "add files without copying" or "link to files" option
    