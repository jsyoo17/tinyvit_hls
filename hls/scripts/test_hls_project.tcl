# scripts/test_hls_project.tcl
# Run from repo root: vitis_hls -f scripts/test_hls_project.tcl

# Make sure output dir exists and cd into it
file mkdir hls_proj
cd hls_proj

# Create/reset project in hls_proj/
open_project -reset test_hls_project

# Add design sources (linked from repo, not copied)
set_top my_top
add_files ../src/top.cpp
add_files ../src/top.hpp
# Add more as your project grows, e.g.:
# add_files ../src/blocks
# add_files ../src/utils
# or to add glob pattern:
# add_files [glob ../src/*.cpp]
# add_files [glob ../src/*.hpp]

# Add testbench
add_files -tb ../tb/tb_top.cpp
# or to add glob pattern:
# add_files -tb [glob ../tb/*.cpp]

# Create solution
open_solution "sol1"

# TODO: set correct part for your target
# For KV260/K26 SOM, device is XCK26-SFVC784-2LV-C :contentReference[oaicite:0]{index=0}
# (Vitis is usually happy with lowercase)
set_part {xck26-sfvc784-2lv-c}

# Set clock (e.g., 200 MHz)
create_clock -period 5 -name default

# Run steps
csim_design
csynth_design
# cosim_design -rtl verilog
# export_design -format ip_catalog

exit
