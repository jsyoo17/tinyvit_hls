# scripts/hls_project.tcl
# Run from repo root: vitis_hls -f scripts/hls_project.tcl

# Create/reset project in hls_proj/
open_project -reset hls_proj/my_top_hls

# Add design sources (linked from repo, not copied)
set_top my_top
add_files ./src/top.cpp
add_files ./src/top.hpp
# Add more as your project grows, e.g.:
# add_files ./src/blocks
# add_files ./src/utils

# Add testbench
add_files -tb ./tb/tb_top.cpp

# Create solution
open_solution "sol1"
# Set the right part for your board (example: KV260)
# Replace this with the exact part later
set_part {xc7z020clg400-1}

# Set clock (e.g., 200 MHz)
create_clock -period 5 -name default

# Optional: enable config options, directives, etc.

# Run steps
csim_design
csynth_design
# cosim_design -rtl verilog
# export_design -format ip_catalog

exit

