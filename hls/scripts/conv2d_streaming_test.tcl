# hls/scripts/conv2d_streaming_test.tcl
# Run from repo root:
#   vitis_hls -f hls/scripts/conv2d_streaming_test.tcl

# =============================================================================
# === USER CONFIG SECTION =====================================================
# =============================================================================

# Project name (used as Vitis HLS project name AND folder name under hls/build/)
set proj_name      "conv2d_streaming_test"

# Top-level function name in your C/C++ source
set top_name       "conv2d_nchw_streaming_top"

# Device / part (KV260: xck26-sfvc784-2LV-c)
set part_name      "xck26-sfvc784-2LV-c"

# Target clock period in ns (10 ns = 100 MHz)
set clock_period   10

# Design source files (relative to hls/src/)
set design_sources {
    "conv2d_streaming_im2col.cpp"
}

# Testbench source files (relative to hls/tb/)
set tb_sources {
    "tb_conv2d_streaming.cpp"
}

# Which steps to run (set to 0/1 as needed)
set run_csim   1
set run_csynth 0
set run_cosim  0
set run_export 0

# =============================================================================
# === PATH RESOLUTION (USUALLY NO NEED TO TOUCH BELOW THIS LINE) ==============
# =============================================================================

# Resolve script directory (hls/scripts/)
set script_dir [file normalize [file dirname [info script]]]
# hls/ directory
set hls_dir    [file normalize [file join $script_dir ".."]]
# build directory: hls/build/
set build_dir  [file join $hls_dir "build"]
# full project directory: hls/build/<proj_name>/
set proj_dir   [file join $build_dir $proj_name]

# Make sure build directory exists
file mkdir $build_dir

# -----------------------------------------------------------------------------
# Create / reset project
# -----------------------------------------------------------------------------
cd $build_dir
open_project -reset $proj_name

# Set top function
set_top $top_name

# Add design sources from hls/src/
foreach f $design_sources {
    set fullpath [file join $hls_dir "src" $f]
    if {[file exists $fullpath]} {
        add_files $fullpath
    } else {
        puts "WARNING: design source not found: $fullpath"
    }
}

# Add testbench sources from hls/tb/
foreach f $tb_sources {
    set fullpath [file join $hls_dir "tb" $f]
    if {[file exists $fullpath]} {
        add_files -tb $fullpath
    } else {
        puts "WARNING: testbench source not found: $fullpath"
    }
}

# -----------------------------------------------------------------------------
# Solution / device / clock
# -----------------------------------------------------------------------------
open_solution -reset "sol1"

set_part $part_name
create_clock -period $clock_period -name default

# -----------------------------------------------------------------------------
# Run flow (controlled by flags in USER CONFIG)
# -----------------------------------------------------------------------------
if {$run_csim} {
    puts "=== Running C Simulation (csim_design) ==="
    csim_design
}

if {$run_csynth} {
    puts "=== Running C Synthesis (csynth_design) ==="
    csynth_design
}

if {$run_cosim} {
    puts "=== Running Co-Simulation (cosim_design) ==="
    cosim_design -rtl verilog
}

if {$run_export} {
    puts "=== Exporting design (export_design) ==="
    export_design -format ip_catalog
}

exit
