# ----------------- BEGIN LICENSE BLOCK ---------------------------------
#
# INTEL CONFIDENTIAL
#
# Copyright (c) 2025 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you (License). Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute, disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express or
# implied warranties, other than those that are expressly stated in the License.
#
# ----------------- END LICENSE BLOCK -----------------------------------

#####################################################################
# robot vision benchmarks
#####################################################################

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# Check if benchmark library is available
find_package(benchmark QUIET)

if(benchmark_FOUND)
    message(STATUS "Google Benchmark found. Building benchmarks.")
    
    set(BENCHMARK_EXEC_NAME RobotVisionBenchmarks)
    
    set(BENCHMARK_SOURCES
        MultipleObjectTrackerBenchmark.cpp
    )
    
    add_executable(${BENCHMARK_EXEC_NAME} ${BENCHMARK_SOURCES})
    
    target_compile_options(${BENCHMARK_EXEC_NAME} PRIVATE 
        -fno-access-control
        -O3  # Optimize for benchmarking
        -DNDEBUG  # Disable debug assertions
    )
    
    target_include_directories(${BENCHMARK_EXEC_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/controller/src/robot_vision/include
    )
    
    target_link_libraries(${BENCHMARK_EXEC_NAME}
        PRIVATE
        ${PROJECT_NAME}
        benchmark::benchmark
        benchmark::benchmark_main
    )
    
    # Add custom target to run the 50-people tracking benchmark
    add_custom_target(run_benchmark_50_people
        COMMAND ${BENCHMARK_EXEC_NAME}
        DEPENDS ${BENCHMARK_EXEC_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running 50-people tracking benchmark"
    )
    
    # Installation target for benchmarks
    install(TARGETS ${BENCHMARK_EXEC_NAME}
        RUNTIME DESTINATION bin/benchmarks
        COMPONENT benchmarks
    )
    
else()
    message(STATUS "Google Benchmark not found. Skipping benchmark build.")
    message(STATUS "To install Google Benchmark:")
    message(STATUS "  sudo apt-get install libbenchmark-dev  # Ubuntu/Debian")
    message(STATUS "  brew install google-benchmark         # macOS")
    message(STATUS "  Or build from source: https://github.com/google/benchmark")
endif()
