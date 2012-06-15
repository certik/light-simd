# The CMakeLists file for Light-SIMD (Benchmark)

cmake_minimum_required(VERSION 2.8)
project(LightSIMD_Benchmark)

include_directories(..)

# Compiler configuration

set(CMAKE_BUILD_TYPE "Release")

set(LANG_FLAGS "-pedantic -march=native ")
set(WARNING_FLAGS "-Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter ")

set(BENCH_CXX_FLAGS "${LANG_FLAGS} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS "${BENCH_CXX_FLAGS}")


# External link libraries

set(SVML_PATH "$ENV{ICC_LIBPATH}")
link_directories(${SVML_PATH})

# Executables

add_executable(bench_sse_arith bench_sse_arith.cpp)
add_executable(bench_sse_reduce bench_sse_reduce.cpp)

add_executable(bench_sse_vecs bench_sse_vecs.cpp)
add_executable(bench_sse_mats bench_sse_mats.cpp)
add_executable(bench_sse_mm   bench_sse_mm.cpp)

add_executable(bench_sse_math_svml bench_sse_math.cpp)

set(ALL_EXECUTABLES
    bench_sse_arith
    bench_sse_reduce
    bench_sse_vecs
    bench_sse_mats
    bench_sse_mm
    bench_sse_math_svml)

set_target_properties(${ALL_EXECUTABLES}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "bin")  

set_target_properties(bench_sse_math_svml
	PROPERTIES
	COMPILE_FLAGS "-DLSIMD_USE_INTEL_SVML"
	LINK_FLAGS "-lsvml"
)



