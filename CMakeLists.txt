# The CMakeLists file for Light-SIMD

cmake_minimum_required(VERSION 2.8)

project(LightSIMD)

include_directories(.)

# Compiler configuration

set(CMAKE_BUILD_TYPE "Release")

set(LANG_FLAGS "-std=c++0x -pedantic -march=native ")
set(WARNING_FLAGS "-Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter ")
set(EXTINC_FLAGS "-isystem $ENV{LIGHT_TEST_HOME}")

if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	set(LANG_FLAGS "-std=c++0x -stdlib=libc++ -pedantic -march=native -Qunused-arguments")
	set(CMAKE_CXX_COMPILER "clang++")
endif (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

set(TEST_CXX_FLAGS "${LANG_FLAGS} ${WARNING_FLAGS} ${EXTINC_FLAGS}")

set(CMAKE_CXX_FLAGS "${TEST_CXX_FLAGS}")

# External link libraries

set(SVML_PATH "$ENV{ICC_LIBPATH}")
link_directories(${SVML_PATH})

# Tests

set(TMAIN "tests/test_main.cpp")

add_executable(test_sse_packs tests/test_sse_packs.cpp ${TMAIN})
add_executable(test_sse_arith tests/test_sse_arith.cpp)

add_executable(test_sse_math_svml tests/test_sse_math.cpp)
set_target_properties(test_sse_math_svml
	PROPERTIES
	COMPILE_FLAGS "-DLSIMD_USE_SVML"
	LINK_FLAGS "-lsvml"
)

add_executable(test_sse_vecs tests/test_sse_vecs.cpp ${TMAIN})
add_executable(test_sse_mats tests/test_sse_mats.cpp ${TMAIN})
add_executable(test_sse_mm   tests/test_sse_mm.cpp   ${TMAIN})
add_executable(test_sse_sol  tests/test_sse_sol.cpp  ${TMAIN})

