# The CMakeLists file for Light-SIMD (Testing)

cmake_minimum_required(VERSION 2.8)
project(LightSIMD)

enable_testing()

include_directories(..)

# Compiler configuration

set(CMAKE_BUILD_TYPE "Release")

set(LANG_FLAGS "-std=c++0x -pedantic -march=native ")
set(WARNING_FLAGS "-Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter ")
set(EXTINC_FLAGS "-isystem $ENV{LIGHT_TEST_HOME}")

if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	set(LANG_FLAGS "${LANG_FLAGS} -stdlib=libc++ -Qunused-arguments")
	set(CMAKE_CXX_COMPILER "clang++")
endif (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

set(TEST_CXX_FLAGS "${LANG_FLAGS} ${WARNING_FLAGS} ${EXTINC_FLAGS}")

set(CMAKE_CXX_FLAGS "${TEST_CXX_FLAGS}")

# External link libraries

set(SVML_PATH "$ENV{ICC_LIBPATH}")
link_directories(${SVML_PATH})

# Header file groups (to be used as dependencies)

set(INC ../light_simd)

set(COMMON_BASIC_HS 
    ${INC}/common/common_base.h 
    ${INC}/common/simd_pack.h
    ${INC}/common/simd_arith.h)
    
set(COMMON_MATH_HS
    ${INC}/common/simd_math.h)
    
set(COMMON_LINALG_HS
    ${INC}/common/simd_vec.h
    ${INC}/common/simd_mat.h)

set(SSE_BASIC_HS 
    ${INC}/sse/sse_base.h 
    ${INC}/sse/sse_pack.h 
    ${INC}/sse/sse_arith.h
    ${INC}/sse/details/sse_pack_bits.h)

set(SSE_MATH_HS 
    ${INC}/sse/sse_math.h)

set(SSE_LINALG_HS 
    ${INC}/sse/sse_vec.h 
    ${INC}/sse/sse_mat.h
    ${INC}/sse/details/sse_mat_bits.h
    ${INC}/sse/details/sse_mat_comp_bits.h
    ${INC}/sse/details/sse_mat_matmul_bits.h
    ${INC}/sse/details/sse_mat_sol_bits.h)
    
set(SSE_BASIC_DEP_HS
    ${COMMON_BASIC_HS}
    ${SSE_BASIC_HS})
    
set(SSE_MATH_DEP_HS
    ${SSE_BASIC_DEP_HS}
    ${SSE_MATH_HS})
    
set(SSE_LINALG_DEP_HS
    ${SSE_BASIC_DEP_HS}
    ${SSE_LINALG_HS})
    

# Executables

add_library(test_main STATIC test_main.cpp)

add_executable(test_sse_packs ${SSE_BASIC_DEP_HS} test_sse_packs.cpp)
add_executable(test_sse_arith ${SSE_BASIC_DEP_HS} test_sse_arith.cpp)

add_executable(test_sse_vecs ${SSE_LINALG_DEP_HS} test_sse_vecs.cpp)
add_executable(test_sse_mats ${SSE_LINALG_DEP_HS} test_sse_mats.cpp)
add_executable(test_sse_mm   ${SSE_LINALG_DEP_HS} test_sse_mm.cpp)
add_executable(test_sse_sol  ${SSE_LINALG_DEP_HS} test_sse_sol.cpp)

add_executable(test_sse_math_svml ${SSE_MATH_DEP_HS} test_sse_math.cpp)

set(ALL_EXECUTABLES 
    test_sse_packs
    test_sse_arith
    test_sse_vecs
    test_sse_mats
    test_sse_mm
    test_sse_sol
    test_sse_math_svml)
    
set_target_properties(${ALL_EXECUTABLES}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "bin")    

target_link_libraries(test_sse_packs test_main)
target_link_libraries(test_sse_arith test_main)
target_link_libraries(test_sse_math_svml test_main)
target_link_libraries(test_sse_vecs test_main)
target_link_libraries(test_sse_mats test_main)
target_link_libraries(test_sse_mm test_main)
target_link_libraries(test_sse_sol test_main)

set_target_properties(test_sse_math_svml
	PROPERTIES
	COMPILE_FLAGS "-DLSIMD_USE_SVML"
	LINK_FLAGS "-lsvml"
)


# Add Tests

add_test(NAME sse_packs COMMAND test_sse_packs)
add_test(NAME sse_arith COMMAND test_sse_arith)

add_test(NAME sse_vecs COMMAND test_sse_vecs)
add_test(NAME sse_mats COMMAND test_sse_mats)
add_test(NAME sse_mm   COMMAND test_sse_mm)
add_test(NAME sse_sol  COMMAND test_sse_sol)

add_test(NAME sse_math_svml COMMAND test_sse_math_svml)




