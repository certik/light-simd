# The make file for Light-SIMD

# Detect platform
#--------------------

UNAME := $(shell uname -s)
MACH_TYPE := $(shell uname -m)

ifndef LIGHT_TEST_HOME
	$(error The environment variable LIGHT_TEST_HOME need to be set)
endif


# Compiler configuration
#-------------------------

ifndef ICC_LIBPATH
	$(error The environment variable ICC_LIBPATH need to be set)
endif

WARNING_FLAGS = -Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter
CPPFLAGS = -I. -I$(LIGHT_TEST_HOME) -L$(ICC_LIBPATH)

ifeq ($(UNAME), Linux)
	CXX=g++
	CXXFLAGS = -std=c++0x -pedantic -march=native $(WARNING_FLAGS) $(CPPFLAGS) 
	CXX_B=g++
	CXXFLAGS_B=-pedantic -march=native -mtune=native -O3 $(WARNING_FLAGS) $(CPPFLAGS) 
endif

ifeq ($(UNAME), Darwin)
	CXX=clang++
	CXXFLAGS = -std=c++0x -stdlib=libc++ -pedantic -march=native $(WARNING_FLAGS) $(CPPFLAGS)
	CXX_B=icpc
	CXXFLAGS_B=-pedantic -xsse4.2 -O3 $(WARNING_FLAGS) $(CPPFLAGS) 
endif



# directory configuration

INC=light_simd
BIN=bin

TMAIN=tests/test_main.cpp


#------ Header groups ----------

COMMON_H = \
	$(INC)/arch.h \
	$(INC)/common/common_base.h \
	$(INC)/common/simd_pack.h \
	$(INC)/common/simd_arith.h \
	$(INC)/common/simd_math.h \
	$(INC)/common/simd_vec.h \
	$(INC)/common/simd_mat.h
	
SSE_H = $(COMMON_H) \
	$(INC)/sse/details/sse_pack_bits.h \
	$(INC)/sse/details/sse_mat_bits.h \
	$(INC)/sse/details/sse_mat_comp_bits.h \
	$(INC)/sse/details/sse_mat_inv_bits.h \
	$(INC)/sse/sse_base.h \
	$(INC)/sse/sse_pack.h \
	$(INC)/sse/sse_arith.h \
	$(INC)/sse/sse_math.h \
	$(INC)/sse/sse_vec.h \
	$(INC)/sse/sse_mat.h \
	$(INC)/sse.h
	

#---------- Target groups -------------------

.PHONY: all
all: test bench

.PHONY: test
test: test_sse

.PHONY: bench
bench: bench_sse

.PHONY: clean

clean:
	-rm $(BIN)/*
	

#--------- Target details -----------------

test_sse: \
	$(BIN)/test_sse_packs \
	$(BIN)/test_sse_arith \
	$(BIN)/test_sse_math_svml \
	$(BIN)/test_sse_vecs \
	$(BIN)/test_sse_mats \
	$(BIN)/test_sse_mm \
	$(BIN)/test_sse_inv
	
$(BIN)/test_sse_packs : $(SSE_H) tests/test_sse_packs.cpp
	$(CXX) $(CXXFLAGS) tests/test_sse_packs.cpp $(TMAIN) -o $@

$(BIN)/test_sse_arith:  $(SSE_H) tests/test_sse_arith.cpp
	$(CXX) $(CXXFLAGS) -O2 tests/test_sse_arith.cpp -o $@
	
$(BIN)/test_sse_math_svml:  $(SSE_H) tests/test_sse_math.cpp
	$(CXX) $(CXXFLAGS) -O2 -DLSIMD_USE_SVML tests/test_sse_math.cpp -lsvml -o $@
	
$(BIN)/test_sse_vecs: $(SSE_H) tests/test_sse_vecs.cpp
	$(CXX) $(CXXFLAGS) tests/test_sse_vecs.cpp $(TMAIN) -o $@
	
$(BIN)/test_sse_mats: $(SSE_H) tests/test_sse_mats.cpp
	$(CXX) $(CXXFLAGS) tests/test_sse_mats.cpp $(TMAIN) -o $@
	
$(BIN)/test_sse_mm: $(SSE_H) tests/test_sse_mm.cpp
	$(CXX) $(CXXFLAGS) tests/test_sse_mm.cpp $(TMAIN) -o $@	
	
$(BIN)/test_sse_inv: $(SSE_H) tests/test_sse_inv.cpp
	$(CXX) $(CXXFLAGS) tests/test_sse_inv.cpp $(TMAIN) -o $@
		
	
bench_sse: \
	$(BIN)/bench_sse_arith \
	$(BIN)/bench_sse_math_svml \
	$(BIN)/bench_sse_reduce \
	$(BIN)/bench_sse_vecs \
	$(BIN)/bench_sse_mats
	
$(BIN)/bench_sse_arith: $(SSE_H) tests/bench_sse_arith.cpp
	$(CXX_B) $(CXXFLAGS_B) tests/bench_sse_arith.cpp -lsvml -o $@
	
$(BIN)/bench_sse_math_svml: $(SSE_H) tests/bench_sse_math.cpp
	$(CXX_B) $(CXXFLAGS_B) -DLSIMD_USE_SVML tests/bench_sse_math.cpp -lsvml -o $@
	
$(BIN)/bench_sse_reduce: $(SSE_H) tests/bench_sse_reduce.cpp
	$(CXX_B) $(CXXFLAGS_B) tests/bench_sse_reduce.cpp -o $@
	
$(BIN)/bench_sse_vecs: $(SSE_H) tests/bench_sse_vecs.cpp
	$(CXX_B) $(CXXFLAGS_B) tests/bench_sse_vecs.cpp -o $@

$(BIN)/bench_sse_mats: $(SSE_H) tests/bench_sse_mats.cpp
	$(CXX_B) $(CXXFLAGS_B) tests/bench_sse_mats.cpp -o $@





	
	
	
