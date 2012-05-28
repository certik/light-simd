# The make file for Light-SIMD

# Detect platform
#--------------------

UNAME := $(shell uname -s)
MACH_TYPE := $(shell uname -m)

ifneq ($(MACH_TYPE), x86_64)
    $(error Only 64-bit platform is supported currently)
endif


# Compiler configuration
#-------------------------

SVML_LNKPATH=/opt/intel/lib

WARNING_FLAGS = -Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter
CPPFLAGS = -I. -L$(SVML_LNKPATH)

ifeq ($(UNAME), Linux)
	CXX=g++
	CXXFLAGS = -std=c++0x -pedantic -march=native $(WARNING_FLAGS) $(CPPFLAGS) 
endif
ifeq ($(UNAME), Darwin)
	CXX=clang++
	CXXFLAGS = -std=c++0x -stdlib=libc++ -pedantic -march=native $(WARNING_FLAGS) $(CPPFLAGS)
endif

# directory configuration

INC=light_simd
BIN=bin


#------ Header groups ----------

COMMON_H = \
	$(INC)/arch.h \
	$(INC)/common/common_base.h
	
SSE_H = $(COMMON_H) \
	$(INC)/sse/sse_base.h \
	$(INC)/sse/sse_arith.h \
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
	$(BIN)/test_sse_vecs \
	$(BIN)/test_sse_arith
	
$(BIN)/test_sse_vecs : $(SSE_H) tests/test_sse_vecs.cpp
	$(CXX) $(CXXFLAGS) tests/test_sse_vecs.cpp -o $@

$(BIN)/test_sse_arith:  $(SSE_H) tests/test_sse_arith.cpp
	$(CXX) $(CXXFLAGS) -O2 tests/test_sse_arith.cpp -o $@
	
	
bench_sse: \
	$(BIN)/bench_sse_arith \
	$(BIN)/bench_sse_math
	
$(BIN)/bench_sse_arith: $(SSE_H) tests/bench_sse_arith.cpp
	$(CXX) $(CXXFLAGS) -O2 tests/bench_sse_arith.cpp -o $@
	
$(BIN)/bench_sse_math: $(SSE_H) tests/bench_sse_math.cpp
	$(CXX) $(CXXFLAGS) -O2 -DLSIMD_USE_SVML tests/bench_sse_math.cpp -lsvml -o $@
	
	
	
