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

WARNING_FLAGS = -Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter
CPPFLAGS = -I. 

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
	$(INC)/sse/sse_const.h \
	$(INC)/sse/sse_arith.h \
	$(INC)/sse.h
	

#---------- Target groups -------------------

.PHONY: all
all: test

.PHONY: test
test: test_sse

.PHONY: bench
bench: bench_sse

.PHONY: clean

clean:
	-rm $(BIN)/*
	

#--------- Target details -----------------

test_sse: \
	$(BIN)/test_sse_packs 
	
$(BIN)/test_sse_packs : $(SSE_H) tests/test_sse_packs.cpp
	$(CXX) $(CXXFLAGS) tests/test_sse_packs.cpp -o $@

	
	
	
	

