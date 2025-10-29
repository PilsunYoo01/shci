# Just change SHCI_DIR to the location of the SHCI directory
SHCI_DIR := ../..

LIB_DIR := $(SHCI_DIR)/lib
SRC_DIR := $(SHCI_DIR)/src
PARENT_DIR := $(SHCI_DIR)/..
SRC := unpack_wf.cc
UTIL := $(SRC_DIR)/util.cc
HDET := $(SRC_DIR)/det/half_det.cc

# Use the same compiler and flags as the main project
CXX := g++
CXXFLAGS := -std=c++11 -O3 -fopenmp -Wall -Wextra -I$(LIB_DIR) -I$(SRC_DIR)

all : unpack_wf.cc
	$(CXX) $(CXXFLAGS) $(UTIL) $(HDET) $(SRC) -o unpack_wf

clean :
	$(RM) unpack_wf
