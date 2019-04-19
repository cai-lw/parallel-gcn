CXX=g++
CXXFLAGS=-O3 -std=c++11 -Wall
LDFLAGS=-lm
CXXFILES=src/main.cpp src/gcn.cpp src/optim.cpp src/module.cpp src/variable.cpp src/parser.cpp
HFILES=src/gcn.h src/optim.h src/module.h src/variable.h src/sparse.h src/parser.h

all: gcn-seq

gcn-seq: $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) $(LDFLAGS)
