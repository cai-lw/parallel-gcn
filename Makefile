CXX=g++
CXXFLAGS=-O3 -std=c++11 -Wall
LDFLAGS=-lm
CXXFILES=src/gcn.cpp src/optim.cpp src/module.cpp src/variable.cpp src/parser.cpp src/cycletimer.cpp
HFILES=src/gcn.h src/optim.h src/module.h src/variable.h src/sparse.h src/parser.h src/cycletimer.h
TEST_CXXFILES=test/module_test.cpp test/optim_test.cpp test/util.cpp
TEST_HFILES=test/util.h
OMP=-fopenmp -DOMP

all: seq omp

seq: src/main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) src/main.cpp $(LDFLAGS)

omp: src/main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) $(OMP) -o gcn-omp $(CXXFILES) src/main.cpp $(LDFLAGS)

test: $(CXXFILES) $(HFILES) $(TEST_CXXFILES) $(TEST_HFILES)
	$(CXX) $(CXXFLAGS) -Iinclude -o gcn-test $(CXXFILES) $(TEST_CXXFILES) test/main.cpp $(LDFLAGS)
