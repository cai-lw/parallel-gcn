CXX=g++
CXXFLAGS=-O3 -std=c++11 -Wall
LDFLAGS=-lm
CXXFILES=main.cpp gcn.cpp optim.cpp module.cpp variable.cpp parser.cpp
HFILES=gcn.h optim.h module.h variable.h sparse.h parser.h

all: gcn-seq

gcn-seq: #$(CXXFILES) $(HFILES)
	cd src; $(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) $(LDFLAGS); cd ..; mv src/gcn-seq .
