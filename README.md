# Parallelized Graph Convolutional Networks
From-scratch multithreaded CPU implementation of Graph Convolutional Networks ([paper](https://arxiv.org/pdf/1609.02907.pdf) and [blog](http://tkipf.github.io/graph-convolutional-networks/)).

A course project of [CMU 15-618: Parallel Computer Architecture and Programming](http://www.cs.cmu.edu/~418/) in 2019 Spring, by Liwei Cai (me) and Chengze Fan ([@ccczzzf](https://github.com/ccczzzf)).

See the [proposal](https://github.com/cai-lw/parallel-gcn/blob/master/reports/proposal.pdf) for details.

## Setup

### Install and run the original GCN implementation

```sh
# This is for GHC machines. Use the right way of "pip install" for your environment on other machines.
python3 -m pip install --user tensorflow scipy networkx
git clone https://github.com/tkipf/gcn.git
cd gcn
python3 setup.py install --user
cd gcn
python3 train.py
```

### Convert data format

Either copy the `data` folder in the original GCN repo to this repo and run `convert_data.py`, or simply decompress `data.tgz`.

### Build and run

```sh
make
./gcn-seq cora # dataset name
```

### Tests

We use [Catch2](https://github.com/catchorg/Catch2) for unit tests.

Download the single header file and put it in `include/catch2/catch.hpp`. Then

```sh
make test
./gcn-test
```