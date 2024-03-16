# A simple implementation of KMeans algorithm in Rust

This repository contains a simple implementation of the KMeans algorithm in the Rust language. Use it carefully!

## How to use

1. Clone and build with cargo
```shell
git clone https://github.com/ramintoosi/kmeans-rust
cd kmeans-rust/
cargo build --release
```
2. Use the app
```shell
>> target/release/kmeans -h

A simple implementation of KMeans algorithm.

Usage: kmeans [OPTIONS] --data-path <DATA_PATH> --num-cluster <NUM_CLUSTER>

Options:
  -d, --data-path <DATA_PATH>      Path to the csv file
  -n, --num-cluster <NUM_CLUSTER>  Number of clusters
  -k, --kpp                        Use Kmeans++ to initialize centers
  -m, --max-iter <MAX_ITER>        Maximum number of iterations [default: 1000]
  -t, --tolerance <TOLERANCE>      Maximum center change tolerance [default: 1e-4]
  -o, --output-path <OUTPUT_PATH>  Path to save indices as csv [default: indices.csv]
  -h, --help                       Print help
  -V, --version                    Print version

```

