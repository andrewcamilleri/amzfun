#!/bin/sh

sudo yum install git
git clone https://github.com/xianyi/OpenBlas.git
cd OpenBlas/
make clean
make -j4
sudo mkdir /usr/lib64/OpenBLAS
sudo chmod o+w,g+w /usr/lib64/OpenBLAS/
make PREFIX=/usr/lib64/OpenBLAS install
sudo rm /etc/ld.so.conf.d/atlas-x86_64.conf
sudo ldconfig
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/libblas.so
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/libblas.so.3
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/libblas.so.3.5
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/liblapack.so
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/liblapack.so.3
sudo ln -sf /usr/lib64/OpenBLAS/lib/libopenblas.so /usr/lib64/liblapack.so.3.5