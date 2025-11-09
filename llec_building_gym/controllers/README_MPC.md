# Pyomo Installation Guide on HPC (`README_MPC.md`)

These guidelines describe the installation and configuration of the IPOPT solver in the HPC environment, including the necessary environment variables and the installation of additional solvers helpers.

### 1. IPOPT Installation

1. incompatibilities when linking the Intel MKL libraries Intel problems
If problems occur, often due to incompatibilities in the linkage of the Intel MKL libraries (`static linkage`,`libmkl_avx512.so.2`), cloning and setting up ThirdParty ASL can help. This solves certain dependencies for ASL that Ipopt requires in some configurations.

    ```bash
    git clone https://github.com/coin-or-tools/ThirdParty-ASL.git
    cd ThirdParty-ASL/
    ./get.ASL
    ./configure --prefix=${HOME}/.local
    make
    make install
    ```

2. Change to the parent directory, download Ipopt, unpack the archive, configure and install it.

    ```bash
    cd ..
    wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.14.4.tar.gz
    tar -xvzf Ipopt-3.14.4.tar.gz
    cd Ipopt-releases-3.14.4/
    ./configure --prefix=${HOME}/.local --with-lapack-lflags="-Wl,--no-as-needed -Wl,--start-group,${MKLROOT}/lib/intel64/libmkl_intel_lp64.a,${MKLROOT}/lib/intel64/libmkl_gnu_thread.a,${MKLROOT}/lib/intel64/libmkl_core.a,--end-group -lgomp -lpthread -lm -ldl"
    make
    make test
    make install
    ```

## 2.GLPK Installation

1. Step: Download the software
2. Step: Unpack archive
3. Step: Change to the directory
4. Step: Configuration
5. Step: Installation
6. Step: Set environment variables -- In order to be able to use the installed solver, the `PATH` and `LD_LIBRARY_PATH` environment variables must be adjusted.

    ```bash
    wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz
    tar xzf glpk-5.0.tar.gz
    cd glpk-5.0/
    ./configure --prefix=$HOME/.local
    make install
    export PATH=$PATH:~/.local/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib
    ```