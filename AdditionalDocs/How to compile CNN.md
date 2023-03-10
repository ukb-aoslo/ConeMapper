# How to compile CNN

## Introduction

  - This instruction shows the way and problems with compilation on Windows OS.

  - If we do it with Matlab R2020b, we need MS Visual Studio 2019, CUDA Toolkit v11.1 (only for gpu version).
  - If we do it with Matlab R2016b, we need MS Visual Studio 2017/2015, CUDA Toolkit v11.1 (only for gpu version).
  To use VS2017 with Matlab R2016b, we need to install 4 bugfixes which described in next topic:
  https://uk.mathworks.com/matlabcentral/answers/335092-can-i-use-microsoft-visual-studio-2017-with-matlab-r2017a-or-r2016b

  - The base instruction you can find here: https://www.vlfeat.org/matconvnet/install/

  Next I will describe my way and which problems I have faced.

## Base steps

  - First step is to show Matlab that we have a C++ compiler.
  Use next command in Matlab command line `mex -setup C++`.
  If you have Matlab R2016b and VS2017, then you need install bugfixes which is described above.
  Otherwise you will have an error.

  - Next we should decide which type of CNN we want to compile (gpu / cpu).
  If you don't have NVIDIA videocard then you cannot use CUDA Toolkit, and you cannot compile gpu version.

## CPU compilation

  - (Starting step) To compile CPU version you shoud do next commands in Matlab command line:
    - `cd <MatConvNet>` - open MatConvNet directory as your working directory
    - `addpath matlab`  - add `./matlab` path to Matlab search path
    - `vl_compilenn`    - compile cpu version
  
  - If all compiles ok, then you will see no errors. 
  - The common error that you can face is 
    ```
    Error using vl_compilenn>check_clpath (line 591)
    Unable to find cl.exe
    ```
    It is described here: https://stackoverflow.com/questions/40226354/matconvnet-error-cl-exe-not-found .
    In short, you need to 
      - Open the Start Search, type in "env", and choose "Edit the system environment variables".
      - Click the "Environment Variables…" button.
      - Find variable "Path" in "System Variables".
      - Add path to `cl.exe`. It should be `<path to Visual Studio>\2019\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64`. In my case: `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx64\x64`.
      - After editing "Environment Variables…" you should fully close your Matlab and open it again.
    
  - After issue is solved you can do again the (Starting step).

## GPU compilation

  - (Starting step) To compile GPU version you shoud do next commands in Matlab command line:
    - `cd <MatConvNet>` - open MatConvNet directory as your working directory
    - `addpath matlab`  - add `./matlab` path to Matlab search path
    - `vl_compilenn('enableGpu', true, 'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1', 'verbose', 1, 'Debug', true)` - compile gpu version
    - Note: 'Debug' option and 'verbose' option only shows more output in console. It is not necessary for compilation.
  
  - If all compiles ok, then you will see no errors. 
  - You can face the issue which is described in "CPU compilation" topic (Unable to find cl.exe).
  - Also, you can face next problems:
  - First.
    ```
    Error using vl_compilenn>nvcc_compile (line 615)
    Command "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvcc" -c -o
    "C:\matconvnet-1.0-beta25\matlab\mex.build\bits\data.obj" "C:\matconvnet-1.0-beta25\matlab\src\bits\data.cu"
    -DENABLE_GPU -DENABLE_DOUBLE -O -DNDEBUG -D_FORCE_INLINES --std=c++11 -I"C:\Program Files\MATLAB\R2017b\extern\include"
    -I"C:\Program Files\MATLAB\R2017b\toolbox\distcomp\gpu\extern\include" -gencode=arch=compute_35,code="sm_35,compute_35"
    --compiler-options=/MD --compiler-bindir="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC..\VC\bin" failed.

    Error in vl_compilenn (line 487)
    nvcc_compile(opts, srcs{i}, objfile, flags) ;
    ```
    - You should check the path `--compiler-bindir=`. If there is no folder, then just create it. For me the pathes were next (for different Visual Studio versions):
      - C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\bin
      - C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\bin
    - Detailed discussion you can find here: https://github.com/vlfeat/matconvnet/issues/1062

  - Second.
    ```
    Error using mex
    'C:\Users\alrabm\Documents\MATLAB\MatConvNet\matconvnet-1.0-beta25\matlab\mex\vl_nnconv.mexw64'
    compiled with '-R2018a' and linked with '-R2017b'. For more information, see MEX
    file compiled with one API and linked with another.

    Error in vl_compilenn>mex_link (line 627)
    mex(args{:}) ;

    Error in vl_compilenn (line 500)
    mex_link(opts, objs, flags.mex_dir, flags) ;

    Error in compileMatConvNet (line 4)
    vl_compilenn('enableGpu', true, ...
    ```
    - Solution. In `vl_compilenn.m` modify next lines:
      - line 620: 
      ```
      args = horzcat({'-outdir', mex_dir}, ...
      flags.base, flags.mexlink, ...
      '-R2018a',...
      {['LDFLAGS=$LDFLAGS ' strjoin(flags.mexlink_ldflags)]}, ...
      {['LDOPTIMFLAGS=$LDOPTIMFLAGS ' strjoin(flags.mexlink_ldoptimflags)]}, ...
      {['LINKLIBS=' strjoin(flags.mexlink_linklibs) ' $LINKLIBS']}, ...
      objs) ;
      ```
      - line 359:
      ```
      flags.mexlink = {'-lmwblas'};
      ```
    - Detailed discussion you can find here: https://github.com/vlfeat/matconvnet/issues/1143

