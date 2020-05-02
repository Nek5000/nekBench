# nekBench 

nekBench is a benchmark suite representing key components of nek5000 and nekRS.
It serves as a lightweight tool for performance analysis on high performance computing architectures. 

The code uses the [CEED](https://ceed.exascaleproject.org/) software products [OCCA](https://github.com/libocca/occa) and [benchParanumal](https://github.com/paranumal/benchparanumal). 

### Available benchmarks
* bw
* gs (not available yet)
* axhelm
* nekBone
* adv

## Installation

The suite can be built using 
```
make <INSTALLDIR=$HOME/nb>
```
Please edit the makefile to customize build settings.


## License
nekBench is released under the BSD 3-clause license (see the LICENSE file).
All new contributions must be made under the BSD 3-clause license.

## Acknowledgment
This research was supported by the Exascale Computing Project (17-SC-20-SC),
a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security
Administration, responsible for delivering a capable exascale ecosystem, including software,
applications, and hardware technology, to support the nation’s exascale computing imperative.
