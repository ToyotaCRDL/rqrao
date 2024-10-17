# RQRAO: Recursive Quantum Random Access Optimization

[![Python 3.6.9](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

RQRAO efficiently solves the MAX-CUT problem.

Paper: [https://arxiv.org/abs/2403.02045](https://arxiv.org/abs/2403.02045)

## Requirements

|Software|Version|
|:---:|:---:|
|Python|3.6.9|
|numpy|1.19.5|
|scikit-learn|0.24.2|
|networkx|2.5.1|
|torch|1.10.0|
|torchvision|0.11.1|

## Usage

see [scalability.ipynb](https://github.com/ToyotaCRDL/rqrao/blob/main/scalability.ipynb)

NOTE: The term `rank2` used in `classical.py` and `scalability.ipynb` refers to the CirCut method.

## Citing RQRAO

If you find it useful to use this module in your research, please cite the following paper.

```
Kondo, R., Sato, Y., Raymond, R., Yamamoto, N., Recursive Quantum Relaxation for Combinatorial Optimization Problems
, arXiv:2403.02045, (2024).
```

In Bibtex format:
 
```bibtex
@article{kondo2024recursive,
  title={Recursive Quantum Relaxation for Combinatorial Optimization Problems},
  author={Kondo, Ruho and Sato, Yuki and Raymond, Rudy and Yamamoto, Naoki},
  journal={arXiv preprint arXiv:2403.02045},
  year={2024},
  doi={10.48550/arXiv.2403.02045},
}
```

## License

See the LICENSE file for details

---

To run Recursive QAOA, Goemans-Williamson, and CirCut, please follow the steps below:


## Recursive QAOA
Please compile the `rqaoa.f90` file using the following command:
```bash
gfortran -o rqaoa rqaoa.f90
```

## Goemans-Williamson

To run Goemans-Williamson, you must have MATLAB and SDPT-3 installed.  
For more details about SDPT-3, please refer to the official site: [SDPT-3 Official Site](https://blog.nus.edu.sg/mattohkc/softwares/sdpt3/).


## CirCut

You are required to download code from an official website and modify specific parts before using it.
Please follow the steps below to apply the necessary modifications.

#### 1. Download the Code from Official Website
First, download the required code from the author's page at Rice University:

- [Rice University Author Page](https://www.cmor-faculty.rice.edu/~zhang/circut/index.html)

#### 2. Modify the Required Section
After downloading the code, you need to modify `Makefile`, `main.f90`, and `get_mod.f90` as follows:

**Filename: `src/Makefile`**

**Before modification (lines 13 and 15):**
```makefile
.f90.o:
        f90 -c $(FLAGS) $<
circut: $(OBJ)
        f90 -o $@ $(FLAGS) $(OBJ)
```

**After modification:**
```makefile
.f90.o:
        gfortran -c $(FLAGS) $<
circut: $(OBJ)
        gfortran -o $@ $(FLAGS) $(OBJ)
```

**Filename: `src/main.f90`**

**Before modification (line 28):**
```fortran
     WRITE(*,'(1X,A)') 'Enter file name (RETURN to quit): '
```

**After modification:**
```fortran
!     WRITE(*,'(1X,A)') 'Enter file name (RETURN to quit): '
```

**Before modification (line 33):**
```fortran
READ (*,'(A)',IOSTAT=i) filename
```

**After modification:**
```fortran
CALL GET_COMMAND_ARGUMENT(number=1, value=filename)
```

**Before modification (lines 52-54):**
```fortran
  WRITE(*,Fmt) 'total   time:', timer(2); WRITE(*,*)

END DO Repeat
```

**After modification:**
```fortran
  WRITE(*,Fmt) 'total   time:', timer(2); WRITE(*,*)

  EXIT

END DO Repeat
```

**Filename: `src/get_mod.f90`**

**Before modification (lines 158-159):**
```fortran
INTEGER  :: time(1)
CALL SYSTEM_CLOCK( time(1) )
```

**After modification:**
```fortran
INTEGER  :: time(12)
CALL SYSTEM_CLOCK( time(12) )
```

### 3. Move complied file
Once the code has been modified, place the compiled file into the RQRAO directory.

Please follow the steps below:

1. Change directory to the `src` directory and execute the `make` command:
```bash
cd src
make
```

2. Once the build is complete, move the generated `circut` file from `tests` to the `rqrao` directory:
```bash
mv tests/circut /path/to/rqrao/
```
