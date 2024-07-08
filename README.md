# dolfinx-examples
Examples for the use of FEniCSx. 

## Overview of contents
- checkpointing: Examples of reading and writing checkpoints for (in)stationary PDEs with adios4dolfinx
- comparison: Side-by-side comparison of legacy FEniCS and FEniCSx
- output: Examples (for standard and mixed spaces) and limitations of the different output formats.




## Getting Started and Usage

All files can be run in the command line in a standard fashion.
All files are meant as templates for experimentation. Different options can be used by commenting out according paragraphs.


## Requirements

All requirements can be found in the requirements text files and can be installed via pip by

```
pip install -r requirements.txt
```

or via conda by

```
conda create --name my-env-name --file requirements.txt -c conda-forge
```

Note that there are different requirement for legacy FEniCS and FEniCSx.




## Authors

* **Maximilian E. V. Reiter**, https://orcid.org/0000-0001-9137-7978

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details