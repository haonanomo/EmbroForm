# Embroform

This is a Python implementation of the routing algorithm in the following paper:

EmbroForm: Digital Fabrication of Soft Freeform Objects with Machine Embroidered Pull-up Strings (CHI 2026)

## Install the package
All required Python dependencies are included in this project configuration and can be installed with a single command:

```
pip install -e .
```

The main entry point of the pipeline is `scripts/run_pipeline.py`, which runs the full layout and routing process.


## External Libraries
The preprocessing stage of this project depends on three external algorithms.
Executable files and their required dependencies are already included in this repository, so manual installation is typically not necessary.

However, if you encounter version conflicts or environment-related issues during preprocessing, please refer to the official repositories of the following three libraries:

- [EvoDevelop][evodevelop] — evolutionary optimization for developable surface approximation.
- [PP][pp] — parameterization pipeline for developable surface processing.
  **Note:** PP depends on the Pardiso linear solver. Using the Pardiso backend requires obtaining a valid license from the official Pardiso project website.
- [DevelopApp][developapp] — piecewise developable mesh approximation toolkit.

[evodevelop]: https://github.com/mmoolee/EvoDevelop
[pp]: https://github.com/ChunyangYe/PP
[developapp]: https://github.com/USTC-GCL-F/DevelopApp

