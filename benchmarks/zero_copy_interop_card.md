# zero_copy_interop card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-13T17:34:39Z

## Facts

```json
{
  "device": {
    "name": "Quadro P2000"
  },
  "example_script": "examples/zero_copy_interop.py",
  "example_script_md5": "3051044400164edb76be6f06c3d08df1",
  "generated_utc": "2026-10-13T17:34:39Z",
  "peak": null,
  "script_md5": "3d2ced36cc2fa0236ca980e15bc77792",
  "software": {
    "compile": {
      "device_flags_note": "hawk.compile.toolchain.device_flags now carries the same opt_level explicitly: -O3 for nvcc's own host-side code generation and -Xptxas -O3 for ptxas; NVRTC device compiles always optimise and ignore opt_level",
      "host_flags": [
        "-O3",
        "-ffp-contract=fast",
        "-march=native",
        "-mprefer-vector-width=512",
        "-fno-trapping-math",
        "-DAETHER_HOST_VECTOR_MATH"
      ],
      "host_profile": "native-vector-math",
      "opt_level": "O3"
    },
    "cpu": "Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz",
    "cupy": "14.2.0",
    "eagle_commit": "7f1e0918a971510f85446598fb5581aedc502d78",
    "hawk": "0.5.0",
    "hawk_commit": "efaed2cd9e04c71c2647007aa54f907db69c7aa7",
    "numpy": "2.5.3",
    "omp_num_threads": null,
    "platform": "Linux-6.12.0-100.28.2.el9uek.x86_64-x86_64-with-glibc2.34",
    "python": "3.12.14",
    "torch": "2.12.1"
  }
}
```

## Results

```json
{
  "batch": 22369621,
  "pcie_bytes_naive": 1073741808,
  "ratio": 101.19398146247515,
  "size_mb": 256.0,
  "wall_s": {
    "aliased": {
      "iqr": 3.65404412150383e-06,
      "median": 0.004746937891468406,
      "q1": 0.004745207028463483,
      "q3": 0.004748861072584987,
      "samples": [
        0.004748861072584987,
        0.004829476121813059,
        0.004728499101474881,
        0.004745207028463483,
        0.004746937891468406
      ]
    },
    "naive": {
      "iqr": 0.11393348500132561,
      "median": 0.4803615449927747,
      "q1": 0.39014115696772933,
      "q3": 0.5040746419690549,
      "samples": [
        0.20754675613716245,
        0.39014115696772933,
        0.4803615449927747,
        0.5987962218932807,
        0.5040746419690549
      ]
    }
  }
}
```
