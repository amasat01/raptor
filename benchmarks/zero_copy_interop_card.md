# zero_copy_interop card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-05T16:14:52Z

## Facts

```json
{
  "device": null,
  "example_script": "examples/zero_copy_interop.py",
  "example_script_md5": "3051044400164edb76be6f06c3d08df1",
  "generated_utc": "2026-10-05T16:14:52Z",
  "peak": null,
  "script_md5": "11040c48d22965d4831ef0556f954d5e",
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
    "eagle_commit": "7e445717a3e0e2ce81e2d26e44da9bde34ff73c4-dirty",
    "hawk": "0.3.0",
    "hawk_commit": "6a9b724cc889401bdb550b42f6206b15d34cc899-dirty",
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
  "ratio": 113.6234721293542,
  "size_mb": 256.0,
  "wall_s": {
    "aliased": {
      "iqr": 2.115592360496521e-05,
      "median": 0.004707104992121458,
      "q1": 0.004696669988334179,
      "q3": 0.004717825911939144,
      "samples": [
        0.004758627153933048,
        0.004717825911939144,
        0.004707104992121458,
        0.0046924620401114225,
        0.004696669988334179
      ]
    },
    "naive": {
      "iqr": 0.027707295026630163,
      "median": 0.5348376128822565,
      "q1": 0.5330876898951828,
      "q3": 0.560794984921813,
      "samples": [
        0.560794984921813,
        0.5348376128822565,
        0.5761333380360156,
        0.5301863770000637,
        0.5330876898951828
      ]
    }
  }
}
```
