# early_termination card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-13T17:33:56Z

## Facts

```json
{
  "device": {
    "name": "Quadro P2000"
  },
  "example_script": "examples/early_termination.py",
  "example_script_md5": "fd0cab6d7ee9921d65ba372678b9a4b5",
  "generated_utc": "2026-10-13T17:33:56Z",
  "peak": null,
  "script_md5": "da8723e95522e5806d722c7bd55d25f4",
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
  "batch": 1000000,
  "last_termination": 285,
  "ratio_eager_over_guarded": 1.0132892654097931,
  "ratio_unguarded_over_guarded": 13.758342440072317,
  "steps": 4000,
  "wall_s_per_step": {
    "eager": {
      "iqr": 6.581252673640878e-07,
      "median": 0.0002133571222657338,
      "q1": 0.0002132584268692881,
      "q3": 0.00021391655213665218,
      "samples": [
        0.0002133571222657338,
        0.00021447598200757055,
        0.0002131597314728424
      ]
    },
    "guarded": {
      "iqr": 3.810011548930573e-09,
      "median": 0.0002105589485145174,
      "q1": 0.00021055529211298562,
      "q3": 0.00021055910212453455,
      "samples": [
        0.0002105589485145174,
        0.0002105592557345517,
        0.00021055163571145387
      ]
    },
    "unguarded": {
      "iqr": 1.232923823408713e-07,
      "median": 0.0028969421174842866,
      "q1": 0.002896890865988098,
      "q3": 0.002897014158370439,
      "samples": [
        0.0028970861992565914,
        0.0028968396144919096,
        0.0028969421174842866
      ]
    }
  }
}
```
