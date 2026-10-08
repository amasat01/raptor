# early_termination card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-05T16:14:01Z

## Facts

```json
{
  "device": null,
  "example_script": "examples/early_termination.py",
  "example_script_md5": "fd0cab6d7ee9921d65ba372678b9a4b5",
  "generated_utc": "2026-10-05T16:14:01Z",
  "peak": null,
  "script_md5": "e069faf8b647a5975698ef24960ac023",
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
  "batch": 1000000,
  "last_termination": 285,
  "ratio_eager_over_guarded": 1.0115356488321738,
  "ratio_unguarded_over_guarded": 13.767678700316678,
  "steps": 4000,
  "wall_s_per_step": {
    "eager": {
      "iqr": 4.645338049158931e-08,
      "median": 0.00021291086223209276,
      "q1": 0.0002128916328656487,
      "q3": 0.0002129380862461403,
      "samples": [
        0.00021287240349920466,
        0.00021291086223209276,
        0.00021296531026018784
      ]
    },
    "guarded": {
      "iqr": 2.4388136807821552e-08,
      "median": 0.00021048280649119987,
      "q1": 0.00021047200672910549,
      "q3": 0.0002104963948659133,
      "samples": [
        0.0002104612069670111,
        0.00021048280649119987,
        0.00021050998324062675
      ]
    },
    "unguarded": {
      "iqr": 1.9901900668662167e-07,
      "median": 0.002897859651711769,
      "q1": 0.0028977343532314986,
      "q3": 0.0028979333722381853,
      "samples": [
        0.0028976090547512285,
        0.0028980070927646013,
        0.002897859651711769
      ]
    }
  }
}
```
