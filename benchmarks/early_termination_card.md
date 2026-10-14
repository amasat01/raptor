# early_termination card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-14T08:23:07Z

## Facts

```json
{
  "device": {
    "name": "Quadro P2000"
  },
  "example_script": "examples/early_termination.py",
  "example_script_md5": "fd0cab6d7ee9921d65ba372678b9a4b5",
  "generated_utc": "2026-10-14T08:23:07Z",
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
    "eagle_commit": "194656511085ea40ae53c6d513123cb072fdd443-dirty",
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
  "ratio_eager_over_guarded": 1.0156907303891312,
  "ratio_unguarded_over_guarded": 13.758791006361308,
  "steps": 4000,
  "wall_s_per_step": {
    "eager": {
      "iqr": 1.5801439876668097e-06,
      "median": 0.00021388297248631715,
      "q1": 0.00021352188763557934,
      "q3": 0.00021510203162324615,
      "samples": [
        0.0002131608027848415,
        0.00021632109076017515,
        0.00021388297248631715
      ]
    },
    "guarded": {
      "iqr": 1.202514977196689e-08,
      "median": 0.00021057883673347531,
      "q1": 0.0002105689882300794,
      "q3": 0.00021058101337985137,
      "samples": [
        0.0002105831900262274,
        0.00021057883673347531,
        0.00021055913972668349
      ]
    },
    "unguarded": {
      "iqr": 1.272478839381419e-07,
      "median": 0.0028973102049785664,
      "q1": 0.0028972473221074322,
      "q3": 0.0028973745699913704,
      "samples": [
        0.002897184439236298,
        0.0028973102049785664,
        0.002897438935004175
      ]
    }
  }
}
```
