# autodiff_vs_torch card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-14T08:24:01Z

## Facts

```json
{
  "device": {
    "name": "Quadro P2000"
  },
  "example_script": "examples/autodiff_vs_torch.py",
  "example_script_md5": "55c125efbca1039f9714a220c842f429",
  "generated_utc": "2026-10-14T08:24:01Z",
  "peak": null,
  "script_md5": "a917a7b2f5f77a3b3594255d90fa9515",
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
    "eagle_commit": "a6c663c9f3d0f92c7731272c3922fdb8cd2ac039-dirty",
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
[
  {
    "batch": 1,
    "hawk_jvp": 1.0845204815268517e-05,
    "hawk_primal": 1.129999291151762e-05,
    "hawk_vjp": 1.0760407894849777e-05,
    "ratio_jvp": 519.0661488456369,
    "ratio_vjp": 179.10444741602404,
    "torch_jvp": 0.005629378696903586,
    "torch_primal": 0.00037604440003633497,
    "torch_vjp": 0.0019272369099780917
  },
  {
    "batch": 32,
    "hawk_jvp": 1.1188816279172897e-05,
    "hawk_primal": 1.1591799557209015e-05,
    "hawk_vjp": 1.1876202188432217e-05,
    "ratio_jvp": 578.3104965914482,
    "ratio_vjp": 173.6356427141678,
    "torch_jvp": 0.006470609898678958,
    "torch_primal": 0.000385903287678957,
    "torch_vjp": 0.002062131999991834
  },
  {
    "batch": 4096,
    "hawk_jvp": 1.917639747262001e-05,
    "hawk_primal": 1.2703798711299897e-05,
    "hawk_vjp": 1.9025080837309362e-05,
    "ratio_jvp": 323.7404264102377,
    "ratio_vjp": 111.25444456818514,
    "torch_jvp": 0.006208175094798207,
    "torch_primal": 0.0003944491036236286,
    "torch_vjp": 0.0021166248014196755
  },
  {
    "batch": 65536,
    "hawk_jvp": 3.766519948840141e-05,
    "hawk_primal": 3.212851006537676e-05,
    "hawk_vjp": 3.9733410812914374e-05,
    "ratio_jvp": 177.8259160491263,
    "ratio_vjp": 67.20777691898857,
    "torch_jvp": 0.006697848602198064,
    "torch_primal": 0.000547284004278481,
    "torch_vjp": 0.0026703942101448776
  },
  {
    "batch": 1048576,
    "hawk_jvp": 0.0004158449824899435,
    "hawk_primal": 0.00028373610693961383,
    "hawk_vjp": 0.00043188608251512053,
    "ratio_jvp": 63.28080487084112,
    "ratio_vjp": 91.18141660475081,
    "torch_jvp": 0.02631500519346446,
    "torch_primal": 0.008950226218439638,
    "torch_vjp": 0.039379984815604986
  }
]
```
