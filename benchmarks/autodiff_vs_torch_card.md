# autodiff_vs_torch card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-05T16:14:44Z

## Facts

```json
{
  "device": null,
  "example_script": "examples/autodiff_vs_torch.py",
  "example_script_md5": "620f0087b44d52b3519a6181480f81af",
  "generated_utc": "2026-10-05T16:14:44Z",
  "peak": null,
  "script_md5": "a88e9c626fdae545d246ddfb54f37b21",
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
[
  {
    "batch": 1,
    "hawk_jvp": 8.764513768255711e-06,
    "hawk_primal": 8.531007915735245e-06,
    "hawk_vjp": 8.361507207155228e-06,
    "ratio_jvp": 724.9246771669859,
    "ratio_vjp": 549.9620938728684,
    "torch_jvp": 0.006353612313978374,
    "torch_primal": 0.00040780541021376846,
    "torch_vjp": 0.004598512011580169
  },
  {
    "batch": 32,
    "hawk_jvp": 8.750683628022671e-06,
    "hawk_primal": 8.704210631549359e-06,
    "hawk_vjp": 8.694687858223915e-06,
    "ratio_jvp": 761.1690910203571,
    "ratio_vjp": 529.8917425836962,
    "torch_jvp": 0.006660749902948737,
    "torch_primal": 0.0003881924087181687,
    "torch_vjp": 0.004607243300415576
  },
  {
    "batch": 4096,
    "hawk_jvp": 9.070313535630702e-06,
    "hawk_primal": 8.889287710189819e-06,
    "hawk_vjp": 9.11140814423561e-06,
    "ratio_jvp": 780.9518568051195,
    "ratio_vjp": 505.8127600094038,
    "torch_jvp": 0.007083478197455406,
    "torch_primal": 0.0003769082017242908,
    "torch_vjp": 0.004608666501007974
  },
  {
    "batch": 65536,
    "hawk_jvp": 3.763390704989433e-05,
    "hawk_primal": 2.6723602786660194e-05,
    "hawk_vjp": 4.0573393926024436e-05,
    "ratio_jvp": 177.6862074384235,
    "ratio_vjp": 103.5995068328385,
    "torch_jvp": 0.0066870262147858735,
    "torch_primal": 0.0005805598106235266,
    "torch_vjp": 0.004203383601270616
  },
  {
    "batch": 1048576,
    "hawk_jvp": 0.00041452629957348106,
    "hawk_primal": 0.0002807494020089507,
    "hawk_vjp": 0.00043368719052523376,
    "ratio_jvp": 63.29814108462091,
    "ratio_vjp": 90.60238427339044,
    "torch_jvp": 0.026238744193688036,
    "torch_primal": 0.008923653094097973,
    "torch_vjp": 0.03929309349041432
  }
]
```
