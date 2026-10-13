# autodiff_vs_torch card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-13T17:34:46Z

## Facts

```json
{
  "device": {
    "name": "Quadro P2000"
  },
  "example_script": "examples/autodiff_vs_torch.py",
  "example_script_md5": "55c125efbca1039f9714a220c842f429",
  "generated_utc": "2026-10-13T17:34:46Z",
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
[
  {
    "batch": 1,
    "hawk_jvp": 1.5857000835239888e-05,
    "hawk_primal": 1.6574887558817865e-05,
    "hawk_vjp": 1.6012904234230517e-05,
    "ratio_jvp": 606.1159792262863,
    "ratio_vjp": 244.48074224753512,
    "torch_jvp": 0.009611181588843465,
    "torch_primal": 0.0007412115810438991,
    "torch_vjp": 0.003914846712723375
  },
  {
    "batch": 32,
    "hawk_jvp": 1.6442988999187948e-05,
    "hawk_primal": 1.6928208060562612e-05,
    "hawk_vjp": 1.7356895841658116e-05,
    "ratio_jvp": 655.4385511051072,
    "ratio_vjp": 213.4219804070704,
    "torch_jvp": 0.010777368885464966,
    "torch_primal": 0.0007960892980918288,
    "torch_vjp": 0.0037043430842459203
  },
  {
    "batch": 4096,
    "hawk_jvp": 1.5866989269852638e-05,
    "hawk_primal": 1.6774586401879788e-05,
    "hawk_vjp": 1.7629493959248067e-05,
    "ratio_jvp": 584.4593415526749,
    "ratio_vjp": 225.6910170751775,
    "torch_jvp": 0.009273610101081431,
    "torch_primal": 0.0007384498836472631,
    "torch_vjp": 0.003978818422183394
  },
  {
    "batch": 65536,
    "hawk_jvp": 4.414310678839684e-05,
    "hawk_primal": 3.801840357482433e-05,
    "hawk_vjp": 4.657821264117956e-05,
    "ratio_jvp": 198.3266794378701,
    "ratio_vjp": 78.27625781109802,
    "torch_jvp": 0.008754755789414048,
    "torch_primal": 0.0008264229167252779,
    "torch_vjp": 0.003645968181081116
  },
  {
    "batch": 1048576,
    "hawk_jvp": 0.0004156948998570442,
    "hawk_primal": 0.0002838781801983714,
    "hawk_vjp": 0.00043076497968286276,
    "ratio_jvp": 63.51657044151549,
    "ratio_vjp": 91.37475225275745,
    "torch_jvp": 0.02640351438894868,
    "torch_primal": 0.008961996901780367,
    "torch_vjp": 0.03936104329768568
  }
]
```
