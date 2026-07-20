# Random Number Generation

AI Factory result databases treat the random-number generator as part of the
reproducibility contract. There are two active RNG roles.

The Python benchmark backend is `pytorch_randn`:

- normal generator: `torch.randn`
- generator policy: native batched PyTorch RNG
- seed scope: result row seeds are retained in JSON for schema parity, but
  Python validation engines do not use them as a pathwise contract
- device policy: same vectorized code on CPU or CUDA

This path is meant as a readable PyTorch CPU/GPU benchmark path and should be
written in vectorized PyTorch style. It is not a bitwise-stable cross-version
RNG contract. Python CPU and Python GPU results are expected to be
statistically close, not pathwise identical.

The C++ reproducible backend is `philox4x32_10_box_muller`:

- uniform generator: Philox-4x32-10
- normal transform: Box-Muller
- counter layout: `[block_low32, block_high32, stream_low32, stream_high32]`
- seed layout: low 32 bits and high 32 bits form the two-word Philox key

C++ CPU and C++ CUDA result databases using the same row id, seed, model
parameters, product parameters, and time grid must match to floating-point
precision.

Reference paper:

Salmon, John K., Mark A. Moraes, Ron O. Dror, and David E. Shaw. "Parallel
random numbers: as easy as 1, 2, 3." Proceedings of 2011 International
Conference for High Performance Computing, Networking, Storage and Analysis.
2011.

Reference implementation:

https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h

The C++ implementation and the archived Python Philox helper intentionally
share the same counter layout and floating-point conversion:

```text
uniform = (uint32_value + 0.5) / 2^32
```

Changing the uniform generator should be done behind the RNG module boundary.
The Python path can stay PyTorch-native while future C/CUDA kernels use Philox
or another counter-based generator.
