# Canonical GraphCast mesh assets

`google_mesh6_vertices.npy` stores the canonical float32 vertices for the
GraphCast mesh hierarchy through mesh size 6. Every coarser level is a prefix
of this array; face indices continue to come from the pinned DeepMind-derived
`build_mesh.py` implementation.

The asset was imported unchanged from colleague source commit
`b7667b42da708db86010fe2d962e5fd0363ffa0a`. It was originally materialized
from commit `e86b19cbdeefd61c71bea3d89c2a2a4e966f61a0` in the H100 baseline
environment (NumPy 1.26.4, SciPy 1.16.3). Its raw vertex tensor has shape
`(40962, 3)`, dtype `float32`, and raw-byte SHA256
`0c6a10735f0f73a4478d4c4665b22f7f556be2cb59b0201d644e757e69fdcb5d`.
The complete NPY file SHA256 is
`aa2462b34f8811cf11006eea1462ed558652cc7c4ce9b3f62ca4786c8814e40a`.

The asset prevents host BLAS reduction order from changing mesh vertices and,
consequently, the grid-to-mesh edge set. It does not contain parameters,
training data, outputs, or platform-specific code.
