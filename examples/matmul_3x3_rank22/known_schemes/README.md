# Known schemes (rank 22)

Empty by design. No verified rank-22 scheme for general 3x3 matrix multiplication
is known to exist, so there is nothing to screen against. See
`examples/matmul_3x3/known_schemes/` for the rank-23 reference set (Laderman,
Oh-Hopcroft-Kim), which is the best known upper bound.

If a rank-22 scheme is ever published or independently verified, add it here the
same way: a verified `.npz` with `U`, `V`, `W`, checked against the Brent
equations by `symmetry.load_known_schemes` before being trusted -- never
transcribed from memory.
