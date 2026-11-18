# Benchmarks

- `padic_flow.py`: GPU throughput and set-up times of `galrax.padic.PAdicFlow` (`flow`, `rectify`, the full series); the settings are in the main README.
- `padic_jump.py`: set-up and query times of jump-ahead: `PAdicMap.iterate` on `Z_p` (scalar and batched), an `Orbit` on `Z_p^2`, and `matrix_power_index`.
- `skolem.py`: zeros of linear recurrences on the dataset of the SKOLEM tool (downloaded separately), certified by Strassmann's bound over residue classes at several primes; the results are in the main README.
- `nonlinear_skolem.py`: the same certificate for orbits of random generalised Hénon maps of $\mathbb{Z}^2$.
- `adelic_sieve.py`: ABS, the sieve by branch type (étale: Strassmann; nilpotent and mixed: the exact orbit cycle modulo $p^k$; the real place: the Hénon filtration and an invariant cone for recurrences), against the étale sieve on recurrences over $\mathbb{Z}$, random quadratic maps of $\mathbb{Z}^2$, $\mathbb{Z}^3$ and Hénon maps; the results are in the main README.
- `henon_escape.py`: the archimedean baseline for the same orbits, by the escape filtration of Hénon maps; it decides all of them exactly in a few milliseconds.
