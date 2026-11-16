# Benchmarks

- `padic_flow.py`: GPU throughput and set-up times of `galrax.padic.PAdicFlow` (`flow`, `rectify`, the full series); the settings are in the main README.
- `padic_jump.py`: set-up and query times of jump-ahead: `PAdicMap.iterate` on `Z_p` (scalar and batched), an `Orbit` on `Z_p^2`, and `matrix_power_index`.
