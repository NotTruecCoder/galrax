# Benchmarks

- `padic_flow.py`: GPU throughput and set-up times of `galrax.padic.PAdicFlow` (`flow`, `rectify`, the full series); the settings are in the main README.
- `padic_jump.py`: set-up and query times of jump-ahead: `PAdicMap.iterate` on `Z_p` (scalar and batched), an `Orbit` on `Z_p^2`, and `matrix_power_index`.
- `skolem.py`: zeros of linear recurrences on the dataset of the SKOLEM tool (downloaded separately), certified by Strassmann's bound over residue classes at several primes; the results are in the main README.
- `nonlinear_skolem.py`: the same certificate for orbits of random generalised Hénon maps of $\mathbb{Z}^2$.
- `adelic_sieve.py`: ABS (`galrax.padic.adelic`), the sieve by branch type (étale: Strassmann; nilpotent and mixed: the exact orbit cycle modulo $p^k$; the real place: the Hénon filtration; for recurrences a dominant root, a dominant complex pair by Baker–Davenport in `galrax/padic/baker.py`, or an invariant cone, in ball arithmetic through `python-flint`), against the étale sieve on recurrences over $\mathbb{Z}$, random quadratic maps of $\mathbb{Z}^2$, $\mathbb{Z}^3$ and Hénon maps; the results are in the main README.
- `oeis.py`: every $n$ with $a(n)=c$, $c\in\{0,\pm1,\pm2\}$, for OEIS sequences of "Index to OEIS: Section Rec" (downloaded once into a local cache), by `value_times`; each signature is checked on the given terms first. A sample of 60 sequences of order $\le 5$: 293 of 300 instances certified, the certified sets agree with every given term.
- `cm_skolem.py`: time in $\mathbb{Z}[i]$: all $n\in\mathbb{Z}[i]$ with $x([n]P_1)+x([n]P_2)=c$ on $E_5\times E_6$, $E_D: y^2=x^3-D^2x$ (CM by $\mathbb{Z}[i]$). At inert $p$ the function on each class $n_0+(p+1)\mathbb{Z}[i]$ is one power series over $\mathbb{Z}_{p^2}$, read off real times only, and Strassmann's bound counts its zeros; at split $p$ it depends on $a+b\iota$ and its zeros in $\mathbb{Z}_p^2$ are lines. 40 of 40 random targets (half with planted zeros $\pm n^*$) certified with 2–6 inert primes $\le 43$, 0.1–10 s each depending on load.
- `henon_escape.py`: the archimedean baseline for the same orbits, by the escape filtration of Hénon maps; it decides all of them exactly in a few milliseconds.

## The Skolem problem

`EventSet.classes` splits Strassmann's bound over the classes $n \equiv c \bmod N$ of an orbit; a class none of whose integers lies in the event set at another prime holds no integer zero. Summing the bounds of the classes that survive several primes, and comparing with the zeros found in exact arithmetic for $|n| \le$ a window, certifies the whole zero set over $\mathbb{Z}$, negative $n$ included, when the two agree. This is the classical Skolem–Mahler–Lech sieve; nothing in it is specific to linear maps.

**Linear recurrences** (`benchmarks/skolem.py`), on the dataset of the SKOLEM tool (Bilu, Luca, Nieuwveld, Ouaknine, Purser, Worrell, *Skolem meets Schanuel*, MFCS 2022): 82,360 random integer recurrences of order 2–10 with entries in $[-20, 20]$, the order uniform. A sample of 50 per order, 60 s per instance, 12 processes on a laptop. Up to eight primes $p < 200$ are tried, in order of $p$ times the period of $u_n$ modulo $p$ (at most 5000), since the cost of a prime grows with that product and varies by orders of magnitude between primes. SKOLEM's mean times are from its paper (60 s timeout, SageMath on a 16-thread server); the paper gives its success rate only for the whole dataset.

| order | galrax: certified | galrax: timed out | galrax: mean time of certified | SKOLEM: mean time of solved |
|---|---|---|---|---|
| 2 | 50 / 50 | 0 | 0.5 s | 0.03 s |
| 3 | 50 / 50 | 0 | 1.2 s | 0.05 s |
| 4 | 50 / 50 | 0 | 5.0 s | 0.19 s |
| 5 | 41 / 50 | 4 | 17 s | 4.8 s |
| 6 | 19 / 50 | 3 | 28 s | 9.0 s |
| 7 | 2 / 50 | 13 | 42 s | 11.7 s |
| 8 | 1 / 50 | 6 | 53 s | 8.1 s |
| 9–10 | 0 / 100 | 13 | – | 7.4 s, 5.7 s |
| **all** | **213 / 450 (47%)** | 39 | | **42,346 / 82,360 (51%)** |

So galrax certifies orders 2–4 completely but is 15–25 times slower than SKOLEM there, and is weaker from order 5 on. SKOLEM's timeouts pass 50% only from order 7, while for galrax most instances of order 7 and above end uncertified: their periods modulo small primes exceed the limit, or the primes that remain do not separate the classes. SKOLEM also adds Baker–Davenport bounds to the sieve, which galrax does not have. Of the examples in the SKOLEM paper, galrax certifies Berstel's sequence (zeros $0, 1, 4, 6, 13, 52$) and Examples 15 and 17 (zeros $2$ and $0, 1, 4$). It does not certify Example 16, where one $p$-adic zero survives all eight primes.

The parameters were tuned on the same sample:

| primes | order of primes | period limit | certified |
|---|---|---|---|
| 4 | $3, 5, 7, 11, \ldots$ | – | 127 |
| 4 | by $p \cdot$ period | 2000 | 171 |
| 8 | by $p \cdot$ period | 5000 | **213** |
| 8 | by $p \cdot$ period | 20000 | 198 |
| 8 | by cost per common factor of the periods | 5000 | 214 |

A larger limit lets in primes too expensive to finish within 60 s. Choosing primes whose periods share factors, which the sieve compares, gains nothing over the simpler order.

**Nonlinear maps** (`benchmarks/nonlinear_skolem.py`), where SKOLEM and other tools for linear recurrences do not apply: orbits of generalised Hénon maps $(x, y) \mapsto (y, \pm x + a_2 y^2 + a_1 y + a_0)$ of $\mathbb{Z}^2$ with $|a_i|, |x_0|, |y_0| \le 5$, and all $n \in \mathbb{Z}$ with $x_n = $ target. For half of the orbits the target is a value $x_m$ of the orbit with $|m| \le 3$; for the rest it is random in $[-20, 20]$. Primes are chosen as for linear recurrences, up to eight, with 60 s per orbit:

| target | orbits | periodic over $\mathbb{Z}$ | certified, with zeros | certified, never reached | not certified | timed out | mean time of certified |
|---|---|---|---|---|---|---|---|
| orbit value | 95 | 1 | 54 | 0 | 0 | 40 | 23 s |
| random | 105 | 0 | 9 | 72 | 0 | 24 | 20 s |
| **all** | **200** | **1** | **63** | **72** | **0** | **64** | **21 s** |

The periodic orbit is solved directly. Of the other 199, 135 are certified: for 63 the list of zeros is complete, and for 72 the target is never reached at any $n \in \mathbb{Z}$. With the fixed primes $3, 5, 7, 11$ the count was 123. No orbit finished without a certificate: every one that finished in time was settled.

These instances have a much simpler archimedean solution. Outside a bounded square the forward orbit of a Hénon map escapes with $|x_n|$ increasing, and so does the backward orbit (the filtration of Bedford–Smillie and Friedland–Milnor). An exact integer computation therefore decides each instance after an explicit number of steps, either reaching a cycle or leaving the square: `benchmarks/henon_escape.py`, with the square $|x|, |y| < |a_1| + |a_0| + 3$, settles all 200 orbits, the 64 timeouts included, in 5.5 ms (1 periodic, 107 with zeros, 92 never reached), in agreement with the exact zeros for $|n| \le 14$. The benchmark is thus a test of the $p$-adic certificate on problems whose answer is known by other means, not a case where it is the only method. The certificate itself uses no archimedean information, so it also applies where no such filtration exists, for example to orbits in $\mathbb{Z}_p^2$.

**Adelic sieve by branch type** (`benchmarks/adelic_sieve.py`). The sieve above needs an étale prime, one where $Dg^L$ is invertible modulo $p$ on the cycle. ABS uses every place:

- At an étale prime it uses Strassmann's bound, as above.
- At a singular prime (nilpotent or mixed in the Fitting decomposition) it reads the exact orbit modulo $p^k$, $k = 1, 2, \ldots$: a tail $T$ and a cycle. When no point of the cycle hits the target, every zero lies below $T$. On a nilpotent branch the orbit tends to an attracting cycle, so this happens once $k$ exceeds the valuation of the target there. The bound comes from the computed cycle, so it holds on any branch without constants.
- The same cycle is computed at étale primes $p < 50$. There it gives the exclusions of `Orbit.times` at a fraction of the cost.
- At the real place: the filtration for Hénon maps. For recurrences, in ball arithmetic (Arb, through `python-flint`): a dominant root $\lambda$ with $u_n = c\lambda^n + \ldots$, $c \ne 0$, gives an explicit $N$ past which $u_n \ne 0$; a dominant complex pair gives one through Matveev's bound and the Baker–Davenport reduction (`galrax/padic/baker.py`, with Matveev's constant raised $10^6$ times as a margin, which moves the final bound only through $\log N_0$). Failing both, an invariant cone $|u_{m+1}| \ge \theta |u_m|$, which exists when $\theta + \sum_{i \ge 1} |r_i| \theta^{-i} \le |r_0|$; $u$ never vanishes inside it.

For $n < 0$, a recurrence is run as the integer recurrence $s_m = r_{k-1}^m u_{-m}$. Records at primes where the map is invertible modulo $p$ hold both ways. The others, and the real place, hold on one side, and each side is settled on its own. Zero sets are certified over $\mathbb{Z}$ for recurrences and Hénon maps, and for $n \ge 0$ for quadratic maps, which are not invertible.

The data:

- Recurrences: 20 per order 2–10, drawn from the distribution of the SKOLEM data set (entries in $[-20, 20]$).
- Random quadratic maps of $\mathbb{Z}^2$ and $\mathbb{Z}^3$: 100 each, from the generator of the étale-prime survey, including the 6 and 4 orbits with no étale prime below 200 and 100.
- The 200 Hénon orbits above.

All runs use 12 processes on a laptop, with 30 s per instance (60 s for Hénon). Up to eight Strassmann primes are used, as above. Every instance certified by both methods has the same zeros; no instance is certified by the sieve alone.

| data | instances | sieve: certified | ABS: certified | timed out, sieve / ABS | mean time of certified, sieve / ABS | ABS settled by |
|---|---|---|---|---|---|---|
| recurrences, order 2–4 | 60 | 60 | 60 | 0 / 0 | 2.3 s / 0.03 s | real place on a side 58, étale modulo $p^k$ 3 |
| recurrences, order 5–6 | 40 | 16 | 40 | 22 / 0 | 13 s / 0.09 s | real place 40 |
| recurrences, order 7–10 | 80 | 0 | 80 | 37 / 0 | – / 0.07 s | real place 80 |
| quadratic, $d = 2$ | 100 | 82 | 100 | 9 / 0 | 5.1 s / 0.2 s | nilpotent 42, mixed 41, étale modulo $p^k$ 16, Strassmann 1 |
| quadratic, $d = 3$ | 100 | 1 | 100 | 96 / 0 | 30 s / 3.0 s | mixed 60, nilpotent 25, étale modulo $p^k$ 14, Strassmann 1 |
| Hénon | 200 | 131 | 200 | 69 / 0 | 21 s / < 1 ms | real place 200 (1 periodic, reported as its zeros on one period and the period) |
| **all** | **580** | **290** | **580** | 233 / 0 | | |

The quadratic rows of ABS are a later run of ABS alone: Strassmann records are first computed to 4 digits (the bound is sound at any precision, only weaker, and `Orbit` grows fast with it), and they are queued after depth records of up to 30 times their cost. Before, 3 and 7 of the 100 orbits timed out, all while a Strassmann record was being computed that a cheap depth record made unnecessary. The 900 SKOLEM instances give the same 899 either way.

In "settled by", a recurrence counts once for each kind of record that settles one of its two sides.

On quadratic maps of $\mathbb{Z}^3$ the sieve fails: a cycle of length $L$ is étale with probability about $(1 - 1/p)^L$, and most cycles are mixed. Those mixed cycles are what ABS uses. The ten orbits with no étale prime are all certified, in 0.2 s and 1.7 s on average for $d = 2$ and $d = 3$. For recurrences the real place now settles almost every instance on both sides; the $p$-adic records are left to those whose dominant root has coefficient $0$ or whose dominant roots are not a single root or a single complex pair.

The sieve's counts depend on load. It ran alongside ABS, so the Hénon count is 131 here against 135 alone at the same timeout.

**On the SKOLEM data set itself** (`--dataset mfcs.tpls`), with the real place built up step by step on the same 180 instances (20 per order, seed 0, 30 s each):

| real place | certified | timed out | mean time of certified |
|---|---|---|---|
| invariant cone | 120 | 35 | 2.3 s |
| + dominant root | 149 | 19 | 1.2 s |
| + dominant complex pair (Baker–Davenport) | **180** | 0 | 0.02 s |

On 900 other instances (100 per order, seed 1) ABS certifies 899; one of order 9 times out. Every certified zero set was checked again, independently, against the exact zeros up to $|n| = N + 2000$, $N$ the larger of the two real-place bounds (up to 76,509): no difference.

**Against SKOLEM on the same instances.** SKOLEM (skolem-solver at commit `9379db1`, SageMath 10.7 under WSL2) ran with each of its two algorithms, Baker–Davenport over $\mathbb{Z}$ (`-B -bi`) and Leapfrogging (`-L`); ABS ran on Windows. Both used 4 processes on the same laptop, 30 s per instance, and the time is that of the computation in a running process, without start-up:

| order | ABS | SKOLEM `-B -bi` | SKOLEM `-L` |
|---|---|---|---|
| 2 | 100 (0.07 s) | 93 (0.30 s) | 93 (0.20 s) |
| 3 | 100 (0.01 s) | 99 (0.39 s) | 99 (0.29 s) |
| 4 | 100 (0.02 s) | 100 (0.41 s) | 99 (0.29 s) |
| 5 | 100 (0.03 s) | 100 (0.45 s) | 77 (3.3 s) |
| 6 | 100 (0.06 s) | 100 (0.49 s) | 32 (5.2 s) |
| 7 | 100 (0.12 s) | 99 (0.72 s) | 3 (8.8 s) |
| 8 | 100 (0.22 s) | 100 (0.69 s) | 1 (1.1 s) |
| 9 | 99 (0.28 s) | 99 (0.95 s) | 0 |
| 10 | 100 (0.51 s) | 100 (1.13 s) | 0 |
| **all** | **899 (0.15 s)** | **890 (0.62 s)** | **404 (1.3 s)** |

Certified instances out of 100 per order, with the mean time of those. SKOLEM's ten misses with `-B` are nine degenerate recurrences, which it rejects, and one timeout; ABS certifies those nine, whose zero sets are finite. Leapfrogging certifies 45%, close to the 51% of its paper, so that figure is for `-L`. On the 180 instances of seed 0 the counts are 180, 179 and 83. No tool gave a wrong zero set. The margin of ABS is thus small in coverage (the degenerate cases) and in time a factor of about 4 overall, 2–40 by order; both tools rest on the same Baker–Davenport argument for recurrences, and ABS's own contribution is for the maps where no such argument exists.
