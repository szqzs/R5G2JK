# Results

This directory contains compact committed certificates.

For a human-readable summary of the results, start with
[`README.md`](../README.md).

## Full-Rank Results

`results/full_rank/full_rank_certificate.json` is the combined certificate for
the full-row-rank checks in Chern degrees

```math
11,13,14,15,16,17,18,19,20,21,22.
```

The omitted Chern degree is `c = 12`, which is handled separately because it
has a one-dimensional relation line.

For convenience, the same data is also split into one compact file per degree
under `results/full_rank/by_degree/`.

Each degree summary records:

- Chern degree;
- source row count;
- rank modulo the chosen prime;
- source nullity;
- prime;
- number of columns used;
- whether the calculation stopped early after reaching full row rank;
- checksum;
- command/provenance metadata.

Some degrees use only a subset of target columns.  This is intentional: once
the accumulated matrix has full source rank, extra target columns cannot reduce
the rank, so they are unnecessary for proving full row rank.

## The `c = 12` Relation

`results/c12_relation/c12_relation_certificate.json` contains:

- the relation vector;
- its verification metadata;
- rank/nullity information for the relevant matrix;
- checksums and provenance.

The certificate records rank `43` for a `44`-dimensional source space and a
displayed vector spanning the one-dimensional relation line modulo
`2305843009213693951`.

`results/c12_relation/c12_relation_over_q_certificate.json` contains the exact
rational check for the lifted integer relation.  It records that the integer
vector pairs to zero with all `1039` target basis elements over `Q`, and that
its reduction modulo `2305843009213693951` spans the same line as the modular
certificate.

## Manifest

`results/MANIFEST.json` records SHA-256 hashes for every committed result
certificate in this directory.
