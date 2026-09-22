#!/usr/bin/env python3
"""Bulk-generate primes with genprime, into a sharded directory tree.

One prime = one .bin (raw big-endian), named by a global index and filed in a
shard of 1000:

    primes/000/p_000000.bin
    ...
    primes/499/p_499999.bin

--hex additionally writes the same bytes as ASCII hex next to each .bin.

Run several workers over the same tree; worker w of n takes the indices with
i % n == w, so they never touch the same file.  Re-running skips indices that
are already on disk, so an interrupted run just resumes.

    python tools/genprimes_batch.py primes 500000 --bits 1024 --worker 0 --workers 2
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genprime


SHARD = 1000


def paths_for(root, i):
    d = os.path.join(root, '%03d' % (i // SHARD))
    stem = os.path.join(d, 'p_%06d' % i)
    return d, stem + '.bin', stem + '.hex'


def existing(root, total, want_hex, nbytes):
    """Indices already written, found with one scandir per shard.

    The .bin is the record of truth; a .hex is only required when --hex asked
    for one, so a run started without --hex resumes cleanly.  A .bin of the
    wrong length is a prime that was interrupted mid-write, so it does not
    count as done and gets regenerated.
    """
    done = set()
    nshards = (total + SHARD - 1) // SHARD
    for s in range(nshards):
        d = os.path.join(root, '%03d' % s)
        try:
            entries = os.scandir(d)
        except OSError:
            continue
        bins, hexes = set(), set()
        with entries as it:
            for e in it:
                name = e.name
                if name.startswith('p_') and len(name) == 12:
                    if name.endswith('.bin'):
                        if e.stat().st_size == nbytes:
                            bins.add(name[2:8])
                    elif name.endswith('.hex'):
                        hexes.add(name[2:8])
        for stem in (bins & hexes if want_hex else bins):
            done.add(int(stem))
    return done


def write_one(root, i, p, nbytes, want_hex):
    d, binname, hexname = paths_for(root, i)
    with open(binname, 'wb') as fp:
        fp.write(p.to_bytes(nbytes, 'big'))
    if want_hex:
        with open(hexname, 'w') as fp:
            fp.write('%0*x\n' % (2 * nbytes, p))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='output directory (created if missing)')
    ap.add_argument('total', type=int, help='how many primes in total')
    ap.add_argument('--bits', type=int, default=1024)
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('-b', '--bound', type=float, default=1e5)
    ap.add_argument('-s', '--span', type=int, default=16384)
    ap.add_argument('-r', '--rounds', type=int, default=8)
    ap.add_argument('--primesieve')
    ap.add_argument('--hex', action='store_true',
                    help='also write a .hex next to each .bin (off by default)')
    ap.add_argument('--every', type=int, default=250,
                    help='progress line every N primes (default 250)')
    args = ap.parse_args()

    nbytes = (args.bits + 7) // 8
    root = args.root

    # the tree is made up front by the launcher, but be standalone-safe
    for s in range((args.total + SHARD - 1) // SHARD):
        os.makedirs(os.path.join(root, '%03d' % s), exist_ok=True)

    exe = genprime.find_primesieve(args.primesieve)
    primes = genprime.small_primes(exe, args.bound)

    t0 = time.time()
    done = existing(root, args.total, args.hex, nbytes)
    mine = [i for i in range(args.total)
            if i % args.workers == args.worker and i not in done]
    print('worker %d/%d: %d sieving primes, %d already on disk, %d to do (scan %.1fs)'
          % (args.worker, args.workers, len(primes), len(done), len(mine),
             time.time() - t0), flush=True)

    t0 = time.time()
    for k, i in enumerate(mine, 1):
        p, _ = genprime.gen_prime(args.bits, primes, args.span, args.rounds,
                                  verbose=False)
        if p.bit_length() != args.bits:
            sys.exit('worker %d: got %d bits, expected %d'
                     % (args.worker, p.bit_length(), args.bits))
        write_one(root, i, p, nbytes, args.hex)
        if k % args.every == 0 or k == len(mine):
            el = time.time() - t0
            rate = k / el
            eta = (len(mine) - k) / rate
            print('worker %d: %d/%d  %.2f p/s  elapsed %.0fs  eta %.0fm'
                  % (args.worker, k, len(mine), rate, el, eta / 60), flush=True)

    print('worker %d: done, %d written in %.0fs'
          % (args.worker, len(mine), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
