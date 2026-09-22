#!/usr/bin/env python3
"""Generate a large random probable prime, using primesieve for the sieve table.

primesieve only sieves below 2^64, so it cannot look at a 1024-bit number
directly.  What it does supply -- very cheaply -- is the small-prime table used
to trial-divide the candidate interval, which throws out ~92% of the odd
candidates before a single expensive modexp runs.

    python tools/genprime.py 1024
    python tools/genprime.py 2048 --bound 1e7 --primesieve ../build/primesieve.exe

Both the decimal and the hex form are printed by default; --format dec/hex (or
--hex) narrows it to one bare line for piping.

    python tools/genprime.py 2048 -o key.bin

-o/--out also writes the prime out as raw big-endian bytes, plus a companion
.hex file holding the same bytes as zero-padded ASCII hex.  With -n the files
are numbered (key_1.bin, key_2.bin, ...).

The final test is Baillie-PSW (Miller-Rabin base 2 + strong Lucas) plus
--rounds random-base Miller-Rabin rounds.
"""

import argparse
import os
import secrets
import subprocess
import sys
import time


# --------------------------------------------------------------------------
# small-prime table, straight out of primesieve
# --------------------------------------------------------------------------

def find_primesieve(explicit):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    names = ['primesieve.exe', 'primesieve']
    for d in [os.path.join(root, 'build'), os.path.join(root, 'build-nogpu'), root]:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return 'primesieve'          # hope it is on PATH


def small_primes(exe, bound):
    """Run `primesieve 2 BOUND -p` and return the primes as a list of ints."""
    cmd = [exe, '2', repr(int(bound)), '-p', '-t', '1', '--no-status', '-q']
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit('could not run primesieve (%s): %s' % (exe, e))
    return [int(line) for line in out.split()]


# --------------------------------------------------------------------------
# interval sieve: cross off base + 2i that have a small factor
# --------------------------------------------------------------------------

def sieve_interval(base, n, primes):
    """Return a bytearray of length n; comp[i] set means base + 2i is composite."""
    comp = bytearray(n)
    for p in primes:
        if p == 2:
            continue
        r = base % p
        # base + 2i = 0 (mod p)  ->  i = -r * inv(2) (mod p), inv(2) = (p+1)/2
        i0 = 0 if r == 0 else (-r * ((p + 1) >> 1)) % p
        if i0 < n:
            comp[i0::p] = b'\x01' * ((n - i0 + p - 1) // p)
    return comp


# --------------------------------------------------------------------------
# primality: Baillie-PSW + random-base Miller-Rabin
# --------------------------------------------------------------------------

def miller_rabin(n, a):
    d = n - 1
    s = (d & -d).bit_length() - 1
    d >>= s
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def jacobi(a, n):
    a %= n
    t = 1
    while a:
        while not a & 1:
            a >>= 1
            if n & 7 in (3, 5):
                t = -t
        a, n = n, a
        if a & 3 == 3 and n & 3 == 3:
            t = -t
        a %= n
    return t if n == 1 else 0


def strong_lucas(n):
    """Strong Lucas probable-prime test, Selfridge parameters (D, P=1, Q)."""
    d = 5
    while True:
        j = jacobi(d, n)
        if j == -1:
            break
        if j == 0 and abs(d) != n:
            return False        # gcd(D, n) is a proper factor
        d = -(d + 2) if d > 0 else -(d - 2)
    q = (1 - d) // 4

    s = n + 1
    r = (s & -s).bit_length() - 1
    s >>= r

    u, v, qk = 1, 1, q
    inv2 = (n + 1) >> 1
    for bit in bin(s)[3:]:
        u, v = u * v % n, (v * v - 2 * qk) % n
        qk = qk * qk % n
        if bit == '1':
            u, v = (u + v) * inv2 % n, (d * u + v) * inv2 % n
            qk = qk * q % n
    if u == 0 or v == 0:
        return True
    for _ in range(r - 1):
        v = (v * v - 2 * qk) % n
        if v == 0:
            return True
        qk = qk * qk % n
    return False


def is_probable_prime(n, rounds=8):
    if not miller_rabin(n, 2):
        return False
    if not strong_lucas(n):                     # -> Baillie-PSW
        return False
    for _ in range(rounds):                     # belt and braces
        if not miller_rabin(n, secrets.randbelow(n - 3) + 2):
            return False
    return True


# --------------------------------------------------------------------------
# output files
# --------------------------------------------------------------------------

def out_names(path, index, count):
    """(binary, hex) filenames for prime `index` of `count`, derived from --out."""
    root, ext = os.path.splitext(path)
    if not ext:
        ext = '.bin'
    if count > 1:
        root = '%s_%d' % (root, index + 1)
    return root + ext, root + '.hex'


def write_prime(path, p, index, count):
    """Write p as raw big-endian bytes, and as ASCII hex next to it."""
    nbytes = (p.bit_length() + 7) // 8
    raw = p.to_bytes(nbytes, 'big')
    binname, hexname = out_names(path, index, count)
    with open(binname, 'wb') as fp:
        fp.write(raw)
    with open(hexname, 'w') as fp:
        fp.write('%0*x\n' % (2 * nbytes, p))
    return binname, hexname, nbytes


# --------------------------------------------------------------------------

def gen_prime(bits, primes, span, rounds, verbose=True):
    stats = {'intervals': 0, 'sieved': 0, 'survivors': 0, 'tested': 0}
    while True:
        # top two bits set: the result is always exactly `bits` long, and a
        # product of two such primes is exactly 2*bits long
        base = secrets.randbits(bits) | (3 << (bits - 2)) | 1
        comp = sieve_interval(base, span, primes)
        survivors = [i for i in range(span) if not comp[i]]
        stats['intervals'] += 1
        stats['sieved'] += span
        stats['survivors'] += len(survivors)
        for i in survivors:
            n = base + 2 * i
            stats['tested'] += 1
            if is_probable_prime(n, rounds):
                return n, stats
        if verbose:
            print('  interval %d: %d survivors, none prime'
                  % (stats['intervals'], len(survivors)), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bits', nargs='?', type=int, default=1024,
                    help='size of the prime in bits (default 1024)')
    ap.add_argument('-n', '--count', type=int, default=1,
                    help='how many primes to generate (default 1)')
    ap.add_argument('-b', '--bound', type=float, default=2e6,
                    help='trial-division bound, i.e. how far primesieve sieves '
                         '(default 2e6)')
    ap.add_argument('-s', '--span', type=int, default=1 << 17,
                    help='odd candidates per sieved interval (default 131072)')
    ap.add_argument('-r', '--rounds', type=int, default=8,
                    help='random-base Miller-Rabin rounds after Baillie-PSW '
                         '(default 8)')
    ap.add_argument('--primesieve', help='path to the primesieve binary')
    ap.add_argument('--format', choices=['both', 'dec', 'hex'], default='both',
                    help='print the prime in decimal, in hex, or both '
                         '(default both)')
    ap.add_argument('--hex', dest='format', action='store_const', const='hex',
                    help='shorthand for --format hex')
    ap.add_argument('-o', '--out', metavar='FILE',
                    help='also write the prime as raw big-endian bytes to FILE '
                         '(and the same bytes as ASCII hex to FILE.hex); with '
                         '-n the names are numbered FILE_1, FILE_2, ...')
    ap.add_argument('-q', '--quiet', action='store_true', help='print only the primes')
    args = ap.parse_args()

    if args.bits < 16:
        sys.exit('use at least 16 bits')

    exe = find_primesieve(args.primesieve)
    t0 = time.time()
    primes = small_primes(exe, args.bound)
    if not args.quiet:
        print('%d sieving primes up to %d from %s (%.2fs)'
              % (len(primes), primes[-1], exe, time.time() - t0), file=sys.stderr)

    span = min(args.span, 1 << max(1, args.bits - 4))
    for k in range(args.count):
        t0 = time.time()
        p, st = gen_prime(args.bits, primes, span, args.rounds,
                          verbose=not args.quiet)
        if not args.quiet:
            pct = 100.0 * st['survivors'] / st['sieved']
            print('[%d/%d] %d bits in %.2fs  (%d candidates sieved, %.2f%% '
                  'survived trial division, %d primality tests)'
                  % (k + 1, args.count, p.bit_length(), time.time() - t0,
                     st['sieved'], pct, st['tested']), file=sys.stderr)
        if args.out:
            binname, hexname, nbytes = write_prime(args.out, p, k, args.count)
            if not args.quiet:
                print('wrote %s (%d bytes) and %s'
                      % (binname, nbytes, hexname), file=sys.stderr)
        if args.format == 'dec':
            print(p)
        elif args.format == 'hex':
            print('%x' % p)
        elif args.quiet:                    # bare lines, decimal then hex
            print(p)
            print('%x' % p)
        else:
            print('dec: %d' % p)
            print('hex: %x' % p)


if __name__ == '__main__':
    main()
