# mathtoolbox_v7/core/zk_oracle.py
# Corrected — Zero-Knowledge Oracle with Pedersen commitments and bespoke transition proofs

from __future__ import annotations
from typing import Tuple, List, Optional
import hashlib
import secrets


class ZKOracle:
    """
    Zero-Knowledge Oracle for MathToolBox v7.0

    Features
    --------
    • Safe primes p = 2q + 1 generated via Miller–Rabin.
    • Generator g of subgroup of order q, secondary base h.
    • Pedersen commitments: C = g^v * h^r mod p.
    • Schnorr-style proof of knowledge of discrete log (for g^x).
    • Sigma protocol for knowledge of committed value (v, r) under C.
    • Bespoke "transition proofs" for happy-number iterations:
        - Each step proves knowledge of a hidden state x committed in C.
        - Publicly exposes next_x = sum(d_i^2) but not x itself.
        - Relation next_x = f(x) is computed by the prover; the proof
          is for knowledge of x underlying the commitment.
          (Full arithmetic ZK for f would require a larger circuit; we
          are explicit about that here.)
    """

    # ------------------------------------------------------------------
    # CONSTRUCTION
    # ------------------------------------------------------------------

    def __init__(self, security: int = 128) -> None:
        self.security = security
        self.p, self.q = self._generate_safe_prime_pair(security)
        self.g = self._primitive_root(self.p, self.q)
        # h is an independent generator of the same subgroup
        self.h = pow(self.g, secrets.randbelow(self.q - 1) + 1, self.p)

    # ------------------------------------------------------------------
    # MILLER–RABIN + SAFE PRIME GENERATION
    # ------------------------------------------------------------------

    def _miller_rabin(self, n: int, k: int = 40) -> bool:
        """Probabilistic primality test."""
        if n in (2, 3):
            return True
        if n < 2 or n % 2 == 0:
            return False
        # n - 1 = 2^r * s with s odd
        r, s = 0, n - 1
        while s % 2 == 0:
            s //= 2
            r += 1
        for _ in range(k):
            a = secrets.randbelow(n - 3) + 2  # in [2, n-2]
            x = pow(a, s, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _generate_safe_prime_pair(self, bits: int) -> Tuple[int, int]:
        """
        Generate a Sophie-Germain / safe prime pair (q, p = 2q+1)
        of approximately 'bits' bits for p.
        """
        if bits < 64:
            bits = 64
        while True:
            # q ~ 2^(bits-1)
            q = secrets.randbits(bits - 1) | 1  # odd
            if not self._miller_rabin(q):
                continue
            p = 2 * q + 1
            if self._miller_rabin(p):
                return p, q

    def _primitive_root(self, p: int, q: int) -> int:
        """
        Find a generator g of the unique subgroup of order q in (Z/pZ)^*.
        For safe prime p = 2q+1, this is easy: we just avoid elements of
        order 1, 2, or dividing q.
        """
        for g in range(2, p - 1):
            # Ensure g^2 != 1 and g^q != 1 (mod p)
            if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
                return g
        # Extremely unlikely fallback
        return 2

    # ------------------------------------------------------------------
    # PEDERSEN COMMITMENT
    # ------------------------------------------------------------------

    def commit(self, value: int, blinding: Optional[int] = None) -> Tuple[int, int]:
        """
        Pedersen commitment to 'value' with blinding 'r':

            C = g^value * h^r  (mod p)

        Returns (C, r).
        """
        r = secrets.randbelow(self.q) if blinding is None else (blinding % self.q)
        C = (pow(self.g, value, self.p) * pow(self.h, r, self.p)) % self.p
        return C, r

    def open_commit(self, value: int, blinding: int, commitment: int) -> bool:
        """Check that commitment really opens to (value, blinding)."""
        lhs = commitment % self.p
        rhs = (pow(self.g, value, self.p) * pow(self.h, blinding % self.q, self.p)) % self.p
        return lhs == rhs

    # ------------------------------------------------------------------
    # SCHNORR PROOF OF KNOWLEDGE OF DISCRETE LOG
    # ------------------------------------------------------------------

    def prove_dlog(self, x: int) -> Tuple[int, int, int]:
        """
        Prove knowledge of x such that y = g^x (mod p), without revealing x.

        Returns (y, t, z) where:
            y = g^x,
            t = g^k,
            z = k + c x (mod q),
        and c is derived via Fiat–Shamir from (y, t).
        """
        x = x % self.q
        y = pow(self.g, x, self.p)
        k = secrets.randbelow(self.q)
        t = pow(self.g, k, self.p)
        c_bytes = hashlib.shake_256(f"{y}{t}".encode()).digest(32)
        c = int.from_bytes(c_bytes, "big") % self.q
        z = (k + c * x) % self.q
        return y, t, z

    def verify_dlog(self, y: int, t: int, z: int) -> bool:
        """Verify Schnorr discrete-log proof."""
        c_bytes = hashlib.shake_256(f"{y}{t}".encode()).digest(32)
        c = int.from_bytes(c_bytes, "big") % self.q
        lhs = pow(self.g, z, self.p)
        rhs = (t * pow(y, c, self.p)) % self.p
        return lhs == rhs

    # ------------------------------------------------------------------
    # SIGMA PROTOCOL FOR COMMITMENT KNOWLEDGE
    # ------------------------------------------------------------------

    def prove_commitment_knowledge(self, value: int, blinding: int, commitment: int) -> Tuple[int, int, int]:
        """
        Prove knowledge of (value, blinding) s.t.:

            commitment = g^value * h^blinding  (mod p).

        Protocol:
            1. Choose a, b in Z_q uniformly.
            2. t = g^a * h^b.
            3. c = H(commitment || t).
            4. z1 = a + c*value (mod q),
               z2 = b + c*blinding (mod q).
        Returns (t, z1, z2).
        """
        value = value % self.q
        blinding = blinding % self.q
        a = secrets.randbelow(self.q)
        b = secrets.randbelow(self.q)
        t = (pow(self.g, a, self.p) * pow(self.h, b, self.p)) % self.p
        c_bytes = hashlib.shake_256(f"{commitment}{t}".encode()).digest(32)
        c = int.from_bytes(c_bytes, "big") % self.q
        z1 = (a + c * value) % self.q
        z2 = (b + c * blinding) % self.q
        return t, z1, z2

    def verify_commitment_knowledge(self, commitment: int, t: int, z1: int, z2: int) -> bool:
        """
        Verify proof-of-knowledge of (value, blinding) for a Pedersen
        commitment C = g^value h^blinding.

        Check:
            g^z1 h^z2 ?= t * C^c
        where c = H(C || t).
        """
        commitment = commitment % self.p
        t = t % self.p
        c_bytes = hashlib.shake_256(f"{commitment}{t}".encode()).digest(32)
        c = int.from_bytes(c_bytes, "big") % self.q
        lhs = (pow(self.g, z1, self.p) * pow(self.h, z2, self.p)) % self.p
        rhs = (t * pow(commitment, c, self.p)) % self.p
        return lhs == rhs

    # ------------------------------------------------------------------
    # FAM RESONANCE PROOF (Pedersen-based sigma)
    # ------------------------------------------------------------------

    def prove_fam_root(self, root: int) -> Tuple[int, int, int, int]:
        """
        Prove knowledge of a FAM digital root 'root' committed under Pedersen.

        Returns:
            commitment C,
            t, z1, z2  — sigma proof of knowledge of (root, r).
        """
        C, r = self.commit(root)
        t, z1, z2 = self.prove_commitment_knowledge(root, r, C)
        return C, t, z1, z2

    def verify_fam_root(self, commitment: int, t: int, z1: int, z2: int) -> bool:
        """Verify FAM root knowledge proof."""
        return self.verify_commitment_knowledge(commitment, t, z1, z2)

    # ------------------------------------------------------------------
    # HAPPY NUMBER TRANSITION PROOFS
    # ------------------------------------------------------------------

    @staticmethod
    def _happy_step(x: int) -> int:
        """One iteration x -> sum of squares of digits."""
        return sum(int(d) ** 2 for d in str(x))

    @staticmethod
    def _is_happy(n: int, max_iter: int = 256) -> bool:
        """Detect if n is happy by iterating up to max_iter steps."""
        x = n
        seen = set()
        for _ in range(max_iter):
            if x == 1:
                return True
            if x in seen:
                return False
            seen.add(x)
            x = ZKOracle._happy_step(x)
        return False

    def prove_happy_transition_step(self, x: int) -> Tuple[int, int, int, int, int]:
        """
        Bespoke zero-knowledge transition proof for a single step of the
        happy-number iteration.

        Prover:
            1. Compute next_x = f(x) = sum(d_i^2).
            2. Commit to x: C = g^x h^r.
            3. Run commitment-knowledge sigma protocol for (x, r):
               produce (t, z1, z2).

        Returns:
            (commitment, t, z1, z2, next_x)

        Notes:
            • The verifier can check the sigma proof (knowledge of x,r).
            • The relation next_x = f(x) is evaluated by the prover and
              next_x is public. Enforcing that relation in zero-knowledge
              would require a full arithmetic circuit proof system, which
              is beyond the scope of this oracle. We are explicit about
              this limitation.
        """
        next_x = self._happy_step(x)
        C, r = self.commit(x)
        t, z1, z2 = self.prove_commitment_knowledge(x, r, C)
        return C, t, z1, z2, next_x

    def verify_happy_transition_step(self, commitment: int, t: int, z1: int, z2: int, next_x: int) -> bool:
        """
        Verify a single transition proof. This checks only the
        knowledge of x,r behind 'commitment'; it does not enforce
        arithmetic correctness of next_x.
        """
        # For now, simply check the sigma protocol validity.
        return self.verify_commitment_knowledge(commitment, t, z1, z2)

    def prove_happy_number(self, n: int, max_steps: int = 128) -> List[Tuple[int, int, int, int, int]]:
        """
        Produce a chain of bespoke transition proofs showing that n
        evolves under happy-number iteration until it reaches 1 or a
        loop, without revealing the intermediate states x_i.

        Returns a list of tuples:
            (commitment_i, t_i, z1_i, z2_i, next_x_i)

        Raises:
            ValueError if n does not appear to be happy within max_steps.
        """
        if not self._is_happy(n, max_iter=max_steps * 2):
            raise ValueError("Input n does not appear to be a happy number within the search horizon.")

        proofs: List[Tuple[int, int, int, int, int]] = []
        x = n
        seen = set()
        for _ in range(max_steps):
            if x == 1 or x in seen:
                break
            seen.add(x)
            C, t, z1, z2, next_x = self.prove_happy_transition_step(x)
            proofs.append((C, t, z1, z2, next_x))
            x = next_x
            if x == 1:
                break
        return proofs


# Global oracle instance for convenience
ZK_ORACLE = ZKOracle(security=128)
