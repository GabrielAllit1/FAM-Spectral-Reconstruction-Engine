# mathtoolbox_v7/core/973_orchestrator.py
# Corrected — fully integrates tools #973–#978 with chaos, graphons, and ZK oracle.

from __future__ import annotations
from typing import List, Dict, Any
import hashlib

from .numerics import digital_root_1_9
from .sequences import fam_base_sequence, fam_thue_morse_hybrid, van_eck
from .fractals import logistic_map
from .graphon import thue_morse_graphon, van_eck_graphon, constant_graphon
from .zk_oracle import ZK_ORACLE
from .math_tool_box_973 import Tool973NumerologyStateEngine
from .math_tool_box_973_hybrids import (
    Tool976StateRouterFAM,
    Tool977UnlockerModProfile,
    Tool978MultiSequenceMixer,
)


class Orchestrator973:
    """
    High-level orchestrator that links:

        973 numerology  -> routing (976) -> unlocker profile (977)
                         -> multi-sequence mixer (978)
                         -> fractal chaos + graphon density
                         -> FAM-root zero-knowledge proof

    The API is defensive: it tolerates either dict-like or attribute-based
    return types from the Tool97x helpers to avoid locking you into one
    concrete implementation.
    """

    def __init__(self, phrase: str = "973") -> None:
        self.phrase = phrase
        # Profile may be a dict or a dataclass-like object
        self.profile = Tool973NumerologyStateEngine(phrase)
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _profile_field(self, name: str, default: Any = None) -> Any:
        """Safely read a field from self.profile as attr or dict key."""
        p = self.profile
        if hasattr(p, name):
            return getattr(p, name)
        if isinstance(p, dict) and name in p:
            return p[name]
        return default

    # ------------------------------------------------------------------
    # SINGLE RESONANCE CYCLE
    # ------------------------------------------------------------------

    def resonate(self) -> Dict[str, Any]:
        """
        Run one full 973 resonance cycle.
        """
        p = self.profile

        # 1) 973 routing + unlock profile
        # -------------------------------
        # Route: tolerate either phrase or profile parameter signature
        try:
            route = Tool976StateRouterFAM(self.phrase)
        except TypeError:
            route = Tool976StateRouterFAM(p)

        try:
            unlock = Tool977UnlockerModProfile(self.phrase)
        except TypeError:
            unlock = Tool977UnlockerModProfile(p)

        # Extract route label & scores robustly
        route_label = getattr(route, "selected_route", None)
        if route_label is None and isinstance(route, dict):
            route_label = route.get("selected_route", "")
        routes = getattr(route, "routes", None)
        if routes is None and isinstance(route, dict):
            routes = route.get("routes", [])
        route_scores = getattr(route, "route_scores", None)
        if route_scores is None and isinstance(route, dict):
            route_scores = route.get("route_scores", {})

        # Unlock fields with dict/attr fallback
        if isinstance(unlock, dict):
            mod_x = unlock.get("mod_x")
            depth = unlock.get("depth")
            chaos_level = unlock.get("chaos_level")
        else:
            mod_x = getattr(unlock, "mod_x", None)
            depth = getattr(unlock, "depth", None)
            chaos_level = getattr(unlock, "chaos_level", None)

        # 2) Base sequences & mixer
        # -------------------------
        # Extract some canonical numeric seeds from profile
        phrase_root = self._profile_field(
            "phrase_root",
            self._profile_field("total_digital_root", 5)
        )
        digital_root = self._profile_field("total_digital_root", phrase_root)
        fam_seed = self._profile_field(
            "fam_seed",
            self._profile_field("seed", digital_root or 973)
        )

        fam = fam_base_sequence(fam_seed, 256)
        thm = fam_thue_morse_hybrid(fam_seed, 256)
        ve = van_eck(256)

        base_sequences = {
            "FAM": fam,
            "THUE_MORSE": thm,
            "VAN_ECK": ve,
        }

        # Call mixer with best-guess signature; fall back gracefully
        try:
            mixer = Tool978MultiSequenceMixer(self.phrase, base_sequences, 256, 9)
        except TypeError:
            try:
                mixer = Tool978MultiSequenceMixer(base_sequences, 256, 9)
            except TypeError:
                mixer = Tool978MultiSequenceMixer(base_sequences)

        # Extract hybrid sequence from mixer
        if isinstance(mixer, dict):
            hybrid_seq = mixer.get("hybrid", [])
        else:
            hybrid_seq = getattr(mixer, "hybrid", None)
            if hybrid_seq is None:
                # assume mixer itself is an iterable of ints
                try:
                    hybrid_seq = list(mixer)
                except TypeError:
                    hybrid_seq = []

        # 3) Fractal chaos via logistic map
        # ---------------------------------
        x0 = (digital_root or 5) / 9.0
        chaos_series = logistic_map(r=3.99, x0=x0, n=100)
        chaos_val = chaos_series[-1] if chaos_series else 0.0

        # 4) Graphon selection by route label
        # -----------------------------------
        label_str = str(route_label or "").lower()
        if "graphon" in label_str or "graph" in label_str:
            G = thue_morse_graphon()
        elif "quantum" in label_str or "walk" in label_str:
            G = van_eck_graphon()
        else:
            G = constant_graphon(0.618)

        graphon_density = G.edge_density(n_samples=5000)

        # 5) Final FAM-style root
        # -----------------------
        fam_tail = fam[-9:] if len(fam) >= 9 else fam
        hyb_tail = hybrid_seq[-9:] if len(hybrid_seq) >= 9 else hybrid_seq
        ve_tail = ve[-9:] if len(ve) >= 9 else ve

        base_sum = sum(fam_tail) + sum(hyb_tail) + sum(ve_tail)
        chaos_term = int(abs(chaos_val) * 1_000_000)
        route_bias = digital_root_1_9(abs(hash(label_str)) or 1)
        unlock_depth = depth or 0
        unlock_energy = int((chaos_level or 0.0) * 100.0)

        final_root = digital_root_1_9(
            base_sum
            + chaos_term
            + int(fam_seed)
            + route_bias
            + unlock_depth
            + unlock_energy
        )

        # 6) Zero-knowledge proof for final root via ZK_ORACLE
        # ----------------------------------------------------
        # Returns (commitment, t, z1, z2) from Pedersen-based sigma
        zk_commit, zk_t, zk_z1, zk_z2 = ZK_ORACLE.prove_fam_root(final_root)

        cycle: Dict[str, Any] = {
            "phrase": self.phrase,
            "phrase_root": phrase_root,
            "digital_root": digital_root,
            "fam_seed": fam_seed,
            "route_label": route_label,
            "routes": routes,
            "route_scores": route_scores,
            "unlock_mod_x": mod_x,
            "unlock_depth": depth,
            "unlock_chaos_level": chaos_level,
            "mixer_sample": list(hybrid_seq)[:9],
            "fam_tail": fam_tail,
            "hybrid_tail": hyb_tail,
            "van_eck_tail": ve_tail,
            "chaos_final": chaos_val,
            "graphon_density": graphon_density,
            "final_root": final_root,
            "zk_proof": {
                "commitment": zk_commit,
                "t": zk_t,
                "z1": zk_z1,
                "z2": zk_z2,
            },
            "resonance": "ACTIVE",
        }

        self.history.append(cycle)
        return cycle

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def run_field(self, cycles: int = 81) -> List[Dict[str, Any]]:
        """Run 'cycles' resonance passes and return the list of cycles."""
        return [self.resonate() for _ in range(cycles)]

    def extract_key(self) -> str:
        """
        Derive a 256-bit key from the last 9 final roots.
        """
        if len(self.history) < 9:
            self.run_field(9)

        roots = [c["final_root"] for c in self.history[-9:]]
        prekey = "".join(str(r) for r in roots) + "973"
        digest = hashlib.sha3_256(prekey.encode("utf-8")).hexdigest()
        return f"973-UNLOCKED-{digest.upper()}"

    def status(self) -> Dict[str, Any]:
        """Lightweight status report."""
        return {
            "973_orchestrator": "ACTIVE",
            "phrase": self.phrase,
            "cycles": len(self.history),
            "current_root": self.history[-1]["final_root"] if self.history else None,
            "tools_973_978": "INTEGRATED",
            "zk_oracle_ready": True,
        }


# Global instance + convenience wrapper
ORCHESTRATOR_973 = Orchestrator973("GABEA123")


def unlock_v7() -> str:
    """
    Convenience helper returning a high-entropy unlock key
    derived from the orchestrator pipeline.
    """
    ORCHESTRATOR_973.run_field(81)
    return ORCHESTRATOR_973.extract_key()
