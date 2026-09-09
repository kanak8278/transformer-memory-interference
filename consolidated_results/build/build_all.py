"""Rebuild every consolidated CSV from the canonical sources.

    cd consolidated_results/build && python3 build_all.py

Each theme builder prints its row counts, provenance-tier split, and the result
of its cross-check against an independent restatement of the same numbers
(the committed paper .tex tables, or the *_comparison.txt files). A non-zero
mismatch count means a source moved underneath us — investigate before using
the output.

LaTeX back-extraction helpers live in common.py (strip_tex / tex_body_rows /
nums), shared by the theme builders that need them.
"""

import build_01_fvq_cvq
import build_02_ivq
import build_03_formats
import build_04_lora
import build_05_mechanistic
import build_05b_entropy_lens
import build_05c_value_identity
import build_06_from_scratch
import build_07_cot_ivq

if __name__ == "__main__":
    for mod in (build_01_fvq_cvq, build_02_ivq, build_03_formats, build_04_lora,
                build_05_mechanistic, build_05b_entropy_lens,
                build_05c_value_identity,
                build_06_from_scratch, build_07_cot_ivq):
        mod.main()
        print()
