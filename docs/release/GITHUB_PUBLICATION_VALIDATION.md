# Publication tooling validation

The existing addon source trees and publication addon were merged in a separate
authoring QA directory. The original DemandCast core checkout, actual final
registration/results and user's Git history were not present or substituted.

The combined suite passed **49 tests**: the earlier 43 addon checks and six
publication checks. These cover file selection excluding raw/prepared M5 data,
runtime predictions and local environments; replacing the pending final README
section while retaining development evidence; adding the plotting dependency
to the copied CI command; rejecting changed forecast code/settings; validating
an explicit completed-result receipt fixture and rejecting an altered artifact;
and rendering both regular and zero-sales chart states.

The Git test performs an actual local commit and clone of a real bundled E02
receipt with `core.autocrlf=true`. The SHA-256 is unchanged when the generated
`* -text` policy is applied. It does not modify global Git identity/settings or
create any remote repository.

A separate chart template rendered from synthetic values was visually inspected
for labels and clipping. That image is an internal layout check; it is not an
M5 result and is not shipped in this addon. The actual publication chart will
be generated from the verified completed-run tables on the user's computer.

Runtime: Linux/Python 3.12.14; NumPy 2.4.2, pandas 3.0.0, LightGBM 4.7.0,
scikit-learn 1.8.0, Streamlit 1.65.0, Plotly 6.9.0, pytest 8.4.2 and
Matplotlib 3.10.8. The actual core/export integration, Windows/Python 3.13 pass
and remote GitHub Actions result remain to be checked on the real checkout.
The complete project test count may exceed 49.

The GitHub connector returned the linked login and no matching DemandCast
repository within the accessible installation search. This is not a claim that
every repository on the account was enumerated. No remote repository, commit,
push, PR, release or tag was created during preparation.
