"""ST1 gate (paper-2 Phase 0): adaptive structural detection and its null validity.

The make-or-break question (notes/plan-for-paper-2.md Sec. 6 ST1): can the null
distribution of the tracker -> phase-resampling -> cyclostationary pipeline be
derived, or defensibly calibrated, including its selection and covariance-
estimation steps? This package builds that pipeline in stages and measures its
realised false-alarm rate against a graded null suite.

Layout (per the approved Phase 0 plan)::

    nulls       registry of null-trace generators (stages 5-7 of the ST1 design)
    multitaper  Thomson DPSS harmonic F-test + harmonic-comb variant (task 19.2)
    cyclo       Dandawate-Giannakis cyclostationary Q statistic (task 19.3)
    resample    phase integration + angle-domain resampling (task 19.5)
    splitting   EST/TEST block splitting + held-out tracker (task 19.6)
    surrogates  Fourier-phase / block-permutation calibrators (task 19.8)
    pipeline    staged detectors under the shared gate signature (tasks 19.3+)
    far         realised-FAR harness with Clopper-Pearson CIs (task 19.4)

This package never mutates code/typeb/ — that is the frozen B0/B1/B2 track. It
reuses typeb's TypeTrace container and detector signature so ST1 detectors can
be scored by the existing gate machinery via explicitly merged registries.
"""
