# ternary-gradient-queue

*Not every parameter deserves equal attention. Update the ones that matter most, first.*

---

Priority queue for ternary gradient updates. In large ternary networks, not all parameters need updating at every step. This crate orders gradient updates by accumulated signal strength — parameters with large net gradients (strong directional signal) get updated before parameters with weak or conflicting gradients.

Implements: Priority levels (Low/Medium/High/Critical) based on gradient magnitude, a bounded GradientQueue with capacity eviction (drops lowest-priority when full), drain-ordered dequeue, a GradientScheduler with per-step budget and per-parameter deduplication, and priority distribution tracking.

The insight: in ternary training, most parameters have zero gradient most of the time. Spending compute on the few parameters with strong gradients first is more efficient than processing all parameters equally.

8 tests covering priority classification, enqueue/dequeue ordering, capacity eviction, distribution, budget enforcement, deduplication, and tracking.

Part of [SuperInstance](https://github.com/SuperInstance/SuperInstance).

License: MIT
