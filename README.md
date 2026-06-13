# Ternary Gradient Queue — Priority Queue for Ternary Gradient Updates

**Ternary Gradient Queue** orders gradient updates by importance: parameters with large accumulated gradients update first, while low-signal parameters wait. It classifies each update into four priority levels (Low, Medium, High, Critical) based on the magnitude of the net ternary signal, and supports capacity-bounded eviction of low-priority updates.

## Why It Matters

In ternary neural networks, each weight has only three possible values {-1, 0, +1}, so individual gradient signals are weak. But accumulated gradients — the sum of many ternary votes across batches — carry strong directional information. Updating all parameters simultaneously wastes compute on parameters with weak gradient signal. This queue ensures that the most impactful parameters (highest accumulated gradient magnitude) are updated first, achieving faster convergence per actual update applied. This is particularly valuable in federated settings where update bandwidth is limited: the queue naturally prioritizes the updates that matter most.

## How It Works

### Priority Classification

Each gradient update carries a `net_signal` (sum of ternary gradients accumulated for that parameter). Priority is assigned by magnitude:

```
Critical:  |net_signal| ≥ threshold_high (default 20)
High:      |net_signal| ≥ threshold_high / 2
Medium:    |net_signal| ≥ threshold_low (default 5)
Low:       |net_signal| < threshold_low
```

Classification is O(1) per update.

### Queue Operations

- **Enqueue**: Add an update with computed priority. If the queue is at capacity, the lowest-priority existing update is compared to the new one; the lower of the two is dropped. O(n) worst case, O(1) amortized with a heap.
- **Dequeue**: Pop the highest-priority update. O(log n) with a binary heap.
- **Drain by priority**: Extract all updates at a given priority level. O(n).

### Gradient Accumulation

The queue tracks `accumulated_count` (how many gradient samples contributed) and `net_signal` (their signed sum). A parameter that received 30 +1 gradients and 10 -1 gradients has `accumulated_count = 40`, `net_signal = +20` → Critical priority.

### Statistics

The queue tracks `total_enqueued` and `total_processed` counters for throughput analysis. Eviction rates indicate whether the queue capacity is too small for the workload.

## Quick Start

```rust
use ternary_gradient_queue::{GradientQueue, Priority};

let mut queue = GradientQueue::new(1000); // capacity 1000

// Enqueue gradient updates
queue.enqueue(0, 1, 5, 15);   // param 0: +1 gradient, 5 samples, net +15 → High
queue.enqueue(1, -1, 20, -25); // param 1: -1 gradient, 20 samples, net -25 → Critical
queue.enqueue(2, 0, 3, 1);     // param 2: 0 gradient, 3 samples, net +1 → Low

// Process highest priority first
while let Some(update) = queue.pop() {
    println!("Param {}: priority={:?}, signal={}", update.param_idx, update.priority, update.net_signal);
    // Apply update...
}
```

```bash
cargo add ternary-gradient-queue
```

## API

| Type / Function | Description |
|---|---|
| `Priority` | `Low`, `Medium`, `High`, `Critical` with `from_magnitude()` |
| `GradientUpdate` | `{ param_idx, gradient, priority, accumulated_count, net_signal }` |
| `GradientQueue` | `new(capacity)`, `enqueue()`, `pop()`, `peek()`, `drain_high_priority()` |
| `Priority::from_magnitude(net, low, high)` | Classify by signal magnitude |

## Architecture Notes

The gradient queue optimizes update efficiency in **SuperInstance** fleet training. By processing the highest-magnitude updates first, the fleet converges faster per round of communication. The γ + η = C conservation law applies: each update has a γ benefit (improved model) and η cost (communication bandwidth), and the queue maximizes the γ/η ratio. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Hinton, Geoffrey. "Neural Networks for Machine Learning," *Coursera*, 2012 — gradient accumulation.
- Reddi, Sashank et al. "On the Convergence of Adam and Beyond," *ICLR*, 2018 — adaptive learning rates.
| Kingma, Diederik & Ba, Jimmy. "Adam: A Method for Stochastic Optimization," *ICLR*, 2015 — momentum-based prioritization.

## License

MIT
