# ternary-gradient-queue

*Priority queue for ternary gradients. Not all parameters deserve equal attention — update the important ones first.*

## Why This Exists

In distributed ternary training, not every gradient update matters equally. A weight whose gradient has been accumulating for 100 steps needs attention more than one that was just updated. Standard training treats all parameters the same — every step updates everything.

This crate implements priority-based gradient scheduling: parameters with large accumulated gradients or high importance scores get updated first. In resource-constrained settings (edge training, federated learning), this means you spend your compute budget where it matters.

## Architecture

```
Gradient Stream: [Δw₁, Δw₂, Δw₃, ...]
       ↓ priority classification
Priority::High  → update immediately
Priority::Medium → batch with others
Priority::Low   → defer to next cycle
```

### Key Types

- **`Priority`** — High / Medium / Low classification based on gradient magnitude and age
- **`GradientUpdate`** — A single parameter update with priority, parameter index, and ternary gradient value
- **`GradientQueue`** — Priority queue that sorts updates by importance. Drains high-priority first.
- **`GradientScheduler`** — Manages the queue across training steps, applies age-based priority escalation (gradients that have been waiting get promoted)

## Usage

```rust
use ternary_gradient_queue::*;

let mut queue = GradientQueue::new();

// Push gradient updates with computed priorities
queue.push(GradientUpdate::new(0, 1, Priority::High));   // param 0, grad +1
queue.push(GradientUpdate::new(1, -1, Priority::Medium)); // param 1, grad -1
queue.push(GradientUpdate::new(2, 0, Priority::Low));     // param 2, grad 0

// Drain high-priority updates first
let high_priority: Vec<_> = queue.drain_by_priority(Priority::High);
assert_eq!(high_priority.len(), 1);

// Scheduler with age-based escalation
let mut scheduler = GradientScheduler::new(10); // escalate after 10 ticks
scheduler.step(&mut queue);
```

## The Deeper Idea

Gradient prioritization is a form of *attention* applied to the training process itself. Just as attention mechanisms in transformers focus computation on important tokens, gradient queues focus updates on important parameters. The connection to the agent-attention crate is not coincidental — it's the same principle at a different scale.

For ternary networks specifically, the ternary gradient {-1, 0, +1} makes priority classification trivial: 0 gradients are always Low priority. This means the queue naturally filters out 30-60% of updates in sparse ternary models.

## Related Crates

- `ternary-accumulator` — Accumulate gradients before queuing
- `ternary-checkpoint` — Save training state including queue
- `ternary-optimizer` — Optimizers that consume the prioritized updates
- `ternary-shard-split` — Splitting gradient queues across devices
