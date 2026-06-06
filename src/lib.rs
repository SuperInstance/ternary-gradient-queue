//! # ternary-gradient-queue
//!
//! Priority queue for ternary gradients. Not all parameters deserve equal
//! attention. This crate orders gradient updates by importance — parameters
//! with large accumulated gradients get updated first.

pub type Trit = i8;

/// Priority level for a gradient update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl Priority {
    pub fn from_magnitude(net_gradient: i64, threshold_low: i64, threshold_high: i64) -> Self {
        let abs = net_gradient.abs();
        if abs >= threshold_high { Priority::Critical }
        else if abs >= threshold_high / 2 { Priority::High }
        else if abs >= threshold_low { Priority::Medium }
        else { Priority::Low }
    }
}

/// A queued gradient update for a single parameter.
#[derive(Debug, Clone)]
pub struct GradientUpdate {
    pub param_idx: usize,
    pub gradient: Trit,
    pub priority: Priority,
    pub accumulated_count: usize,
    pub net_signal: i64,
}

/// Priority queue for ternary gradient updates.
#[derive(Debug, Clone)]
pub struct GradientQueue {
    updates: Vec<GradientUpdate>,
    capacity: usize,
    total_enqueued: usize,
    total_processed: usize,
}

impl GradientQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            updates: Vec::with_capacity(capacity),
            capacity,
            total_enqueued: 0,
            total_processed: 0,
        }
    }

    /// Enqueue a gradient update.
    pub fn enqueue(&mut self, param_idx: usize, gradient: Trit, accumulated_count: usize, net_signal: i64) {
        let priority = Priority::from_magnitude(net_signal, 5, 20);
        if self.updates.len() >= self.capacity {
            // Evict lowest priority
            if let Some(min_idx) = self.updates.iter().enumerate()
                .min_by_key(|(_, u)| u.priority)
                .map(|(i, _)| i)
            {
                if self.updates[min_idx].priority < priority {
                    self.updates.remove(min_idx);
                } else {
                    return; // drop this update
                }
            }
        }
        self.updates.push(GradientUpdate {
            param_idx,
            gradient,
            priority,
            accumulated_count,
            net_signal,
        });
        self.total_enqueued += 1;
    }

    /// Dequeue the highest-priority update.
    pub fn dequeue(&mut self) -> Option<GradientUpdate> {
        let best_idx = self.updates.iter().enumerate()
            .max_by_key(|(_, u)| (u.priority, u.net_signal.abs()))
            .map(|(i, _)| i)?;
        self.total_processed += 1;
        Some(self.updates.remove(best_idx))
    }

    /// Dequeue all updates in priority order.
    pub fn drain_ordered(&mut self) -> Vec<GradientUpdate> {
        self.updates.sort_by(|a, b| b.priority.cmp(&a.priority)
            .then(b.net_signal.abs().cmp(&a.net_signal.abs())));
        let result = self.updates.clone();
        self.total_processed += result.len();
        self.updates.clear();
        result
    }

    /// Number of queued updates.
    pub fn len(&self) -> usize {
        self.updates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    /// Distribution of priorities in queue.
    pub fn priority_distribution(&self) -> [usize; 4] {
        let mut dist = [0usize; 4];
        for u in &self.updates {
            dist[u.priority as usize] += 1;
        }
        dist
    }

    /// Total updates enqueued since creation.
    pub fn total_enqueued(&self) -> usize { self.total_enqueued }
    pub fn total_processed(&self) -> usize { self.total_processed }
}

/// Gradient scheduler that controls update ordering across training steps.
#[derive(Debug)]
pub struct GradientScheduler {
    queue: GradientQueue,
    /// Parameters that have been updated this step
    updated_params: Vec<usize>,
    /// Maximum updates per step (budget)
    budget: usize,
    /// Step counter
    step: usize,
}

impl GradientScheduler {
    pub fn new(queue_capacity: usize, budget: usize) -> Self {
        Self {
            queue: GradientQueue::new(queue_capacity),
            updated_params: Vec::new(),
            budget,
            step: 0,
        }
    }

    /// Submit a gradient for scheduling.
    pub fn submit(&mut self, param_idx: usize, gradient: Trit, count: usize, signal: i64) {
        self.queue.enqueue(param_idx, gradient, count, signal);
    }

    /// Get next batch of updates (up to budget).
    pub fn next_batch(&mut self) -> Vec<GradientUpdate> {
        let mut batch = Vec::with_capacity(self.budget);
        self.updated_params.clear();
        while batch.len() < self.budget {
            if let Some(update) = self.queue.dequeue() {
                if !self.updated_params.contains(&update.param_idx) {
                    self.updated_params.push(update.param_idx);
                    batch.push(update);
                }
            } else {
                break;
            }
        }
        self.step += 1;
        batch
    }

    pub fn step(&self) -> usize { self.step }
    pub fn queue_len(&self) -> usize { self.queue.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_from_magnitude() {
        assert_eq!(Priority::from_magnitude(50, 5, 20), Priority::Critical);
        assert_eq!(Priority::from_magnitude(15, 5, 20), Priority::High);
        assert_eq!(Priority::from_magnitude(7, 5, 20), Priority::Medium);
        assert_eq!(Priority::from_magnitude(2, 5, 20), Priority::Low);
    }

    #[test]
    fn test_enqueue_dequeue_highest_first() {
        let mut q = GradientQueue::new(10);
        q.enqueue(0, 1, 10, 5);     // Medium
        q.enqueue(1, -1, 50, 100);  // Critical
        q.enqueue(2, 0, 5, 1);      // Low
        let first = q.dequeue().unwrap();
        assert_eq!(first.param_idx, 1); // Critical dequeued first
    }

    #[test]
    fn test_drain_ordered() {
        let mut q = GradientQueue::new(10);
        q.enqueue(0, 1, 10, 5);
        q.enqueue(1, -1, 50, 100);
        q.enqueue(2, 0, 5, 1);
        let drained = q.drain_ordered();
        assert_eq!(drained[0].param_idx, 1); // Critical first
        assert!(q.is_empty());
    }

    #[test]
    fn test_capacity_eviction() {
        let mut q = GradientQueue::new(2);
        q.enqueue(0, 1, 5, 5);    // Medium
        q.enqueue(1, -1, 5, 5);   // Medium
        q.enqueue(2, 1, 100, 200); // Critical — should evict one Medium
        assert_eq!(q.len(), 2);
        let drained = q.drain_ordered();
        assert_eq!(drained[0].param_idx, 2); // Critical survived
    }

    #[test]
    fn test_priority_distribution() {
        let mut q = GradientQueue::new(10);
        q.enqueue(0, 1, 5, 2);     // Low
        q.enqueue(1, -1, 5, 10);   // Medium
        q.enqueue(2, 1, 5, 30);    // Critical
        let dist = q.priority_distribution();
        assert_eq!(dist[0], 1); // Low
        assert_eq!(dist[2], 1); // High
        assert_eq!(dist[3], 1); // Critical
    }

    #[test]
    fn test_scheduler_budget() {
        let mut sched = GradientScheduler::new(100, 2);
        sched.submit(0, 1, 5, 100); // Critical
        sched.submit(1, -1, 5, 50); // High
        sched.submit(2, 0, 5, 5);   // Medium
        let batch = sched.next_batch();
        assert_eq!(batch.len(), 2); // budget = 2
        assert_eq!(batch[0].param_idx, 0); // Critical first
    }

    #[test]
    fn test_scheduler_no_duplicate_params() {
        let mut sched = GradientScheduler::new(100, 5);
        sched.submit(0, 1, 5, 100);
        sched.submit(0, -1, 5, 100); // same param, should be deduped
        let batch = sched.next_batch();
        let param_0_count = batch.iter().filter(|u| u.param_idx == 0).count();
        assert_eq!(param_0_count, 1);
    }

    #[test]
    fn test_total_tracking() {
        let mut q = GradientQueue::new(10);
        q.enqueue(0, 1, 5, 5);
        q.enqueue(1, -1, 5, 5);
        assert_eq!(q.total_enqueued(), 2);
        q.dequeue();
        assert_eq!(q.total_processed(), 1);
    }
}
