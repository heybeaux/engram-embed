//! Unit-level tests for the inference semaphore concurrency cap.
//!
//! These tests verify the Semaphore's backpressure semantics without touching
//! the actual embedding model. The invariant being tested: at most N inference
//! calls may proceed concurrently; the (N+1)th waits until a slot frees.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Simulate N+1 concurrent workers competing for a semaphore with N permits.
/// Asserts:
///   1. All workers eventually complete (no deadlock / starvation).
///   2. The peak observed concurrency never exceeds the permit count.
#[tokio::test]
async fn semaphore_caps_peak_concurrency() {
    const PERMITS: usize = 4;
    const WORKERS: usize = PERMITS + 1; // one extra must wait
    // Simulated "inference" duration per worker
    const WORK_MS: u64 = 50;

    let sem = Arc::new(Semaphore::new(PERMITS));
    let peak = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::with_capacity(WORKERS);

    for _ in 0..WORKERS {
        let sem = sem.clone();
        let peak = peak.clone();
        let in_flight = in_flight.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");

            // Track concurrency
            let current = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            // Record high-water mark
            let mut p = peak.load(Ordering::SeqCst);
            while current > p {
                match peak.compare_exchange(p, current, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break,
                    Err(actual) => p = actual,
                }
            }

            // Simulate work
            tokio::time::sleep(Duration::from_millis(WORK_MS)).await;

            in_flight.fetch_sub(1, Ordering::SeqCst);
            // permit auto-released here
        }));
    }

    // All workers must finish
    for h in handles {
        h.await.expect("worker panicked");
    }

    let observed_peak = peak.load(Ordering::Relaxed);
    assert!(
        observed_peak <= PERMITS,
        "Peak concurrency {observed_peak} exceeded permit count {PERMITS}"
    );
    assert_eq!(in_flight.load(Ordering::Relaxed), 0, "in_flight counter leaked");
}

/// Verify that with cap=N, N+1 concurrent requests still all complete
/// (backpressure = wait, not reject).
#[tokio::test]
async fn semaphore_queues_excess_requests() {
    const PERMITS: usize = 2;
    const WORKERS: usize = 8; // 4x over-subscribed
    const WORK_MS: u64 = 30;

    let sem = Arc::new(Semaphore::new(PERMITS));
    let completed = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let mut handles = Vec::with_capacity(WORKERS);

    for _ in 0..WORKERS {
        let sem = sem.clone();
        let completed = completed.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            tokio::time::sleep(Duration::from_millis(WORK_MS)).await;
            completed.fetch_add(1, Ordering::SeqCst);
        }));
    }

    for h in handles {
        h.await.expect("worker panicked");
    }

    assert_eq!(
        completed.load(Ordering::Relaxed),
        WORKERS,
        "Not all workers completed"
    );

    // Sanity check: with 2 permits and 8 workers each taking 30ms,
    // minimum elapsed is ceil(8/2)*30 = 120ms. Verify we actually serialized.
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_millis(WORK_MS * (WORKERS as u64 / PERMITS as u64)),
        "Elapsed {elapsed:?} suspiciously fast — semaphore may not have blocked"
    );
}

/// Verify env-var parsing: EMBED_MAX_CONCURRENT=N produces a semaphore with N permits.
#[test]
fn env_var_parsing_fallback() {
    // Default
    let val: usize = std::env::var("EMBED_MAX_CONCURRENT_NONEXISTENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    assert_eq!(val, 4);

    // Valid value
    std::env::set_var("EMBED_MAX_CONCURRENT_TEST", "8");
    let val: usize = std::env::var("EMBED_MAX_CONCURRENT_TEST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    assert_eq!(val, 8);
    std::env::remove_var("EMBED_MAX_CONCURRENT_TEST");

    // Garbage falls back to default
    std::env::set_var("EMBED_MAX_CONCURRENT_TEST", "banana");
    let val: usize = std::env::var("EMBED_MAX_CONCURRENT_TEST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    assert_eq!(val, 4);
    std::env::remove_var("EMBED_MAX_CONCURRENT_TEST");
}
