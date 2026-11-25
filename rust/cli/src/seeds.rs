use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn run_seeds<T, F>(seeds: &[String], jobs: usize, run: F) -> Vec<T>
where
    T: Send,
    F: Fn(&str) -> T + Sync,
{
    let jobs = jobs.clamp(1, seeds.len().max(1));
    let next = AtomicUsize::new(0);
    let results = Mutex::new(
        std::iter::repeat_with(|| None)
            .take(seeds.len())
            .collect::<Vec<Option<T>>>(),
    );
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(seed) = seeds.get(index) else {
                        break;
                    };
                    let result = run(seed);
                    results.lock().expect("seed results lock")[index] = Some(result);
                }
            });
        }
    });
    results
        .into_inner()
        .expect("seed results lock")
        .into_iter()
        .map(|result| result.expect("every seed produces a result"))
        .collect()
}
