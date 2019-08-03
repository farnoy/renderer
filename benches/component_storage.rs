extern crate criterion;
extern crate nalgebra as na;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use hashbrown::HashMap;
use std::collections::BTreeMap;

fn criterion_benchmark(c: &mut Criterion) {
    let mut btree = BTreeMap::<u32, na::Vector3<u32>>::new();
    let mut hashmap = HashMap::<u32, na::Vector3<u32>>::new();
    let mut vec = Vec::<na::Vector3<u32>>::new();
    vec.resize(5000, na::zero());
    for x in 0..5000 {
        btree.insert(x, na::Vector3::new(x, 0, 0));
        hashmap.insert(x, na::Vector3::new(x, 0, 0));
        vec[x as usize] = na::Vector3::new(x, 0, 0);
    }
    // Doing deep clones for every iteration to avoid fully cached structures
    c.bench_function("btree get sequential", {
        let btree = btree.clone();
        move |b| {
            b.iter_batched(
                || btree.clone(),
                |btree| {
                    let mut sum = 0;
                    for x in 0..5000 {
                        sum += btree.get(&x).unwrap().x;
                    }
                    black_box(sum);
                },
                BatchSize::SmallInput,
            )
        }
    });

    c.bench_function("btree values iter", {
        let btree = btree.clone();
        move |b| {
            b.iter_batched(
                || btree.clone(),
                |btree| {
                    let sum = btree.values().map(|v| v.x).sum::<u32>();
                    black_box(sum);
                },
                BatchSize::SmallInput,
            )
        }
    });

    c.bench_function("hashmap get sequential", {
        let hashmap = hashmap.clone();
        move |b| {
            b.iter_batched(
                || hashmap.clone(),
                |hashmap| {
                    let mut sum = 0;
                    for x in 0..5000 {
                        sum += hashmap.get(&x).unwrap().x;
                    }
                    black_box(sum);
                },
                BatchSize::SmallInput,
            )
        }
    });

    c.bench_function("hashmap values iter", {
        let hashmap = hashmap.clone();
        move |b| {
            b.iter_batched(
                || hashmap.clone(),
                |hashmap| {
                    let sum = hashmap.values().map(|v| v.x).sum::<u32>();
                    black_box(sum);
                },
                BatchSize::SmallInput,
            )
        }
    });

    c.bench_function("vec get sequential", {
        let vec = vec.clone();
        move |b| {
            b.iter_batched(
                || vec.clone(),
                |vec| {
                    let mut sum = 0;
                    for x in 0..5000 {
                        sum += vec.get(x as usize).unwrap().x;
                    }
                    black_box(sum);
                },
                BatchSize::SmallInput,
            )
        }
    });

    c.bench_function("vec values iter", {
        let vec = vec.clone();
        move |b| {
            b.iter_batched(
                || vec.clone(),
                |vec| {
                    let sum = vec.iter().map(|v| v.x).sum::<u32>();
                    black_box(sum);
                },
                BatchSize::SmallInput,
            )
        }
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
