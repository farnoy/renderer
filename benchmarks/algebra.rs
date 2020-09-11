use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra as na;
use simba::simd::{self, SimdValue};

fn add_two(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add two lanewise");
    for size in [16, 1024].iter().cloned() {
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<na::Vector3<f32>>()) as u64,
        ));

        let lhs = vec![na::Vector3::<f32>::zeros(); size];
        let rhs = vec![na::Vector3::<f32>::new(1.0, 1.0, 1.0); size];

        group.bench_with_input(
            BenchmarkId::new("vec3", size),
            &(lhs, rhs),
            |b, (ref lhs, ref rhs)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs + rhs);
                    }
                })
            },
        );

        let lhs_2 = vec![na::Vector3::<simba::simd::f32x2>::splat(na::Vector3::zeros()); size / 2];
        let rhs_2 = vec![
            na::Vector3::<simba::simd::f32x2>::splat(na::Vector3::new(1.0, 1.0, 1.0));
            size / 2
        ];

        group.bench_with_input(
            BenchmarkId::new("f32x2", size / 2),
            &(lhs_2, rhs_2),
            |b, (ref lhs, ref rhs)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs + rhs);
                    }
                })
            },
        );

        let lhs_4 = vec![na::Vector3::<simba::simd::f32x4>::splat(na::Vector3::zeros()); size / 4];
        let rhs_4 = vec![
            na::Vector3::<simba::simd::f32x4>::splat(na::Vector3::new(1.0, 1.0, 1.0));
            size / 4
        ];

        group.bench_with_input(
            BenchmarkId::new("f32x4", size / 4),
            &(lhs_4, rhs_4),
            |b, (ref lhs, ref rhs)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs + rhs);
                    }
                })
            },
        );

        let lhs_8 = vec![na::Vector3::<simba::simd::f32x8>::splat(na::Vector3::zeros()); size / 8];
        let rhs_8 = vec![
            na::Vector3::<simba::simd::f32x8>::splat(na::Vector3::new(1.0, 1.0, 1.0));
            size / 8
        ];

        group.bench_with_input(
            BenchmarkId::new("f32x8", size / 8),
            &(lhs_8, rhs_8),
            |b, (ref lhs, ref rhs)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs + rhs);
                    }
                })
            },
        );

        let lhs_16 =
            vec![na::Vector3::<simba::simd::f32x16>::splat(na::Vector3::zeros()); size / 16];
        let rhs_16 = vec![
            na::Vector3::<simba::simd::f32x16>::splat(na::Vector3::new(1.0, 1.0, 1.0));
            size / 16
        ];

        group.bench_with_input(
            BenchmarkId::new("f32x16", size / 16),
            &(lhs_16, rhs_16),
            |b, (ref lhs, ref rhs)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs + rhs);
                    }
                })
            },
        );
    }
    group.finish();
}

fn mask_two(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mask two lanewise");
    for size in [16, 1024].iter().cloned() {
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<na::Vector3<f32>>()) as u64,
        ));

        let lhs = vec![na::Vector3::<f32>::zeros(); size];
        let rhs = vec![na::Vector3::<f32>::new(1.0, 1.0, 1.0); size];

        group.bench_with_input(
            BenchmarkId::new("vec3", size),
            &(lhs, rhs),
            |b, (ref lhs, ref rhs)| {
                b.iter(|| {
                    for (ix, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
                        black_box(if ix % 2 == 0 { lhs } else { rhs });
                    }
                })
            },
        );

        let lhs_2 = vec![na::Vector3::<simba::simd::f32x2>::splat(na::Vector3::zeros()); size / 2];
        let rhs_2 = vec![
            na::Vector3::<simba::simd::f32x2>::splat(na::Vector3::new(1.0, 1.0, 1.0));
            size / 2
        ];
        let m2 = simba::simd::m32x2::new(true, false);

        group.bench_with_input(
            BenchmarkId::new("f32x2", size / 2),
            &(lhs_2, rhs_2, m2),
            |b, (ref lhs, ref rhs, ref mask)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs.select(*mask, *rhs));
                    }
                })
            },
        );

        let lhs_4 = vec![na::Vector3::<simba::simd::f32x4>::splat(na::Vector3::zeros()); size / 4];
        let rhs_4 = vec![
            na::Vector3::<simba::simd::f32x4>::splat(na::Vector3::new(1.0, 1.0, 1.0));
            size / 4
        ];
        let m4 = simba::simd::m32x4::new(true, false, true, false);

        group.bench_with_input(
            BenchmarkId::new("f32x4", size / 4),
            &(lhs_4, rhs_4, m4),
            |b, (ref lhs, ref rhs, ref mask)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs.select(*mask, *rhs));
                    }
                })
            },
        );

        let lhs_8 = vec![na::Vector3::<simba::simd::f32x8>::splat(na::Vector3::zeros()); size / 8];
        let rhs_8 = vec![
            na::Vector3::<simba::simd::f32x8>::splat(na::Vector3::new(1.0, 1.0, 1.0));
            size / 8
        ];
        let m8 = simba::simd::m32x8::new(true, false, true, false, true, false, true, false);

        group.bench_with_input(
            BenchmarkId::new("f32x8", size / 8),
            &(lhs_8, rhs_8, m8),
            |b, (ref lhs, ref rhs, ref mask)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs.select(*mask, *rhs));
                    }
                })
            },
        );

        let lhs_16 =
            vec![na::Vector3::<simba::simd::f32x16>::splat(na::Vector3::zeros()); size / 16];
        let rhs_16 = vec![
            na::Vector3::<simba::simd::f32x16>::splat(na::Vector3::new(1.0, 1.0, 1.0));
            size / 16
        ];
        let m16 = simba::simd::m32x16::new(
            true, false, true, false, true, false, true, false, true, false, true, false, true,
            false, true, false,
        );

        group.bench_with_input(
            BenchmarkId::new("f32x16", size / 16),
            &(lhs_16, rhs_16, m16),
            |b, (ref lhs, ref rhs, ref mask)| {
                b.iter(|| {
                    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
                        black_box(lhs.select(*mask, *rhs));
                    }
                })
            },
        );
    }
    group.finish();
}

fn chonker_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Chonker");

    pub fn chonker(xs: &[f32]) -> simd::f32x4 {
        let mut acc = simd::f32x4::splat(1.0);

        for chunk in xs.chunks(4) {
            let mut r = simd::f32x4::splat(9999.0);
            let mut mask = simd::m32x4::splat(false);
            for (ix, item) in chunk.iter().enumerate() {
                r.replace(ix, *item);
                mask.replace(ix, true);
            }
            acc *= r.select(mask, simd::f32x4::splat(1.0));
        }

        acc
    }

    pub fn exact_chonker(xs: &[f32]) -> simd::f32x4 {
        let mut acc = simd::f32x4::splat(1.0);
        let mut chunk_iter = xs.chunks_exact(4);

        for chunk in &mut chunk_iter {
            acc *= simd::f32x4::from_slice_unaligned(chunk);
        }

        let mut r = simd::f32x4::splat(1.0);
        let mut mask = simd::m32x4::splat(false);

        for (ix, item) in chunk_iter.remainder().iter().enumerate() {
            r.replace(ix, *item);
            mask.replace(ix, true);
        }

        acc * r
    }

    for size in [64, 512, 4096, 40960].iter().map(|x| x + 2) {
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>()) as u64,
        ));

        let xs = vec![1.0; size];

        group.bench_with_input(BenchmarkId::new("scalar", size), xs.as_slice(), |b, xs| {
            b.iter(|| {
                let mut r = [1.0f32; 4];
                for (ix, x) in xs.iter().enumerate() {
                    r[ix % 4] *= x;
                }
                black_box(r);
            })
        });

        group.bench_with_input(BenchmarkId::new("chonker", size), xs.as_slice(), |b, xs| {
            b.iter(|| {
                black_box(chonker(xs));
            })
        });

        group.bench_with_input(
            BenchmarkId::new("chonker exact", size),
            xs.as_slice(),
            |b, xs| {
                b.iter(|| {
                    black_box(exact_chonker(xs));
                })
            },
        );
    }
    group.finish();
}

criterion_group!(adds, add_two);
criterion_group!(masks, mask_two);
criterion_group!(chonker, chonker_bench);
criterion_main!(adds, masks, chonker);
