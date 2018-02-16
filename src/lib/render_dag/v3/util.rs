use futures_cpupool::{CpuFuture, CpuPool};
use futures::{future::{join_all, ok, Shared}, prelude::*};
use petgraph::{prelude::*, visit::{self, Walker}};
use std::sync::{Arc, RwLock};

pub type Dynamic<T> = Arc<RwLock<Shared<CpuFuture<T, ()>>>>;

pub trait WaitOn {
    fn waitable(&self, pool: &CpuPool) -> CpuFuture<(), ()>;
}

pub fn search_deps<T, F, N, E>(graph: &StableDiGraph<N, E>, start: NodeIndex, f: F) -> Vec<T>
where
    F: Fn(&N) -> Option<T>,
{
    let reversed = visit::Reversed(graph);
    let mut dfs = visit::DfsPostOrder::new(&reversed, start);
    dfs.iter(&reversed)
        .filter_map(|ix| f(&graph[ix]))
        .collect::<Vec<_>>()
}

pub fn search_deps_exactly_one<T, F, N, E>(graph: &StableDiGraph<N, E>, start: NodeIndex, f: F) -> Option<T>
where
    F: Fn(&N) -> Option<T>,
{
    let mut results = search_deps(graph, start, f);
    if results.len() == 1 {
        Some(results.remove(0))
    } else {
        None
    }
}

pub fn search_direct_deps<T, F, N, E>(graph: &StableDiGraph<N, E>, start: NodeIndex, direction: Direction, f: F) -> Vec<T>
where
    F: Fn(&N) -> Option<T>,
{
    graph
        .neighbors_directed(start, direction)
        .filter_map(|ix| f(&graph[ix]))
        .collect::<Vec<_>>()
}

pub fn search_direct_deps_exactly_one<T, F, N, E>(graph: &StableDiGraph<N, E>, start: NodeIndex, direction: Direction, f: F) -> Option<T>
where
    F: Fn(&N) -> Option<T>,
{
    let mut results = search_direct_deps(graph, start, direction, f);
    if results.len() == 1 {
        Some(results.remove(0))
    } else {
        None
    }
}

pub fn wait_on_direct_deps<N, E>(cpu_pool: &CpuPool, graph: &StableDiGraph<N, E>, start: NodeIndex) -> Shared<CpuFuture<(), ()>>
where
    N: WaitOn,
{
    let futures = search_direct_deps(graph, start, Direction::Incoming, |node| {
        Some(node.waitable(cpu_pool))
    });
    cpu_pool
        .spawn(join_all(futures).map_err(|_| ()).map(|_| ()))
        .shared()
}

pub fn spawn_const<T: 'static + Send>(pool: &CpuPool, val: T) -> CpuFuture<T, ()> {
    pool.spawn(ok(val))
}

pub fn dyn<T: 'static + Send>(pool: &CpuPool, a: T) -> Dynamic<T> {
    Arc::new(RwLock::new(spawn_const(pool, a).shared()))
}
