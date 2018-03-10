use futures::{future::{join_all, ok, Shared}, prelude::*};
use futures_cpupool::*;
use petgraph::{self, visit, EdgeDirection, prelude::*};
use std::{fmt, marker::{Send, Sync}, sync::{Arc, RwLock}};

use super::{dyn, edge_filter, spawn_const, Dynamic, Edge, EdgeFilter};

decl_node_runtime! {
    Adder {
        Lit make_lit { static [num: isize] dynamic [] },
        Add make_add { static [] dynamic [res: isize] },
        Print make_print { static [] dynamic [] }
    }
}

type RuntimeGraph = StableDiGraph<Adder, Edge>;

pub struct RenderDAG {
    graph: RuntimeGraph,
}

impl RenderDAG {
    pub fn eval(&self, pool: &CpuPool) -> Option<NodeIndex> {
        let graph = &self.graph;
        let mut walker = visit::Topo::new(graph);
        let mut last = None;
        loop {
            match walker.next(graph) {
                None => {
                    println!("no next node");
                    break;
                }
                Some(ix) => {
                    println!("asd {:?}", graph[ix]);
                    last = Some(ix);
                    match &graph[ix] {
                        &Adder::Add { ref dynamic, .. } => {
                            let mut guard = dynamic.write().expect("failed to acquire write lock");
                            *guard = inputs! {
                                pool graph ix
                                { EdgeFilter::Propagating }
                                {
                                    &Adder::Lit { num, .. } => Some(spawn_const(num)),
                                    &Adder::Add { ref dynamic } => Some(pool.spawn(dynamic.read().expect("failed to lock input").clone().map(|field| field.res).map_err(|_| ()))),
                                    _ => None
                                }
                                |nums: Vec<isize>| {
                                    use std::ops::Deref;
                                    fields::Add::Dynamic { res: nums.iter().map(|r| *r.deref()).sum()}
                                }
                            };
                        }
                        &Adder::Print { ref dynamic } => {
                            let mut guard = dynamic.write().expect("failed to acquire write lock");
                            *guard = inputs! {
                                pool graph ix
                                { EdgeFilter::Propagating }
                                {
                                    &Adder::Add { ref dynamic } => Some(dynamic.read().expect("failed to lock input").clone().map(|field| field.res).map_err(|_| ())),
                                    _ => None
                                }
                                |nums: Vec<isize>| {
                                    for num in nums {
                                        println!("printing {}", num);
                                    }
                                    fields::Print::Dynamic {}
                                }
                            };
                        }
                        _ => println!("no action"),
                    }
                }
            }
        }
        last
    }
}

#[test]
fn eval_test() {
    let mut g = RuntimeGraph::new();
    let pool = CpuPool::new_num_cpus();
    let five = g.add_node(Adder::make_lit(&pool, 5));
    let four = g.add_node(Adder::make_lit(&pool, 4));
    let add = g.add_node(Adder::make_add(&pool, 0));
    let add2 = g.add_node(Adder::make_add(&pool, 0));
    g.add_edge(five, add, Edge::Propagate);
    g.add_edge(four, add, Edge::Propagate);
    g.add_edge(four, add2, Edge::Direct);
    g.add_edge(add, add2, Edge::Propagate);
    let dot = petgraph::dot::Dot::new(&g);
    println!("graph is \n{:?}", dot);
    let last = RenderDAG { graph: g.clone() }.eval(&pool);
    if let &Adder::Add { ref dynamic } = &g[last.unwrap()] {
        let lock = dynamic.read().unwrap();
        use std::ops::Deref;
        assert_eq!(9, lock.clone().wait().unwrap().res);
    } else {
        panic!("Add not the last node")
    }
}
