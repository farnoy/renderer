use petgraph;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum ExecResult {
    Number(u32),
}

type Graph = petgraph::Graph<(String, Arc<Fn(Vec<ExecResult>) -> ExecResult + Send + Sync>), ()>;

pub struct RenderDAG {
    graph: Graph,
}

impl RenderDAG {
    pub fn new() -> RenderDAG {
        let mut g = Graph::new();
        let end = g.add_node((
            String::from("end"),
            Arc::new(|results| {
                let sum = results.iter().fold(0, |accu, res| {
                    let &ExecResult::Number(n) = res;
                    accu + n
                }) + 100;
                ExecResult::Number(sum)
            }),
        ));
        RenderDAG { graph: g }
    }

    pub fn run(&self) -> ExecResult {
        use futures;
        use futures::Future;
        use futures::future::Shared;
        use futures::sync::oneshot;
        use futures_cpupool::{CpuPool, CpuFuture};

        let pool = CpuPool::new_num_cpus();

        let mut g: petgraph::Graph<Option<Shared<CpuFuture<ExecResult, ()>>>, _> =
            self.graph.map(|_ix, _node| None, |_ix, edge| edge);
        for node in petgraph::algo::toposort(&self.graph) {
            println!("arrived at {}", (self.graph[node].0));
            let future = {
                let f = (self.graph[node].1).clone();
                let inputs = g.neighbors_directed(node, petgraph::EdgeDirection::Incoming)
                    .map(|ix| {
                        g[ix]
                            .clone()
                            .expect("previous computation not available")
                            .clone()
                    })
                    .map(|fut| fut.wait().expect("Future failed"))
                    .map(|shared_item| *shared_item)
                    .collect();
                pool.spawn_fn::<_, Result<ExecResult, ()>>(move || Ok(f(inputs)))
                    .shared()
            };
            if self.graph[node].0 == "end" {
                return *future.wait().expect("computation failed");
            } else {
                g[node] = Some(future);
            }
        }
        panic!("no end node!");
    }
}
