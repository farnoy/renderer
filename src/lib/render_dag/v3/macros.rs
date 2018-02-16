use petgraph::{visit, EdgeDirection};
use futures_cpupool::{CpuFuture, CpuPool};
use super::util::*;

macro_rules! decl_node_runtime {
    ($name:ident {
        $($field:ident {
            make $make_field:ident
            static [$($static_name:ident: $static_type:ty),*]
            dynamic [$($dynamic_name:ident: $dynamic_type:ty),*]
            $(forward $forward:path)*
        })+
    }) => {
        mod fields {
            $(
                #[allow(non_snake_case)]
                pub mod $field {
                    use super::super::{$($forward),*};

                    #[derive(Debug)]
                    pub struct Dynamic {
                        $(
                            pub $dynamic_name : $dynamic_type,
                        ),*
                    }

                    impl Dynamic {
                        pub fn new($($dynamic_name: $dynamic_type),*) -> Dynamic {
                            Dynamic {
                                $($dynamic_name: $dynamic_name),*
                            }
                        }
                    }
                }
            )*
        }

        #[derive(Clone)]
        pub enum $name {
            $($field {
                $($static_name: $static_type),*,
                dynamic: Dynamic<fields::$field::Dynamic>,
            }),+
        }

        impl $name {
            $(
                fn $make_field(pool: &CpuPool, $($static_name: $static_type),*, $($dynamic_name: $dynamic_type),*) -> $name {
                    $name::$field {
                        dynamic: dyn(pool, fields::$field::Dynamic::new($($dynamic_name),*)),
                        $($static_name: $static_name),*,
                    }
                }
            )*
        }

        impl WaitOn for $name {
            fn waitable(&self, pool: &CpuPool) -> CpuFuture<(), ()> {
                match *self {
                    $($name::$field { ref dynamic, .. } => {
                        let fut = dynamic.read().expect("can't read the waitable dynamic").clone();
                        pool.spawn(fut.map_err(|_| ()).map(|_| ()))
                    })*
                }
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let output = match *self {
                    $($name::$field { .. } => stringify!($field)),*
                };
                write!(f, "{}::{}", stringify!($name), output)
            }
        }
    }
}

macro_rules! inputs {
    ($pool:ident $graph:ident $ix:ident { $edge_filter:expr } { $($match_pat:pat => $match_expr:expr),+ } $process:expr) => {{
        let ins = $graph.edges_directed($ix, EdgeDirection::Incoming)
            .filter(|edge_ref| edge_filter($edge_filter, edge_ref.weight()))
            .map(|edge_ref| edge_ref.source())
            .filter_map(|ix| match &$graph[ix] {
                $($match_pat => $match_expr),+
            })
            .collect::<Vec<_>>();
        $pool.spawn(join_all(ins).map_err(|_| ()).map(|nums| {
            // use std::ops::Deref;
            // fields::Add::Dynamic { res: nums.iter().map(|r| *r.deref()).sum()}
            $process(nums)
        })).shared()
    }}
}
