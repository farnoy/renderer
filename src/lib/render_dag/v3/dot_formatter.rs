use std::{borrow::Cow, collections::HashMap, io::{self, Write}};
use petgraph::{prelude::*,
               visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences, NodeIndexable, NodeRef}};

use super::RuntimeGraph;

pub fn dot(g: &RuntimeGraph, names: &HashMap<NodeIndex, Cow<str>>) -> io::Result<String> {
    let mut buffer: Vec<u8> = vec![];
    writeln!(buffer, "digraph {{")?;

    // output all labels
    for node in g.node_references() {
        match names.get(&node.id()) {
            Some(name) => writeln!(buffer, "  {} [label=\"{}\"]", g.to_index(node.id()), name)?,
            None => writeln!(
                buffer,
                "  {} [label=\"{:?}\"]",
                g.to_index(node.id()),
                node.weight()
            )?,
        }
    }
    // output all edges
    for edge in g.edge_references() {
        writeln!(
            buffer,
            "  {} -> {} [label=\"{:?}\"]",
            g.to_index(edge.source()),
            g.to_index(edge.target()),
            edge.weight()
        )?;
    }

    writeln!(buffer, "}}")?;
    Ok(String::from_utf8(buffer).unwrap())
}
