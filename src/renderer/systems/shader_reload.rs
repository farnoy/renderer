use std::{
    path::Path,
    process::{Command, Output, Stdio},
    sync::mpsc::{channel, Receiver},
    time::{Duration, Instant},
};

use bevy_ecs::prelude::*;
use hashbrown::HashMap;
use notify::{watcher, DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};

pub(crate) struct ShaderReload {
    #[allow(dead_code)]
    watcher: RecommendedWatcher,
    rx: Receiver<DebouncedEvent>,
}

#[derive(Default)]
pub(crate) struct ReloadedShaders(pub(crate) HashMap<String, (Instant, Vec<u8>)>);

impl Default for ShaderReload {
    fn default() -> Self {
        let (tx, rx) = channel();
        let mut watcher = watcher(tx, Duration::from_millis(100)).unwrap();
        watcher
            .watch(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("shaders"),
                RecursiveMode::NonRecursive,
            )
            .unwrap();

        ShaderReload { watcher, rx }
    }
}

pub(crate) fn reload_shaders(shader_reload: NonSend<ShaderReload>, mut reloaded_shaders: ResMut<ReloadedShaders>) {
    loop {
        match shader_reload.rx.try_recv() {
            Ok(DebouncedEvent::Write(path)) => {
                let c = Command::new("glslc")
                    .args(&[
                        "-g",
                        "--target-env=vulkan1.2",
                        "-o",
                        "-", // stdout
                        path.to_str().unwrap(),
                    ])
                    .stdout(Stdio::piped())
                    .spawn()
                    .unwrap();

                // TODO: async, parallel & error recovery
                let Output { stdout, status, .. } = c.wait_with_output().unwrap();
                if status.success() {
                    let now = Instant::now();
                    reloaded_shaders
                        .0
                        .insert(path.to_str().unwrap().to_owned(), (now, stdout));
                }
            }
            Ok(_) => {}
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => panic!("shader notify service disconnected"),
        }
    }
}
