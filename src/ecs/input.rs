use crate::{
    ecs::{
        resources::Camera,
        systems::{Gui, RuntimeConfiguration},
    },
    renderer::{right_vector, up_vector, RenderFrame, Resized},
};
use std::{cell::RefCell, rc::Rc};

use hashbrown::HashSet;
use imgui_winit_support::WinitPlatform;
#[cfg(feature = "microprofile")]
use microprofile::scope;
use na::RealField;
use parking_lot::Mutex;
use std::sync::Arc;
use winit::{
    self,
    dpi::PhysicalSize,
    event::{
        ButtonId, DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode,
        WindowEvent,
    },
    platform::desktop::EventLoopExtDesktop,
};

#[derive(Debug)]
pub struct InputActions {
    pressed: HashSet<VirtualKeyCode>,
    hold: HashSet<VirtualKeyCode>,
    released: HashSet<VirtualKeyCode>,
    mouse_buttons: HashSet<MouseButton>,
}

impl InputActions {
    pub fn get_key_down(&self, key: VirtualKeyCode) -> bool {
        self.pressed.contains(&key) || self.hold.contains(&key)
    }

    pub fn get_mouse_down(&self, button: MouseButton) -> bool {
        self.mouse_buttons.contains(&button)
    }

    fn promote(&mut self) {
        self.released.clear();
        self.hold.extend(&self.pressed);
        self.pressed.clear();
        // TODO: mouse buttons
    }

    fn record_press(&mut self, keycode: VirtualKeyCode) {
        self.pressed.insert(keycode);
    }

    fn record_release(&mut self, keycode: VirtualKeyCode) {
        self.released.insert(keycode);
        self.hold.remove(&keycode);
    }
}

impl Default for InputActions {
    fn default() -> InputActions {
        InputActions {
            pressed: HashSet::new(),
            hold: HashSet::new(),
            released: HashSet::new(),
            mouse_buttons: HashSet::new(),
        }
    }
}

pub struct InputState {
    key_presses: Vec<Option<VirtualKeyCode>>,
    key_releases: Vec<Option<VirtualKeyCode>>,
    button_presses: Vec<ButtonId>,
}

impl InputState {
    fn clear(&mut self) {
        self.key_presses.clear();
        self.key_releases.clear();
        self.button_presses.clear();
    }
}

impl Default for InputState {
    fn default() -> InputState {
        InputState {
            key_presses: vec![],
            key_releases: vec![],
            button_presses: vec![],
        }
    }
}

pub struct InputHandler {
    pub events_loop: winit::event_loop::EventLoop<()>,
    pub quit_handle: Arc<Mutex<bool>>,
    pub imgui_platform: WinitPlatform,
}

impl InputHandler {
    pub fn exec_system(
        input_handler: Rc<RefCell<InputHandler>>,
        gui: Rc<RefCell<Gui>>,
    ) -> Box<(dyn legion::systems::schedule::Runnable + 'static)> {
        use legion::prelude as lp;
        lp::SystemBuilder::<()>::new("InputHandler")
            .read_resource::<RenderFrame>()
            .write_resource::<RuntimeConfiguration>()
            .write_resource::<InputState>()
            .write_resource::<InputActions>()
            .write_resource::<Camera>()
            .write_resource::<Resized>()
            .build_thread_local(move |_commands, _world, resources, _query| {
                let (
                    ref renderer,
                    ref mut runtime_config,
                    ref mut input_state,
                    ref mut input_actions,
                    ref mut camera,
                    ref mut resized,
                ) = resources;
                #[cfg(feature = "profiling")]
                microprofile::scope!("ecs", "input handler");
                let mut borrowed = input_handler.borrow_mut();
                let InputHandler {
                    ref mut events_loop,
                    ref mut imgui_platform,
                    ref mut quit_handle,
                } = *borrowed;
                // let quit_handle = Arc::clone(&input_handler.quit_handle);
                input_state.clear();
                input_actions.promote();
                let fly_mode = runtime_config.fly_mode;
                let mut toggle_fly_mode = false;
                resized.0 = false;
                events_loop.run_return(|event, _window_target, control_flow| {
                    imgui_platform.handle_event(
                        gui.borrow_mut().imgui.io_mut(),
                        &renderer.instance.window,
                        &event,
                    );
                    match event {
                        Event::WindowEvent {
                            event: WindowEvent::Resized(PhysicalSize { width, height }),
                            ..
                        } => {
                            println!("The window was resized to {}x{}", width, height);
                            // hangs for now
                            // let logical_size = PhysicalSize { width, height }.to_logical::<f32>(window.scale_factor());
                            // println!("logical {:?}", logical_size);
                            resized.0 = true;
                        }
                        Event::WindowEvent {
                            event: WindowEvent::ScaleFactorChanged { scale_factor, .. },
                            ..
                        } => {
                            println!("Scale factor changed {}", scale_factor);
                        }
                        Event::WindowEvent {
                            event: WindowEvent::CloseRequested,
                            ..
                        } => {
                            *quit_handle.lock() = true;
                        }
                        Event::WindowEvent {
                            event:
                                WindowEvent::KeyboardInput {
                                    input:
                                        KeyboardInput {
                                            state,
                                            virtual_keycode,
                                            scancode,
                                            ..
                                        },
                                    ..
                                },
                            ..
                        } => {
                            match state {
                                ElementState::Pressed => {
                                    input_state.key_presses.push(virtual_keycode);
                                    match virtual_keycode {
                                        Some(virtual_keycode) => {
                                            input_actions.record_press(virtual_keycode)
                                        }
                                        None => {
                                            dbg!(scancode);
                                        }
                                    }
                                }
                                ElementState::Released => {
                                    input_state.key_releases.push(virtual_keycode);
                                    match virtual_keycode {
                                        Some(virtual_keycode) => {
                                            input_actions.record_release(virtual_keycode)
                                        }
                                        None => {
                                            dbg!(scancode);
                                        }
                                    }
                                }
                            }
                            match virtual_keycode {
                                Some(VirtualKeyCode::G) if state == ElementState::Pressed => {
                                    toggle_fly_mode = true;
                                }
                                Some(VirtualKeyCode::Escape) => {
                                    *quit_handle.lock() = true;
                                }
                                _ => (),
                            }
                        }
                        Event::DeviceEvent {
                            event: DeviceEvent::MouseMotion { delta: (x, y), .. },
                            ..
                        } if fly_mode => {
                            let y_angle = f32::pi() / 180.0 * y as f32;
                            let x_angle = f32::pi() / 180.0 * x as f32;
                            camera.rotation *=
                                na::Rotation3::from_axis_angle(&right_vector(), y_angle);
                            camera.rotation = na::Rotation3::from_axis_angle(&up_vector(), x_angle)
                                * camera.rotation;
                        }
                        Event::DeviceEvent {
                            event:
                                DeviceEvent::Button {
                                    button,
                                    state: ElementState::Pressed,
                                },
                            ..
                        } => {
                            input_state.button_presses.push(button);
                        }
                        _ => (),
                    };
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                });
                runtime_config.fly_mode = if toggle_fly_mode { !fly_mode } else { fly_mode };
                imgui_platform
                    .prepare_frame(gui.borrow_mut().imgui.io_mut(), &renderer.instance.window)
                    .expect("Failed to prepare frame");
            })
    }

    /*
    /// Returns true if resized
    pub fn exec(
        &mut self,
        window: &winit::window::Window,
        gui: &mut imgui::Context,
        input_state: &mut InputState,
        camera: &mut Camera,
        runtime_config: &mut RuntimeConfiguration,
    ) -> bool {
        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "input handler");
        let quit_handle = Arc::clone(&self.quit_handle);
        input_state.clear();
        let fly_mode = runtime_config.fly_mode;
        let platform = &mut self.imgui_platform;
        let mut toggle_fly_mode = false;
        let mut resized = false;
        self.events_loop
            .run_return(|event, _window_target, control_flow| {
                platform.handle_event(gui.io_mut(), &window, &event);
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::Resized(PhysicalSize { width, height }),
                        ..
                    } => {
                        println!("The window was resized to {}x{}", width, height);
                        // hangs for now
                        // let logical_size = PhysicalSize { width, height }.to_logical::<f32>(window.scale_factor());
                        // println!("logical {:?}", logical_size);
                        resized = true;
                    }
                    Event::WindowEvent {
                        event: WindowEvent::ScaleFactorChanged { scale_factor, .. },
                        ..
                    } => {
                        println!("Scale factor changed {}", scale_factor);
                    }
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        *quit_handle.lock() = true;
                    }
                    Event::WindowEvent {
                        event:
                            WindowEvent::KeyboardInput {
                                input:
                                    KeyboardInput {
                                        state,
                                        virtual_keycode,
                                        ..
                                    },
                                ..
                            },
                        ..
                    } => {
                        match state {
                            ElementState::Pressed => input_state.key_presses.push(virtual_keycode),
                            ElementState::Released => {
                                input_state.key_releases.push(virtual_keycode)
                            }
                        }
                        match virtual_keycode {
                            Some(VirtualKeyCode::G) if state == ElementState::Pressed => {
                                toggle_fly_mode = true;
                            }
                            Some(VirtualKeyCode::Escape) => {
                                *quit_handle.lock() = true;
                            }
                            _ => (),
                        }
                    }
                    Event::DeviceEvent {
                        event: DeviceEvent::MouseMotion { delta: (x, y), .. },
                        ..
                    } if fly_mode => {
                        let y_angle = f32::pi() / 180.0 * y as f32;
                        let x_angle = f32::pi() / 180.0 * x as f32;
                        camera.rotation *= na::Rotation3::from_axis_angle(&right_vector(), y_angle);
                        camera.rotation =
                            na::Rotation3::from_axis_angle(&up_vector(), x_angle) * camera.rotation;
                    }
                    Event::DeviceEvent {
                        event:
                            DeviceEvent::Button {
                                button,
                                state: ElementState::Pressed,
                            },
                        ..
                    } => {
                        input_state.button_presses.push(button);
                    }
                    _ => (),
                };
                *control_flow = winit::event_loop::ControlFlow::Exit;
            });
        runtime_config.fly_mode = if toggle_fly_mode { !fly_mode } else { fly_mode };
        platform
            .prepare_frame(gui.io_mut(), &window)
            .expect("Failed to prepare frame");

        resized
    }
    */
}