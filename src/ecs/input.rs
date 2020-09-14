use crate::{
    ecs::{
        resources::Camera,
        systems::{Gui, RuntimeConfiguration},
    },
    renderer::{right_vector, up_vector, RenderFrame, Resized},
};
use std::{cell::RefCell, rc::Rc};

use bevy_ecs::prelude::*;
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
pub(crate) struct InputActions {
    pressed: HashSet<VirtualKeyCode>,
    hold: HashSet<VirtualKeyCode>,
    released: HashSet<VirtualKeyCode>,
    mouse_buttons: HashSet<MouseButton>,
}

impl InputActions {
    pub(crate) fn get_key_down(&self, key: VirtualKeyCode) -> bool {
        self.pressed.contains(&key) || self.hold.contains(&key)
    }

    pub(crate) fn get_mouse_down(&self, button: MouseButton) -> bool {
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

pub(crate) struct InputState {
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

pub(crate) struct InputHandler {
    pub(crate) events_loop: winit::event_loop::EventLoop<()>,
    pub(crate) quit_handle: Arc<Mutex<bool>>,
    pub(crate) imgui_platform: WinitPlatform,
}

impl InputHandler {
    pub(crate) fn run(
        input_handler: Rc<RefCell<InputHandler>>,
        gui: Rc<RefCell<Gui>>,
        resources: &mut Resources,
    ) {
        let (
            renderer,
            mut runtime_config,
            mut input_state,
            mut input_actions,
            mut camera,
            mut resized,
        ) = resources
            .query::<(
                Res<RenderFrame>,
                ResMut<RuntimeConfiguration>,
                ResMut<InputState>,
                ResMut<InputActions>,
                ResMut<Camera>,
                ResMut<Resized>,
            )>()
            .unwrap();

        #[cfg(feature = "profiling")]
        microprofile::scope!("ecs", "InputHandler");
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
        imgui_platform
            .prepare_frame(gui.borrow_mut().imgui.io_mut(), &renderer.instance.window)
            .expect("Failed to prepare frame");
    }
}
