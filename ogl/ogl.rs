extern crate cgmath;
extern crate gl;
extern crate gltf;
extern crate gltf_importer;
extern crate gltf_utils;
extern crate glfw;

use std::ffi::CStr;
use gl::types::*;
use gltf_utils::PrimitiveIterators;
use glfw::{Action, Context, Key};
use std::os::raw::c_void;
use std::mem::{size_of, size_of_val};
use std::ptr;

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    glfw.window_hint(glfw::WindowHint::OpenGlDebugContext(true));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 5));
    glfw.window_hint(glfw::WindowHint::Resizable(true));

    // Create a windowed mode window and its OpenGL context
    let (mut window, events) = glfw.create_window(1280, 720, "Hello this is window", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.set_framebuffer_size_polling(true);
    // Make the window's context current
    window.make_current();
    window.set_key_polling(true);
    glfw.set_swap_interval(glfw::SwapInterval::Sync(1));

    // the supplied function must be of the type:
    // `&fn(symbol: &str) -> Option<extern "C" fn()>`
    unsafe {
        gl::load_with(|s| glfw.get_proc_address_raw(s) as *const c_void);

        gl::Enable(gl::DEBUG_OUTPUT);

        gl::DebugMessageCallback(debug_callback, ptr::null());
    }

    let program = unsafe {
        use std::fs::File;
        use std::io::Read;
        use std::path::PathBuf;
        let vertex_shader = {
            let shader = gl::CreateShader(gl::VERTEX_SHADER);
            let mut source = vec![];
            let mut lengths = vec![];
            lengths.push(
                File::open(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("basic.vert"))
                    .expect("failed to open triangle.vert")
                    .read_to_end(&mut source)
                    .expect("failed to read triangle.vert") as i32,
            );
            gl::ShaderSource(
                shader,
                1,
                &source.as_slice().as_ptr() as *const *const u8 as *const *const i8,
                lengths.as_slice().as_ptr(),
            );
            gl::CompileShader(shader);
            shader
        };
        let fragment_shader = {
            let shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            let mut source = vec![];
            let mut lengths = vec![];
            lengths.push(
                File::open(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("basic.frag"))
                    .expect("failed to open triangle.frag")
                    .read_to_end(&mut source)
                    .expect("failed to read triangle.frag") as i32,
            );
            gl::ShaderSource(
                shader,
                1,
                &source.as_slice().as_ptr() as *const *const u8 as *const *const i8,
                lengths.as_slice().as_ptr(),
            );
            gl::CompileShader(shader);
            shader
        };
        let program = gl::CreateProgram();
        gl::AttachShader(program, vertex_shader);
        gl::AttachShader(program, fragment_shader);
        gl::LinkProgram(program);
        program
    };

    let (ubo, ubo_mem) = unsafe {
        let mut buf = 0;
        gl::CreateBuffers(1, &mut buf);
        let size = size_of::<cgmath::Matrix4<f32>>() as isize;
        gl::NamedBufferStorage(
            buf,
            size,
            ptr::null(),
            gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT,
        );
        let mem = gl::MapNamedBufferRange(
            buf,
            0,
            size,
            gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_FLUSH_EXPLICIT_BIT | gl::MAP_UNSYNCHRONIZED_BIT,
        );
        (buf, mem)
    };

    let (vertex_buffer, index_buffer, index_len) = unsafe {
        // Mesh load
        let path = "../glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf";
        let importer = gltf_importer::import(path);
        let (loaded, buffers) = importer.unwrap();
        let mesh = loaded.meshes().next().unwrap();
        let primitive = mesh.primitives().next().unwrap();
        let vertices = primitive.positions(&buffers).unwrap().collect::<Vec<_>>();
        
        let mut vertex_buffer = 0;
        gl::CreateBuffers(1, &mut vertex_buffer);
        gl::NamedBufferStorage(
            vertex_buffer,
            (vertices.len() * size_of::<f32>() * 3) as isize,
            vertices.as_ptr() as *const GLvoid,
            0
        );

        let indices = PrimitiveIterators::indices(&primitive, &buffers).unwrap().into_u32().collect::<Vec<_>>();
        let mut index_buffer = 0;
        gl::CreateBuffers(1, &mut index_buffer);
        gl::NamedBufferStorage(
            index_buffer,
            (indices.len() * size_of::<u32>()) as isize,
            indices.as_ptr() as *const GLvoid,
            0
        );

        (vertex_buffer, index_buffer, indices.len())
    };

    let vao = unsafe {
        let mut vao = 0;
        gl::CreateVertexArrays(1, &mut vao);
        gl::EnableVertexArrayAttrib(vao, 0);
        gl::VertexArrayAttribBinding(vao, 0, 0);
        gl::VertexArrayAttribFormat(vao, 0, 3, gl::FLOAT, 0, 0);
        gl::VertexArrayVertexBuffer(vao, 0, vertex_buffer, 0, (size_of::<f32>() * 3) as i32);
        let name = CStr::from_bytes_with_nul(b"main vao\0").unwrap();
        gl::ObjectLabel(gl::VERTEX_ARRAY, vao, name.to_bytes().len() as i32, name.as_ptr());
        vao
    };


    let mut ix = 0;

    // Loop until the user closes the window
    while !window.should_close() {
        // Swap front and back buffers
        window.swap_buffers();

        let (width, height) = window.get_size();

        let projection = cgmath::perspective(cgmath::Deg(60.0), width as f32 / height as f32, 0.1, 10.0);
        let view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(0.0, 0.5, -2.0),
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::vec3(0.0, 1.0, 0.0),
        );
        let model = cgmath::Matrix4::from_angle_y(cgmath::Deg(ix as f32)) * cgmath::Matrix4::from_scale(0.3);
        let mvp = projection * view * model;

        unsafe {
            *(ubo_mem as *mut cgmath::Matrix4<f32>) = mvp;
            gl::FlushMappedNamedBufferRange(ubo, 0, size_of_val(&mvp) as isize);
        }

        unsafe {
            let k: Vec<u8> = "Draw".into();
            gl::PushDebugGroup(gl::DEBUG_SOURCE_APPLICATION, gl::DEBUG_SEVERITY_NOTIFICATION, k.len() as i32, k.as_ptr() as *const i8);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::UseProgram(program);
            gl::BindVertexArray(vao);
            gl::BindBufferBase(gl::UNIFORM_BUFFER, 0, ubo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer);
            gl::DrawElements(gl::TRIANGLES, index_len as i32, gl::UNSIGNED_INT, ptr::null());
            gl::PopDebugGroup();
        }

        ix += 1;
        // Poll for and process events
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            println!("{:?}", event);
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
                glfw::WindowEvent::FramebufferSize(w, h) => unsafe { gl::Viewport(0, 0, w, h) },
                _ => {}
            }
        }
    }
}

extern "system" fn debug_callback(_source: GLenum, _type: GLenum, _id: GLuint, _severity: GLenum, _length: GLsizei, message: *const GLchar, _userdata: *mut GLvoid) {
    unsafe {
        let cstr = CStr::from_ptr(message);
        println!("some error {:?}", cstr);
    }
}
