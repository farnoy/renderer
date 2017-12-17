use ash::vk;

use super::super::ExampleBase;
use super::super::buffer::Buffer;
use super::super::device::AshDevice;

pub struct TriangleMesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_type: vk::IndexType,
    pub index_count: u32,
}

impl TriangleMesh {
    pub fn dummy(base: &ExampleBase) -> TriangleMesh {
        #[derive(Copy, Clone)]
        struct Vertex {
            _pos: [f32; 2],
            _color: [f32; 3],
        }
        let vertices = vec![
            Vertex {
                _pos: [1.0, 1.0],
                _color: [1.0, 0.0, 0.0],
            },
            Vertex {
                _pos: [-1.0, 1.0],
                _color: [0.0, 1.0, 0.0],
            },
            Vertex {
                _pos: [0.0, -1.0],
                _color: [0.0, 0.0, 1.0],
            },
        ];
        let vertex_buffer = Buffer::upload_from::<Vertex, _>(
            base,
            vk::BUFFER_USAGE_VERTEX_BUFFER_BIT,
            &vertices.iter().cloned(),
        );

        let (index_buffer, index_type, index_count) = {
            let indices = vec![0 as u32, 1, 2];
            (
                Buffer::upload_from::<u32, _>(
                    base,
                    vk::BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    &indices.iter().cloned(),
                ),
                vk::IndexType::Uint32,
                3,
            )
        };

        TriangleMesh {
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
            index_type: index_type,
            index_count: index_count as u32,
        }
    }

    unsafe fn _free(self, device: &AshDevice) {
        self.vertex_buffer.free(device);
        self.index_buffer.free(device);
    }
}
