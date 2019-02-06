#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp[4096];
};

layout(set = 1, binding = 0) uniform ModelBuffer {
    mat4 model[4096];
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec3 o_normal;
layout (location = 1) out vec3 o_world_pos;
layout (location = 2) out vec2 o_uv;

void main() {
    uint entity_id = gl_InstanceIndex;
    gl_Position = mvp[entity_id] * vec4(position, 1.0);
    o_world_pos = vec3(model[entity_id] * vec4(position, 1.0));
    o_normal = transpose(inverse(mat3(model[entity_id]))) * normal;
    o_uv = uv;
}
