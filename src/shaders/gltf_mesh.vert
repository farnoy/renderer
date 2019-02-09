#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp[4096];
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec3 o_normal;
layout (location = 1) out vec2 o_uv;
layout (location = 2) out uint o_entity_id;

void main() {
    uint entity_id = gl_InstanceIndex;
    gl_Position = mvp[entity_id] * vec4(position, 1.0);
    o_normal = normal;
    o_uv = uv;
    o_entity_id = entity_id;
}
