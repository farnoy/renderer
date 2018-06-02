#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp[1024];
};
layout (location = 0) in vec3 position;

void main() {
    uint entity_id = gl_InstanceIndex;
    gl_Position = mvp[entity_id] * vec4(position, 1.0);
}
