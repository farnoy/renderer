#version 450

layout(set = 0, binding = 0) uniform ModelMatrices {
    mat4 model[4096];
};
layout(set = 1, binding = 0) uniform CameraMatrices {
    mat4 projection;
    mat4 view;
    vec4 pos;
    mat4 pv;
};
layout (location = 0) in vec3 position;

void main() {
    uint entity_id = gl_InstanceIndex;
    gl_Position = pv * model[entity_id] * vec4(position, 1.0);
}
