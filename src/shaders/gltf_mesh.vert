#version 450

#extension GL_ARB_shader_draw_parameters: require

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp[1024];
} ubo;

layout (location = 0) in vec3 position;

void main() {
    uint entity_id = gl_DrawIDARB + gl_InstanceIndex; 
    gl_Position = ubo.mvp[entity_id] * vec4(position, 1.0);
}
