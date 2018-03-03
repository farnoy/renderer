#version 450

layout (push_constant) uniform PushConstant {
    vec2 positions[3];
} pushConstants;

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp;
} ubo;

layout (location = 0) out vec3 o_color;

void main() {
    o_color = normalize(vec3(gl_VertexIndex, 3.0, 0.5));
    gl_Position = vec4(pushConstants.positions[gl_VertexIndex], 0.0, 1.0) * ubo.mvp;
}