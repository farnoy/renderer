#version 450

layout (push_constant) uniform PushConstant {
    vec2 positions[3];
} pushConstants;

layout (location = 0) out vec3 o_color;

void main() {
    o_color = normalize(vec3(gl_VertexIndex, 3.0, 0.5));
    gl_Position = vec4(pushConstants.positions[gl_VertexIndex], 0.5, 1.0);
}