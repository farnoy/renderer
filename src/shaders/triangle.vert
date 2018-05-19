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
    vec2 positions[3] = pushConstants.positions;
    gl_Position = ubo.mvp * vec4(positions[gl_VertexIndex%3], 0.0, 1.0);
}