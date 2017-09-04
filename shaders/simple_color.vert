#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp;
} ubo;
// layout(push_constant) uniform PushConstants {
    // int entityId;
// } pushConstants; 

layout (location = 0) in vec3 position;
// layout (location = 1) in vec2 uv;
// layout (location = 0) out vec2 texCoord;

void main() {
    // texCoord = uv;
    gl_Position = ubo.mvp * vec4(position, 1.0);
}

