#version 450

layout(binding = 0, std140) uniform UBO {
    mat4 mvp;
};

layout (location = 0) in vec3 pos;

void main() {
    gl_Position = mvp * vec4(pos, 1.0);
}