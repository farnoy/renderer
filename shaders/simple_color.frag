#version 450

layout(set = 1, binding = 0) uniform sampler2D s;

layout (location = 0) in vec2 texCoord;
layout (location = 0) out vec4 uFragColor;

void main() {
    uFragColor = texture(s, texCoord);
}
