#version 450

layout(set = 0, binding = 1) uniform sampler2D s;

layout (location = 0) in vec2 texCoord;
layout (location = 0) out vec4 uFragColor;

void main() {
    uFragColor = texture(s, texCoord);
}
