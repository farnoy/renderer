#version 450

layout(set = 1, binding = 0) uniform sampler2D s;
layout(set = 1, binding = 1) uniform texture2D t;
layout(push_constant) uniform PushConstants {
    int entityId;
} pushConstants; 

layout (location = 0) in vec2 texCoord;
layout (location = 0) out vec4 uFragColor;

void main() {
    uFragColor = texture(s, texCoord);
}
