#version 450

layout(set = 1, binding = 0) uniform sampler2D base_color;

layout (location = 0) in vec3 normal;
layout (location = 1) in vec2 uv;
layout (location = 0) out vec4 o_color;

void main() {
    // debug UVs
    o_color.xy = uv;
    o_color.zw = vec2(0.0, 1.0);

    o_color = texture(base_color, uv);    
}
