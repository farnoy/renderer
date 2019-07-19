#version 450

#extension GL_EXT_nonuniform_qualifier: require

layout(set = 1, binding = 0) uniform sampler2D base_color[];

layout (location = 0) in vec3 normal;
layout (location = 1) in vec2 uv;
layout (location = 2) in flat uint entity_id;
layout (location = 0) out vec4 o_color;

void main() {
    o_color = texture(base_color[entity_id], uv);
}
