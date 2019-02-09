#version 450

#extension GL_EXT_nonuniform_qualifier: require

layout(set = 1, binding = 0) uniform sampler2D base_color[];

layout (location = 0) in vec3 normal;
layout (location = 1) in vec2 uv;
layout (location = 2) in flat uint entity_id;
layout (location = 0) out vec4 o_color;

void main() {
    // debug UVs
    o_color.xy = uv;
    o_color.zw = vec2(0.0, 1.0);

    o_color = texture(base_color[entity_id], uv);    
    /*
    o_color.rgb = vec3(0.0);
    if (entity_id > 314)
        o_color.r = 1.0;
    o_color.a = 1.0;
    */
}
