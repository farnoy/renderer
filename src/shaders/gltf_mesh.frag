#version 450
// TODO: define in shadow compiler
#define SHADOW_MAP_DIM 4

#extension GL_EXT_nonuniform_qualifier: require

layout(set = 1, binding = 0) uniform sampler2D base_color[];
layout(set = 2, binding = 1) uniform sampler2DShadow shadow_maps;

layout (location = 0) in vec3 normal;
layout (location = 1) in vec2 uv;
layout (location = 2) in flat uint entity_id;
layout (location = 3) in vec3 world_position;
layout (location = 4) in vec4 in_light_pos[2];
layout (location = 0) out vec4 o_color;

void main() {
    o_color = texture(base_color[entity_id], uv);

    for (uint ix = 0; ix < 2; ix++) {
        vec3 light_pos = in_light_pos[ix].xyz / in_light_pos[ix].w;
        // negative viewport height
        light_pos.y *= -1.;
        // convert to NDC
        light_pos *= .5;
        light_pos += .5;
        light_pos.xy /= SHADOW_MAP_DIM;
        // slice the shadow map atlas
        light_pos.x += float(ix) / SHADOW_MAP_DIM;

        float depth = texture(shadow_maps, vec3(light_pos.xy, light_pos.z - .14));
        o_color.rgb *= clamp(depth, .3, 1.);
    }
}
