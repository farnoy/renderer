#version 450
// TODO: define in shadow compiler
#define SHADOW_MAP_DIM 4

#extension GL_EXT_nonuniform_qualifier: require

layout(set = 2, binding = 0) buffer readonly LightMatrices {
    mat4 projection;
    mat4 view;
    vec4 position;
} light_data[SHADOW_MAP_DIM * SHADOW_MAP_DIM];
layout(set = 2, binding = 1) uniform sampler2DShadow shadow_maps;
layout(set = 3, binding = 0) uniform sampler2D base_color[];

layout (location = 0) in vec3 normal;
layout (location = 1) in vec2 uv;
layout (location = 2) in flat uint entity_id;
layout (location = 3) in vec3 world_position;
layout (location = 4) in vec4 position_lightspace[2];
layout (location = 0) out vec4 o_color;

void main() {
    o_color = texture(base_color[entity_id], uv);

    for (uint ix = 0; ix < 2; ix++) {
        vec3 light_pos = position_lightspace[ix].xyz;
        // slice the shadow map atlas
        light_pos.xy /= float(SHADOW_MAP_DIM);
        light_pos.x += float(ix) / SHADOW_MAP_DIM;

        vec3 light_dir = normalize(light_data[ix].position.xyz - world_position);
        float diff = max(dot(light_dir, normal), 0.5) * 1.25;

        float depth = texture(shadow_maps, vec3(light_pos.xy, light_pos.z)); //, light_pos.z - 0.2));
        o_color.rgb *= diff;
        o_color.rgb *= depth < 1.0 ? 0.6 : 1.0;
    }
}
