#version 450
#extension GL_EXT_nonuniform_qualifier: require
#extension GL_EXT_scalar_block_layout: require
// #extension GL_ARB_sparse_texture2: require

layout(constant_id = 10) const uint SHADOW_MAP_DIM = 4;
layout(constant_id = 11) const uint SHADOW_MAP_DIM_SQUARED = 4 * 4;

layout(set = 2, binding = 0, scalar) uniform LightMatrices {
    mat4 projection;
    mat4 view;
    vec4 position;
    mat4 pv;
} light_data[SHADOW_MAP_DIM_SQUARED];
layout(set = 2, binding = 1) uniform sampler2DShadow shadow_maps;
layout(set = 3, binding = 0) uniform sampler2D base_color[];

layout (location = 0) in vec3 normal;
layout (location = 1) in vec2 uv;
layout (location = 2) in flat uint entity_id;
layout (location = 3) in flat vec3 world_position;
layout (location = 4) in vec4 position_lightspace[2];
layout (location = 0) out vec4 o_color;

void main() {
    o_color = texture(base_color[entity_id], uv);

    for (uint ix = 0; ix < 2; ix++) {
        // NOTE: Order of these next few operations around light_pos is critical
        vec3 light_pos = position_lightspace[ix].xyz / position_lightspace[ix].w;
        // negative viewport height
        light_pos.y *= -1.;
        // convert to NDC
        light_pos.xy *= .5;
        light_pos.xy += .5;

        // check frustum intersection while in NDC
        bool use_shadow = light_pos == clamp(light_pos, vec3(0.0), vec3(1.0));

        // slice the shadow map atlas
        // columns first, then rows of a square SHADOW_MAP_DIM x SHADOW_MAP_DIM texture
        light_pos.x += float(ix % SHADOW_MAP_DIM);
        light_pos.y += float(ix / SHADOW_MAP_DIM);
        light_pos.xy /= float(SHADOW_MAP_DIM);

        vec3 light_dir = normalize(light_data[ix].position.xyz - world_position);
        float diff = max(dot(light_dir, normal), 0.5) * 1.25;
        o_color.rgb *= diff;
        float depth = texture(shadow_maps, vec3(light_pos.xy, light_pos.z));
        o_color.rgb *= use_shadow && depth < 1.0 ? 0.6 : 1.0;
    }
    /*
    o_color = texture(base_color[entity_id], uv);
    vec4 ret;
    int code = sparseTextureARB(base_color[entity_id], uv, ret);
    if (sparseTexelsResidentARB(code))
        o_color = vec4(1.0);
    */
}
