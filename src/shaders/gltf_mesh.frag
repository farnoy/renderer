#version 460
#extension GL_EXT_nonuniform_qualifier: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require
// #extension GL_EXT_debug_printf : require
// #extension GL_ARB_sparse_texture2: require

#include "helpers/helper.glsl"

layout(constant_id = 10) const uint SHADOW_MAP_DIM = 4;
layout(constant_id = 11) const uint SHADOW_MAP_DIM_SQUARED = 4 * 4;

layout(set = 1, binding = 0, scalar) uniform CameraMatrices {
    mat4 projection;
    mat4 view;
    vec4 position;
    mat4 pv;
} camera;
layout(set = 2, binding = 0, scalar) readonly buffer LightMatrices {
    mat4 projection;
    mat4 view;
    vec4 position;
    mat4 pv;
} light_data[SHADOW_MAP_DIM_SQUARED];
layout(set = 2, binding = 1) uniform sampler2DShadow shadow_maps;
layout(set = 3, binding = 0) uniform sampler2D base_color[];
layout(set = 3, binding = 1) uniform sampler2D normal_map[];
#ifdef RT
layout(set = 4, binding = 0) uniform accelerationStructureEXT accelerationStructure;
layout(set = 4, binding = 1) uniform RandomSeed {
    uint seed;
} random;
#endif

layout (location = 0) in vec3 normal;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec4 tangent;
layout (location = 3) in flat uint entity_id;
layout (location = 4) in vec3 world_position;
layout (location = 5) in vec4 position_lightspace[2];
layout (location = 7) in flat uint draw_id;

layout (location = 0) out vec4 o_color;

float visibility_f(in vec3 eye_dir, in vec3 light_dir, in vec3 normal, in vec3 half_vector, in float alpha) {
    float a2 = pow(alpha, 2);
    float V = (dot(normal, eye_dir) + sqrt(a2 + (1 - a2) * pow(dot(normal, eye_dir), 2)));
    float L = (dot(normal, light_dir) + sqrt(a2 + (1 - a2) * pow(dot(normal, light_dir), 2)));
    return 0.5 / (V + L);
}

float microfacet_distribution_ggx(in vec3 normal, in vec3 half_vector, in float alpha) {
    float a2 = pow(alpha, 2);
    return a2 /** heaviside(dot(normal, half_vector)) */ / (radians(180) * pow(pow(dot(normal, half_vector), 2) * (a2 - 1) + 1, 2));
}

vec3 conductor_fresnel(in vec3 f0, in vec3 bsdf, in vec3 eye_dir, in vec3 half_vector) {
  return bsdf * (f0 + (1 - f0) * pow(1 - abs(dot(eye_dir, half_vector)), 5));
}

vec3 F_Schlick(in vec3 eye_dir, in vec3 half_vector, vec3 f0) {
    float f = pow(1.0 - dot(eye_dir, half_vector), 5.0);
    return f + f0 * (1.0 - f);
}

vec3 fresnel_mix(in float ior, in vec3 base, in vec3 layer, in vec3 eye_dir, in vec3 half_vector) {
  float f0 = pow((1 - ior) / (1 + ior), 2);
  float fr = f0 + (1 - f0) * pow(1 - abs(dot(eye_dir, half_vector)), 5);
  return mix(base, layer, fr);
}

// https://nvpro-samples.github.io/vk_mini_path_tracer/index.html#antialiasingandpseudorandomnumbergeneration
// Random number generation using pcg32i_random_t, using inc = 1. Our random state is a uint.
uint stepRNG(uint rngState)
{
  return rngState * 747796405 + 1;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState)
{
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  rngState  = stepRNG(rngState);
  uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
  word      = (word >> 22) ^ word;
  return float(word) / 4294967295.0f;
}

void main() {
    o_color = vec4(0, 0, 0, 1);

    vec3 normal_unit = normalize(normal);
    vec3 bitangent = cross(normalize(tangent.xyz), normal_unit) * tangent.w;
    vec3 sampled_normal = texture(normal_map[entity_id], uv).rgb * 2 - 1;
    mat3 TBN = transpose(mat3(normalize(tangent.xyz), bitangent, normal_unit));
    mat3 TBN_inverse = inverse(TBN);
    vec3 final_normal = (TBN_inverse * sampled_normal);
    final_normal = normalize(final_normal);

    #ifdef RT
    uvec2 rngSeed = uvec2(vec2(2000, 2000) * gl_FragCoord.xy);
    uint rngState = rngSeed.x ^ rngSeed.y ^ random.seed;
    #endif

    for (uint ix = 0; ix < 2; ix++) {
        // NOTE: Order of these next few operations around light_pos is critical
        vec3 light_pos = position_lightspace[ix].xyz / position_lightspace[ix].w;
        // negative viewport height
        light_pos.y *= -1.;
        // convert to NDC
        light_pos.xy *= .5;
        light_pos.xy += .5;

        // slice the shadow map atlas
        // columns first, then rows of a square SHADOW_MAP_DIM x SHADOW_MAP_DIM texture
        light_pos.x += float(ix % SHADOW_MAP_DIM);
        light_pos.y += float(ix / SHADOW_MAP_DIM);
        light_pos.xy /= float(SHADOW_MAP_DIM);

        vec3 light_dir = normalize(light_data[ix].position.xyz - world_position);
        float diff = max(dot(light_dir, final_normal), 0.5) * 1.25;
        // o_color.rgb *= diff;
        // o_color.rgb *= use_shadow && depth < 1.0 ? 0.6 : 1.0;

        float light_distance = sqrt(dot(light_data[ix].position.xyz - world_position, light_data[ix].position.xyz - world_position));
        const float max_light_distance = 100;
        const float light_strength = 60;
        vec3 color_light = vec3(pow(light_strength / max(light_distance, 0.01), 2));
        color_light *= window(light_distance, max_light_distance);

        float NdotL = max(dot(light_dir, final_normal), 0);
        // o_color.rgb += texture(base_color[entity_id], uv).rgb * color_light * NdotL;

        float shadow_multiplier = 1.0;
        #ifdef RT
        for (uint samples = 0; samples < 8; samples++) {
            vec3 rayTarget = light_data[ix].position.xyz;

            rayTarget.x -= stepAndOutputRNGFloat(rngState) * 2 - 1;
            rayTarget.y -= stepAndOutputRNGFloat(rngState) * 2 - 1;
            rayTarget.z -= stepAndOutputRNGFloat(rngState) * 2 - 1;

            rayQueryEXT rayQuery;
            rayQueryInitializeEXT(
                rayQuery,
                accelerationStructure,
                gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
                0xFF,
                world_position,
                0.01,
                rayTarget - world_position,
                distance(rayTarget, world_position)
            );

            while (rayQueryProceedEXT(rayQuery));

            if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
                shadow_multiplier *= 0.8;
        }
        #else
        bool use_shadow = light_pos == clamp(light_pos, vec3(0.0), vec3(1.0));
        float depth = texture(shadow_maps, vec3(light_pos.xy, light_pos.z));
        shadow_multiplier *= use_shadow && depth < 1.0 ? 0.2 : 1.0;
        #endif

        color_light *= shadow_multiplier;

        vec3 eye_dir = normalize(camera.position.xyz - world_position);
        vec3 half_vector = normalize(light_dir + eye_dir.xyz);
        float metallic = 0.2;
        float roughness = 0.7;
        roughness = pow(roughness, 2);
        float V = visibility_f(eye_dir, light_dir, final_normal, half_vector, roughness);

        float D = microfacet_distribution_ggx(final_normal, half_vector, roughness);
        vec3 F = F_Schlick(eye_dir, half_vector, vec3(0.04));
        vec3 specular_brdf = (vec3(V * D) * F); // / (4 * dot(final_normal, eye_dir) * dot(final_normal, light_dir));

        vec3 diffuse_term = texture(base_color[entity_id], uv).rgb / radians(180);

        vec3 dielectric = fresnel_mix(1.0, diffuse_term, specular_brdf, eye_dir, half_vector);
        vec3 metal = conductor_fresnel(texture(base_color[entity_id], uv).rgb, specular_brdf, eye_dir, half_vector);

        // o_color.rgb = normalize(eye_dir.xyz);
        // o_color.rgb = final_normal;
        o_color.rgb += color_light * NdotL * (diffuse_term + specular_brdf);
        // o_color.rgb += mix(dielectric, metal, metallic);
    }


    // if (rayQueryGetIntersectionTypeEXT(rayQuery, true) > 0)
    //     debugPrintfEXT("pos= %v3f, direction=%v3f, intersection result=%u", world_position, vec3(0, 5, 0) - world_position, rayQueryGetIntersectionTypeEXT(rayQuery, true));

    // if (abs(distance(world_position, vec3(1))) < 5.0) {
        // o_color.rgb = vec3(0.5);
    // }

    /*
    o_color = texture(base_color[entity_id], uv);
    vec4 ret;
    int code = sparseTextureARB(base_color[entity_id], uv, ret);
    if (sparseTexelsResidentARB(code))
        o_color = vec4(1.0);
    */
}
