#version 450
#define SHADOW_MAP_DIM 4

layout(set = 0, binding = 0) uniform ModelMatrices {
    mat4 model[4096];
};
layout(set = 1, binding = 0) uniform CameraMatrices {
    mat4 projection;
    mat4 view;
    vec4 position;
} camera;
layout(set = 2, binding = 0) uniform LightMatrices {
    mat4 projection;
    mat4 view;
    vec4 position;
} light_data[SHADOW_MAP_DIM * SHADOW_MAP_DIM];

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec3 o_normal;
layout (location = 1) out vec2 o_uv;
layout (location = 2) out flat uint o_entity_id;
layout (location = 3) out vec3 o_world_pos;
layout (location = 4) out vec4 position_lightspace[2];

void main() {
    uint entity_id = gl_InstanceIndex;
    // https://paroj.github.io/gltut/Illumination/Tut09%20Normal%20Transformation.html
    o_normal = transpose(inverse(mat3(model[entity_id]))) * normal;
    o_world_pos = vec3(model[entity_id] * vec4(position, 1.0));
    gl_Position = camera.projection * camera.view * vec4(o_world_pos, 1.0);
    o_uv = uv;
    o_entity_id = entity_id;
    for (uint ix = 0; ix < 2; ix++) {
        // http://www.dissidentlogic.com/old/images/NormalOffsetShadows/GDC_Poster_NormalOffset.png
        vec3 to_light = normalize(light_data[ix].position.xyz - o_world_pos);
        float cos_light = dot(to_light, normal);
        float slope_scale = clamp(1 - cos_light, 0.0, 1.0);
        // TODO: tweak these
        float normal_offset = -1.;
        float slope_offset = 10. * slope_scale;
        vec3 shadow_position = o_world_pos + o_normal * (normal_offset + slope_offset);
        position_lightspace[ix] = light_data[ix].projection * light_data[ix].view * vec4(shadow_position, 1.0);
    }
}
