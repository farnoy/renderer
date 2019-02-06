#version 450

layout(set = 2, binding = 0) uniform sampler2D base_color;

layout (location = 0) in vec3 normal;
layout (location = 1) in vec3 world_pos;
layout (location = 2) in vec2 uv;
layout (location = 0) out vec4 o_color;

void main() {
    const vec3 light_pos = {0.0, 3.0, 0.0};
    const vec3 light_cone_dir = {0.0, -1.0, 0.0};
    const vec3 view_pos = {0.0, 1.0, -2.0};
    const float specular_strength = 2.0;
    const vec3 light_color = vec3(1.0);

    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(light_pos - world_pos);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * light_color;

    vec3 view_dir = normalize(view_pos - world_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    vec3 specular = specular_strength * spec * light_color;

    o_color = vec4(diffuse + specular, 1.0);

    // normal
    o_color = vec4(norm, 1.0);

    // gooch
    const vec3 cool = {0, 0, 0.55};
    const vec3 warm = {0.3, 0.3, 0};
    const vec3 highlight = vec3(1.0);
    float t = (dot(norm, light_dir) + 1.0) / 2;
    float s = clamp(100 * dot(reflect_dir, view_dir) - 97, 0, 1);
    o_color.xyz = mix(mix(cool, warm, t), highlight, s);

    if (dot(normalize(world_pos - light_pos), light_cone_dir) >= 0.6)
        o_color.xyz = vec3(240,230,60) / 255;

    // debug UVs
    o_color.xy = uv;
    o_color.zw = vec2(0.0, 1.0);

    o_color = texture(base_color, uv);    
}
