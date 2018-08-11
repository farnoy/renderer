#version 450

layout (location = 0) in vec3 normal;
layout (location = 1) in vec3 world_pos;
layout (location = 0) out vec4 o_color;

void main() {
    const vec3 light_pos = {0.0, 3.0, 0.0};
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
    o_color = vec4(1.0);
}
