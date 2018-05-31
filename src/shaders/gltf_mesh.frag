#version 450

layout (location = 0) in vec3 normal;
layout (location = 1) in vec3 world_pos;
layout (location = 0) out vec4 o_color;

void main() {
    vec3 light_pos = {0.0, 1.0, -2.0};
    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(light_pos - world_pos);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * vec3(0.0, 1.0, 0.0);
    o_color = vec4(diffuse, 1.0);
}
