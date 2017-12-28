#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp;
    mat4 mv;
    mat4 normal_matrix;
} ubo;
layout(set = 0, binding = 1) uniform sampler2D s;
layout(set = 0, binding = 2) uniform sampler2D s_normal;

layout (location = 0) in vec2 texCoord;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec4 v_tangent;
layout (location = 3) in vec3 light_direction;
layout (location = 4) in vec3 eye_direction;

layout (location = 0) out vec4 uFragColor;

const vec3 LightColor = vec3(1.0, 1.0, 1.0) * 0.4;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 reflect_direction = reflect(light_direction, normal);
    float diffuse = max(dot(normal, light_direction), 0);
    float specular = max(dot(eye_direction, reflect_direction), 0);
    specular = pow(specular, 6);
    vec3 ambient = texture(s, texCoord).rgb;
    uFragColor.rgb = normalize(ambient + diffuse + specular * 0.5);
    uFragColor.a = 1;
}
