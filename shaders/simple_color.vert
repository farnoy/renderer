#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp;
    mat4 mv;
    mat4 normal_matrix;
} ubo;

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec4 tangent;

layout (location = 0) out vec2 texCoord;
layout (location = 1) out vec3 v_normal;
layout (location = 2) out vec4 v_tangent;
layout (location = 3) out vec3 light_direction;
layout (location = 4) out vec3 eye_direction;

const vec3 light_position_world = vec3(-1, 0.3, 1);

void main() {
    eye_direction = (ubo.mv * vec4(position, 1)).xyz;
    vec3 light_position = (ubo.mv * vec4(light_position_world, 1)).xyz;
    texCoord = uv;
    vec3 n = normalize(ubo.normal_matrix * vec4(normal, 1)).xyz;
    vec3 t = normalize(ubo.normal_matrix * tangent).xyz;
    vec3 b = cross(n, t);
    mat3 tbn = transpose(mat3(
        t,
        b,
        n
    ));
    vec3 v;
    /*
    v.x = dot(light_position, t);
    v.y = dot(light_position, b);
    v.z = dot(light_position, n);
    light_direction = normalize(v);
    */
    light_direction = tbn * light_position;
    /*
    v.x = dot(eye_direction, t);
    v.y = dot(eye_direction, b);
    v.z = dot(eye_direction, n);
    eye_direction = normalize(v);
    */
	vec3 vertexPosition_cameraspace = (ubo.mv * vec4(position, 1)).xyz;
    eye_direction = vec3(0,0,0) - vertexPosition_cameraspace;
    eye_direction *= tbn;
    gl_Position = ubo.mvp * vec4(position, 1.0);
    v_normal = normal;
    v_tangent = tangent;
}
