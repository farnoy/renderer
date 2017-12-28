#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout (location = 0) in vec2 texCoord[3];
layout (location = 1) in vec3 v_PositionCS[3];
layout (location = 2) in vec3 v_normal[3];
layout (location = 3) in vec4 v_tangent[3];
layout (location = 0) out vec2 out_texCoord;
layout (location = 1) out vec3 out_v_PositionCS;
layout (location = 2) out vec3 out_v_normal;
layout (location = 3) out vec4 out_v_tangent;
layout (location = 4) out vec3 out_my_normal;

vec3 surface_normal(vec3 triangle0, vec3 triangle1, vec3 triangle2) {
    vec3 u = triangle1 - triangle0;
    vec3 v = triangle2 - triangle0;

    return normalize(cross(u, v));
}

void main() {
    out_my_normal = surface_normal(
        gl_in[0].gl_Position.xyz,
        gl_in[1].gl_Position.xyz,
        gl_in[2].gl_Position.xyz
    );
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        out_texCoord = texCoord[i];
        out_v_PositionCS = v_PositionCS[i];
        out_v_normal = v_normal[i];
        out_v_tangent = v_tangent[i];
        EmitVertex();
    }
    EndPrimitive();
}
