#version 450

layout (set = 0, binding = 0) uniform sampler2D tex;

layout (location = 0) in struct {
    vec4 color;
    vec2 uv;
} In;

layout (location = 0) out vec4 color;

void main() {
  color = In.color * texture(tex, In.uv.st);
}
