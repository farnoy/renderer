#version 450

#include "helpers/helper.glsl"

layout (set = 0, binding = 0) uniform sampler2D tex;

layout (location = 0) in vec4 color;
layout (location = 1) in vec2 uv;

layout (location = 0) out vec4 out_color;

void main() {
  out_color = color * texture(tex, uv.st) * test();
}
