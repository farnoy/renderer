#version 450

#extension GL_EXT_scalar_block_layout: require

layout(push_constant, scalar) uniform PushConstants {
    vec2 scale;
    vec2 translate;
} pushConstants;

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec4 col;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec2 out_uv;

void main() {
  out_color = col;
  out_uv = uv;
  gl_Position = vec4(pos * pushConstants.scale + pushConstants.translate, 0, 1);
  gl_Position.y *= -1.0;
}
