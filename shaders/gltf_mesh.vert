#version 450

#extension GL_ARB_shader_draw_parameters: require

// layout (push_constant) uniform PushConstant {
    // uint entity_id;
// } pushConstants;

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp[1024];
} ubo;

layout (location = 0) in vec3 position;
// layout (location = 0) out float o_id;

void main() {
    // o_id = float(pushConstants.entity_id);
    // uint entity_id = pushConstants.entity_id; 
    uint entity_id = gl_DrawIDARB; 
    gl_Position = ubo.mvp[entity_id] * vec4(position, 1.0);
}