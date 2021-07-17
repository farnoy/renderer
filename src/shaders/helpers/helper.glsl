vec4 test() {
    return vec4(1.0);
}

float saturate(in float x) {
    return clamp(x, 0, 1);
}

float window(in float r, in float r_max) {
    return pow(max(1 - pow(r / r_max, 4), 0), 2);
}

float fresnel(in float F0, in vec3 n, in vec3 l) {
    return mix(F0, pow(1 - saturate(dot(n, l)), 5), F0);
}

float heaviside(in float x) {
    return x > 0 ? 1 : 0;
}
