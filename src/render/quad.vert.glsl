#version 300 es
precision highp float;

in vec2 a_localPos;
in vec2 a_worldPos;
in vec2 a_worldSize;
in vec4 a_color;

uniform vec2 u_viewportPx;
uniform float u_dpr;
uniform float u_scale;
uniform vec2 u_offsetPx;

out vec4 v_color;

void main() {
    vec2 world = a_worldPos + (a_localPos * a_worldSize);
    vec2 screenCss = u_offsetPx + (world * u_scale);
    vec2 screenPx = screenCss * u_dpr;
    vec2 clip = (screenPx / u_viewportPx) * 2.0 - 1.0;
    clip.y = -clip.y;
    gl_Position = vec4(clip, 0.0, 1.0);
    v_color = a_color;
}
