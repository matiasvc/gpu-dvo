#version 450 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 color;
layout (location = 2) in float size;
out vec3 point_color;

uniform mat4 view;
uniform mat4 projection;

void main(){
    point_color = color;
    gl_PointSize = size;
    gl_Position = projection * view * vec4(pos, 1.0);
}