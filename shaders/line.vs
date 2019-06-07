#version 450 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 color;
out vec3 line_color;

uniform mat4 view;
uniform mat4 projection;

void main(){
    line_color = color;
    gl_Position = projection * view * vec4(pos, 1.0);
}
