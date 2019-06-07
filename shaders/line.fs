#version 450 core

in vec3 line_color;
out vec4 frag_color;

void main(){
    frag_color = vec4(line_color, 1.0);
}
