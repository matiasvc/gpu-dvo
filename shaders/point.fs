#version 450 core

in vec3 point_color;
out vec4 frag_color;

void main(){

    float a = 1.0;
    if ( length(gl_PointCoord.st - vec2(0.5, 0.5)) > 0.5 ){
        a = 0.0;
    }
    frag_color = vec4(point_color, a);
}
