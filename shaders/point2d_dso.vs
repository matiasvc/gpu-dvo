#version 330 core
layout (location = 0) in vec2 pos;
layout (location = 1) in vec3 color;

flat out vec3 pointColor;

void main()
{
    gl_PointSize = 1.5;
	gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
    pointColor = color;
}
