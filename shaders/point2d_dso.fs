#version 330 core
layout(location = 0) out vec3 color;

flat in vec3 pointColor;


void main()
{
    color = pointColor;
}
