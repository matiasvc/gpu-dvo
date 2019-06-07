#version 330 core
//out vec4 FragColor;
layout(location = 0) out vec3 color;

in vec2 TexCoord;

uniform float gradientStrength = 15.0;
uniform sampler2D grayTexture;

void main()
{
	color = gradientStrength*(texture(grayTexture, vec2(TexCoord.x, TexCoord.y)).rrr - 0.5) + 0.5;
}