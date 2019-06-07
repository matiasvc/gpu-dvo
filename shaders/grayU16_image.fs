#version 330 core
//out vec4 FragColor;
layout(location = 0) out vec3 color;

in vec2 TexCoord;

// texture samplers
uniform sampler2D grayTexture;

void main()
{
	color = texture(grayTexture, vec2(TexCoord.x, TexCoord.y)).rrr;
}