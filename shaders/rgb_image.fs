#version 330 core
//out vec4 FragColor;
layout(location = 0) out vec3 color;

in vec2 TexCoord;

// texture samplers
uniform sampler2D rgbTexture;

void main()
{
	color = texture(rgbTexture, TexCoord).rgb;
}