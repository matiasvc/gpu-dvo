#version 330 core
//out vec4 FragColor;
layout(location = 0) out vec3 color;

in vec2 TexCoord;

// texture samplers
uniform sampler2D grayTexture;

float colormap_red(float x) {
	if (x < 0.7) {
		return 4.0 * x - 1.5;
	} else {
		return -4.0 * x + 4.5;
	}
}

float colormap_green(float x) {
	if (x < 0.5) {
		return 4.0 * x - 0.5;
	} else {
		return -4.0 * x + 3.5;
	}
}

float colormap_blue(float x) {
	if (x < 0.3) {
		return 4.0 * x + 0.5;
	} else {
		return -4.0 * x + 2.5;
	}
}

vec3 colorMap(float x) {
	float r = clamp(colormap_red(x), 0.0, 1.0);
	float g = clamp(colormap_green(x), 0.0, 1.0);
	float b = clamp(colormap_blue(x), 0.0, 1.0);
	return vec3(r, g, b);
}

void main()
{
	color = colorMap(texture(grayTexture, TexCoord).r*5);
}