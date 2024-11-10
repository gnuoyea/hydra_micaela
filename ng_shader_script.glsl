//neuroglancer shader script for rendering colormap

#uicontrol invlerp normalized
void main() {
    float value = normalized();
  	float threshold = 0.1;
  
  	//if the value is below the threshold, make transparent
    if (value < threshold) {
        discard;  //skip
    }

    //nonlinear transformation
    value = pow(value, 0.4);  //change exponent to change contrast

    vec3 color = vec3(0.0);

    //range of colors is based on value from data
    if (value < 0.33) {
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), value * 3.0); // blue -> green
    } else if (value < 0.66) {
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (value - 0.33) * 3.0); // green -> yellow
    } else {
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (value - 0.66) * 3.0); // yellow -> red (use this range)
    }

    emitRGB(color);
}
