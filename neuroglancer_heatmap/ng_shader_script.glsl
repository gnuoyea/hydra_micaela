//shader script for neuroglancer rendering

//set normalization in ng to around 50 to take out gb
//set opacity of all heatmap layers to 1 (under rendering)

#uicontrol invlerp normalized

void main() {
    float value = normalized();
    float threshold = 0.23;
  
    //taking out the bg
    if (value < threshold) {
        discard; //if value below threshold, make transparent
    }
  
    //nonlinear transformation to adjust contrast
    value = pow(value, 1.0);  //can adjust exponent to adjust initial contrast
  
    vec3 color = vec3(0.0);

    //color transitions based on value in density map data
    if (value < 0.33) { //green -> yellow range (0.0 to 0.33)
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), value / 0.33);
    }
    else if (value < 0.66) { //yellow -> orange range (0.33 to 0.66)
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.25, 0.0), (value - 0.33) / 0.33);
    }
    else {
        // orange -> red range (0.66 to 1.0)
        color = mix(vec3(1.0, 0.25, 0.0), vec3(1.0, 0.0, 0.0), (value - 0.66) / 0.33);
    }
  
    //contrast increase to look more vibrant
    color = pow(color, vec3(1.8));
    
    //clamp color values to 0 -> 1 range to avoid overflow
    color = clamp(color, 0.0, 1.0);

    emitRGB(color);
}