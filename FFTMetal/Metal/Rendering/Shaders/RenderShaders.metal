/*
 Copyright (C) 2015 Apple Inc. All Rights Reserved.
 See LICENSE.txt for this sample’s licensing information
 
 Abstract:
 Environment mapping shader that mixes several textures.
 */


#include <metal_graphics>
#include <metal_matrix>
#include <metal_geometric>
#include <metal_math>
#include <metal_texture>
#include <metal_stdlib>

#include "MetalSharedTypes.h"

using namespace metal;
//using namespace MetalRenderState;


#pragma mark -- Cube Texture Mapping Vertex Buffer

struct CubeVertexOutput
{
    float4 position [[position]];
    float3 texCoords;
};

vertex CubeVertexOutput skyboxVertex(constant float4 *pos_data [[ buffer(SKYBOX_VERTEX_BUFFER) ]],
                                     constant float4 *texcoord [[ buffer(SKYBOX_TEXCOORD_BUFFER) ]],
                                     constant MetalRenderState::uniforms_t& uniforms [[ buffer(SKYBOX_CONSTANT_BUFFER) ]],
                                     uint vid [[vertex_id]])
{
    CubeVertexOutput out;
    out.position = uniforms.skybox_modelview_projection_matrix * pos_data[vid];
    out.texCoords = texcoord[vid].xyz;
    return out;
}

fragment half4 skyboxFragment(CubeVertexOutput in [[stage_in]],
                              texturecube<half> skybox_texture [[texture(SKYBOX_IMAGE_TEXTURE)]])
{
    constexpr sampler s_cube(filter::linear, mip_filter::linear);
    return skybox_texture.sample(s_cube, in.texCoords);
}


struct MetalRenderVertexOutput
{
    float4 position [[position]];
    float4 eye;
    half4 color;
    float3 eye_normal;
    float3 normal;
    float2 uv;
};

typedef struct
{
    packed_float3 position;
    packed_float3 normal;
    packed_float2 texCoord;
} vertex_t;

constant float4 copper_ambient = float4(0.19125f, 0.0735f, 0.0225f, 1.0f);
constant float4 copper_diffuse = float4(0.7038f, 0.27048f, 0.0828f, 1.0f);
constant float3 light_position = float3(0.0f, 1.0f, -1.0f);

/*
 *  Vertex Shader:  Render to 3D Space Quad Vertices
 */
vertex MetalRenderVertexOutput renderQuadVertex(device vertex_t* vertex_array [[ buffer(QUAD_VERTEX_BUFFER) ]],
                                            constant MetalRenderState::uniforms_t& uniforms [[ buffer(QUAD_VERTEX_CONSTANT_BUFFER) ]],
                                            uint vid [[vertex_id]])
{
    // get per vertex data
    float3 position = float3(vertex_array[vid].position);
    //float3 normal = float3(vertex_array[vid].normal);
    float2 uv = float2(vertex_array[vid].texCoord);
    
    // output transformed geometry data
    MetalRenderVertexOutput out;
    out.position = /*uniforms.modelview_projection_matrix */ float4(position, 1.0);
    //out.normal = normalize(uniforms.normal_matrix * float4(normal, 0.0)).xyz;
    
    // fix the uv's to fit the video camera's coordinate system and the device's orientation

    //assume the texture is being passed with the correctly ordered dimensions
    //based on its orientation
    
    switch (uniforms.orientation)
    {
        case MetalRenderState::PortraitUpsideDown:
            out.uv.x = 1.0f - uv.y;
            out.uv.y = uv.x;
            break;
        case MetalRenderState::Portrait:
            out.uv.x = uv.x;
            out.uv.y = 1.0f - uv.y;
            break;
        case MetalRenderState::LandscapeLeft:
            out.uv.x = 1.0 - uv.y;
            out.uv.y = 1.0 - uv.x;
            break;
        case MetalRenderState::LandscapeRight:
            out.uv.x = uv.y;
            out.uv.y = uv.x;
            break;
        default:
            out.uv.x = 0;
            out.uv.y = 0;
            break;
    }
    
    
    //rearrange the texture coordinates based on the device coordinates
    //and the orientatin of the incoming buffer
    /*
    switch (uniforms.orientation)
    {
        case MetalRenderState::PortraitUpsideDown:
            out.uv.x = 1.0f - uv.x;
            out.uv.y = 1.0f - uv.y;
            break;
        case MetalRenderState::Portrait:
            out.uv.x = uv.x;
            out.uv.y = uv.y;
            break;
        case MetalRenderState::LandscapeLeft:
            out.uv.x = 1.0f - uv.y;
            out.uv.y = uv.x;
            break;
        case MetalRenderState::LandscapeRight:
            out.uv.x = uv.y;
            out.uv.y = 1.0f - uv.x;
            break;
        default:
            out.uv.x = 0;
            out.uv.y = 0;
            break;
    }
    */
    // calculate the incident vector and normal vectors for reflection in the quad's modelview space
    //out.eye = normalize(uniforms.modelview_matrix * float4(position, 1.0));
    //out.eye_normal = normalize(uniforms.modelview_matrix * float4(normal, 0.0)).xyz;
    
    // calculate diffuse lighting with the material color
    //float n_dot_l = dot(out.normal, normalize(light_position));
    //n_dot_l = fmax(0.0, n_dot_l);
    out.color = half4(1, 0, 0, 1);//half4(copper_ambient) + half4(copper_diffuse * n_dot_l);
    
    return out;
}

/*
 *  Fragment Shader:  Render a Metal Texture to a 3D Space Quad
 */
fragment half4 renderQuadFragment(MetalRenderVertexOutput in [[stage_in]],
                                   /*texturecube<half> env_tex [[ texture(QUAD_ENVMAP_TEXTURE) ]],*/
                                   texture2d<half> tex [[ texture(QUAD_IMAGE_TEXTURE) ]],
                                   constant MetalRenderState::uniforms_t& uniforms [[ buffer(QUAD_FRAGMENT_CONSTANT_BUFFER) ]])
{
    // get reflection vector
    //float3 reflect_dir = reflect(in.eye.xyz, in.eye_normal);
    
    // return reflection vector to world space
    //float4 reflect_world = uniforms.inverted_view_matrix * float4(reflect_dir, 0.0);
    
    // use the inverted reflection vector to sample from the cube map
    //constexpr sampler s_cube(filter::linear, mip_filter::linear);
    //half4 tex_color = env_tex.sample(s_cube, reflect_world.xyz);
    
    // sample from the 2d textured quad as well
    constexpr sampler s_quad(filter::linear);
    half4 image_color = tex.sample(s_quad, in.uv);
    half4 firstColor = tex.read(uint2(0,0));
    
    //float magLogN = log( length( firstColor.rg ) + 1.0f );
    //float magLog = log( length( image_color.rg ) + 1.0f );

    //float out = magLog/(magLogN*2.0f);
    //half4 outColor = half4(out, out, out, 1);
    // combine with texture, light, and envmap reflaction
    //half4 color = mix(in.color, image_color, 0.9h);
    //color = mix(tex_color, color, 0.6h);
    
    // RGB to grayscale
    //half color = dot(image_color.rgb, half3(0.30h, 0.59h, 0.11h));
    //half4 color = image_color;
    
    //half4 outColor = half4(color, color, color, 1.0);
    
    float color = image_color.r ;/// firstColor.r;
    half4 outColor = image_color;//half4(color, color, color, 1.0);
    
    //return image_color;
    return outColor;
}

/*
 *  Fragment Shader:  Render a float2 buffer to a 3D Space Quad
 */
fragment half4 renderBufferToQuadFragment(MetalRenderVertexOutput in [[stage_in]],
                                  const device float2 * complexBuffer [[ buffer(0) ]],
                                  float4 gid [[position]],
                                  uint sampleID [[ sample_id ]],
                                  float2 pointCoord [[ point_coord]] )
{
    // get reflection vector
    //float3 reflect_dir = reflect(in.eye.xyz, in.eye_normal);
    
    // return reflection vector to world space
    //float4 reflect_world = uniforms.inverted_view_matrix * float4(reflect_dir, 0.0);
    
    // use the inverted reflection vector to sample from the cube map
    //constexpr sampler s_cube(filter::linear, mip_filter::linear);
    //half4 tex_color = env_tex.sample(s_cube, reflect_world.xyz);
    
    // sample from the 2d textured quad as well
    //constexpr sampler s_quad(filter::linear);
    
    float columnIndex = gid.x/1024.f * 512.f;
    float rowIndex = gid.y/1366.f * 512.f;
    
    uint column = columnIndex;
    uint row = rowIndex * 512;
    
    float2 complexValue = complexBuffer[row + column];
    
    half4 image_color = half4( complexValue.x, complexValue.x, complexValue.x, 1);
    
    //half4 image_color = half4(1, 0, 0, 1);
    //half4 image_color = tex.sample(s_quad, in.uv);
    
    // combine with texture, light, and envmap reflaction
    //half4 color = mix(in.color, image_color, 0.9h);
    //color = mix(tex_color, color, 0.6h);
    
    // RGB to grayscale
    //half color = dot(image_color.rgb, half3(0.30h, 0.59h, 0.11h));
    //half4 color = image_color;
    
    //half4 outColor = half4(color, color, color, 1.0);
    
    
    if( gid.x > 1024 )
        image_color = half4( 1, 0, 0, 1 );
    
    if( gid.y > 1366 )
        image_color = half4( 0,0,1,1);
    
    //return outColor;
    return image_color;
}

vertex MetalRenderVertexOutput reflectQuadVertex(device vertex_t* vertex_array [[ buffer(QUAD_VERTEX_BUFFER) ]],
                                            constant MetalRenderState::uniforms_t& uniforms [[ buffer(QUAD_VERTEX_CONSTANT_BUFFER) ]],
                                            uint vid [[vertex_id]])
{
    // get per vertex data
    float3 position = float3(vertex_array[vid].position);
    float3 normal = float3(vertex_array[vid].normal);
    float2 uv = float2(vertex_array[vid].texCoord);
    
    // output transformed geometry data
    MetalRenderVertexOutput out;
    out.position = uniforms.modelview_projection_matrix * float4(position, 1.0);
    out.normal = normalize(uniforms.normal_matrix * float4(normal, 0.0)).xyz;
    
    // fix the uv's to fit the video camera's coordinate system and the device's orientation
    switch (uniforms.orientation)
    {
        case MetalRenderState::PortraitUpsideDown:
            out.uv.x = 1.0f - uv.y;
            out.uv.y = uv.x;
            break;
        case MetalRenderState::Portrait:
            out.uv.x = uv.y;
            out.uv.y = 1.0f - uv.x;
            break;
        case MetalRenderState::LandscapeLeft:
            out.uv.x = 1.0f - uv.x;
            out.uv.y = 1.0f - uv.y;
            break;
        case MetalRenderState::LandscapeRight:
            out.uv.x = uv.x;
            out.uv.y = uv.y;
            break;
        default:
            out.uv.x = 0;
            out.uv.y = 0;
            break;
    }
    
    // calculate the incident vector and normal vectors for reflection in the quad's modelview space
    out.eye = normalize(uniforms.modelview_matrix * float4(position, 1.0));
    out.eye_normal = normalize(uniforms.modelview_matrix * float4(normal, 0.0)).xyz;
    
    // calculate diffuse lighting with the material color
    float n_dot_l = dot(out.normal, normalize(light_position));
    n_dot_l = fmax(0.0, n_dot_l);
    out.color = half4(copper_ambient) + half4(copper_diffuse * n_dot_l);
    
    return out;
}

fragment half4 reflectQuadFragment(MetalRenderVertexOutput in [[stage_in]],
                                   texturecube<half> env_tex [[ texture(QUAD_ENVMAP_TEXTURE) ]],
                                   texture2d<half> tex [[ texture(QUAD_IMAGE_TEXTURE) ]],
                                   constant MetalRenderState::uniforms_t& uniforms [[ buffer(QUAD_FRAGMENT_CONSTANT_BUFFER) ]])
{
    // get reflection vector
    float3 reflect_dir = reflect(in.eye.xyz, in.eye_normal);
    
    // return reflection vector to world space
    float4 reflect_world = uniforms.inverted_view_matrix * float4(reflect_dir, 0.0);
    
    // use the inverted reflection vector to sample from the cube map
    constexpr sampler s_cube(filter::linear, mip_filter::linear);
    half4 tex_color = env_tex.sample(s_cube, reflect_world.xyz);
    
    // sample from the 2d textured quad as well
    constexpr sampler s_quad(filter::linear);
    half4 image_color = tex.sample(s_quad, in.uv);
    
    // combine with texture, light, and envmap reflaction
    half4 color = mix(in.color, image_color, 0.9h);
    color = mix(tex_color, color, 0.6h);
    
    return color;
}
