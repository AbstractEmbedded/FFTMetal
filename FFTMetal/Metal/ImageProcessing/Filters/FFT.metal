//
//  FFT.metal
//  FFTMetal
//
//  Created by MACMaster on 12/17/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct FFTImageFilterUniforms
{
    unsigned int signalLength;
    unsigned int kernelSize;
    unsigned int fftStage;

};

struct ThreadGroupBuffer
{
    unsigned int threadGroupIndex;
    unsigned int threadGroupParam1;
    unsigned int threadGroupParam2;
    unsigned int threadGroupParam3;
    
};

#pragma mark -- FFT Kernel Definitions

#define PI  3.14159265f

//the 1D fft component calculation is defined by recursively splitting the image into 2 sets G and H of N/2 even/odd samples,
//calculating the FFT on each set and multiply by a complex twiddle factor, (FFT(G) + twiddle*FFT(H).  Here, we define the complex twiddle factor, twiddle.

#define TWIDDLE_FACTOR(k, angle, in){ \
    float2 tw, v;\
    tw.x = fast::cos(k*angle);\
    tw.y = fast::sin(k*angle);\
    v.x = tw.x*in.x;\
    v.x -= tw.y*in.y;\
    v.y = tw.x*in.y + tw.y*in.x;\
    in.x = v.x;\
    in.y = v.y;\
}



#pragma mark -- 8 Point FFT Kernel

#define FFT_8(in0, in1, in2, in3, in4, in5, in6, in7) {\
float2 v0;\
v0 = in0;\
in0 = v0 + in4;\
in4 = v0 - in4;\
v0 = in2;\
in2 = v0 + in6;\
in6 = v0 - in6;\
v0 = in1;\
in1 = v0 + in5;\
in5 = v0 - in5;\
v0 = in3;\
in3 = v0 + in7;\
in7 = v0 - in7;\
}


// T = N/8 = number of threads.

// P is the length of input sub-sequences, 1,8,64,...,N/8.

#define MUL_RE(a,b) (a.even*b.even - a.odd*b.odd)

#define MUL_IM(a,b) (a.even*b.odd + a.odd*b.even)



#define mul_1(a, b) { float2 tmp = b; b.even = MUL_RE(a,tmp); b.odd = MUL_IM(a,tmp); }

/*
float2 mul_1(float2 a,float2 b)

{ float2 x; x.even = MUL_RE(a,b); x.odd = MUL_IM(a,b); return x; }
*/

#define mul_2(a,b) { float4 x; x.even = MUL_RE(a,b); x.odd = MUL_IM(a,b); return x; }

/*
float4 mul_2(float4 a,float4 b)

{ float4 x; x.even = MUL_RE(a,b); x.odd = MUL_IM(a,b); return x; }
*/




// Return cos(alpha)+I*sin(alpha)  (3 variants)

#define exp_alpha_1(alpha, out)\
{\
    float cs,sn;\
    cs = fast::cos(alpha); sn = fast::sin(alpha);\
    out = (float2)(cs,sn);\
}


#define mul_p0q1(a) (a)

#define mul_p0q2 mul_p0q1

#define mul_p1q2(a) { a = (float2)(a.y,-a.x); }


// Return a^2

#define sqr_1(a) { a=(float2)(a.x*a.x-a.y*a.y,2.0f*a.x*a.y); }

__constant float SQRT_1_2 = 0.707106781188f; // cos(Pi/4)

#define mul_p0q4 mul_p0q2

#define mul_p1q4(a) { a = (float2)(SQRT_1_2)*(float2)(a.x+a.y,-a.x+a.y); }

#define mul_p2q4 mul_p1q2

#define mul_p3q4(a) { a = (float2)(SQRT_1_2)*(float2)(-a.x+a.y,-a.x-a.y); }

// Radix-2 kernel

#define dft2_2(a) { a = float4(a.lo+a.hi,a.lo-a.hi); }


kernel void fft_radix2(      device float2 *inBuffer [[ buffer(0) ]],
                         /*device float2 *outBuffer [[ buffer(1) ]],*/
                         device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                         uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                         uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                         uint2 gid [[thread_position_in_grid]])


{
 
    int N = uniforms.signalLength;
    int t = N/2;//get_global_size(0); // number of threads
    
    int i = gid.x % t;// get_global_id(0); // current thread
    
    int p = uniforms.fftStage;
    int k = i & (p-1); // index in input sequence, in 0..P-1
    
    // Inputs indices are I+{0,1,2,3,4,5,6,7}*T
    
    int rowIndex = gid.y * N;
    
    int inIndex = rowIndex + i;
    
    // Output indices are J+{0,1,2,3,4,5,6,7}*P, where
    
    // J is I with three 0 bits inserted at bit log2(P)
    
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    float2 in1 = inBuffer[inIndex];
    float2 in2 = inBuffer[inIndex+t];
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    int outIndex = rowIndex + (i<<1) - k;
    
    float2 twiddle;
    exp_alpha_1(-PI*(float)k/(float)p, twiddle);
    
    //float2 result = inBuffer[inIndex+t];
    mul_1( twiddle , in2 )
    float4 u = float4( in1, in2 );
    dft2_2( u );
    
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    inBuffer[outIndex] = u.lo;
    inBuffer[outIndex+p] = u.hi;
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
}

kernel void fft_radix2_OOP(      device float2 *inBuffer [[ buffer(0) ]],
                       device float2 *outBuffer [[ buffer(1) ]],
                       device FFTImageFilterUniforms &uniforms [[buffer(2)]],
                       uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                       uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                       uint2 gid [[thread_position_in_grid]])


{
    
    int N = uniforms.signalLength;
    int t = N/2;//get_global_size(0); // number of threads
    
    int i = gid.x % t;// get_global_id(0); // current thread
    
    int p = uniforms.fftStage;
    int k = i & (p-1); // index in input sequence, in 0..P-1
    
    // Inputs indices are I+{0,1,2,3,4,5,6,7}*T
    
    int rowIndex = gid.y * N;
    
    int inIndex = rowIndex + i;
    
    // Output indices are J+{0,1,2,3,4,5,6,7}*P, where
    
    // J is I with three 0 bits inserted at bit log2(P)
    
    float2 in1 = inBuffer[inIndex];
    float2 in2 = inBuffer[inIndex+t];
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    int outIndex = rowIndex + (i<<1) - k;
    
    float2 twiddle;
    exp_alpha_1(-PI*(float)k/(float)p, twiddle);
    
    //float2 result = inBuffer[inIndex+t];
    mul_1( twiddle , in2 )
    float4 u = float4( in1, in2 );
    dft2_2( u );
    
    outBuffer[outIndex] = u.lo;
    outBuffer[outIndex+p] = u.hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
}


kernel void fft_radix2_2(      device float2 *evenBuffer [[ buffer(0) ]],
                         device float2 *oddBuffer [[ buffer(1) ]],
                         device FFTImageFilterUniforms &uniforms [[buffer(2)]],
                         /*uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],*/
                         /*uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],*/
                         uint2 gid [[thread_position_in_grid]])


{

    int N = uniforms.signalLength;
    int t = N/2;//get_global_size(0); // number of threads
    
    int i = gid.x % t;// get_global_id(0); // current thread
    
    int p = uniforms.fftStage;
    int k = i & (p-1); // index in input sequence, in 0..P-1
    
    // Inputs indices are I+{0,1,2,3,4,5,6,7}*T
    
    int rowIndex = gid.y * N;
    
    int inIndex = rowIndex + i;
    
    // Output indices are J+{0,1,2,3,4,5,6,7}*P, where
    
    // J is I with three 0 bits inserted at bit log2(P)
    
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    float2 in1 = evenBuffer[inIndex];
    float2 in2 = evenBuffer[inIndex+t];
    
    float2 in3 = oddBuffer[inIndex];
    float2 in4 = oddBuffer[inIndex+t];
    
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    int outIndex = rowIndex + (i<<1) - k;
    
    float2 twiddle;
    exp_alpha_1(-PI*(float)k/(float)p, twiddle);
    
    //float2 result = inBuffer[inIndex+t];
    mul_1( twiddle , in2 )
    float4 u = float4( in1, in2 );
    dft2_2( u );
    
    mul_1( twiddle , in4 )
    float4 w = float4( in3, in4 );
    dft2_2( w );
    
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    oddBuffer[outIndex] = u.lo;
    oddBuffer[outIndex+p] = u.hi;
    
    oddBuffer[outIndex] = w.lo;
    oddBuffer[outIndex+p] = w.hi;
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

}

kernel void fft_radix8_OOP(      device float2 *inBuffer [[ buffer(0) ]],
                                 device float2 *outBuffer [[ buffer(1) ]],
                        /*device float2 *outBuffer [[ buffer(1) ]],*/
                        device FFTImageFilterUniforms &uniforms [[buffer(2)]],
                        uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                        uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                        uint2 gid [[thread_position_in_grid]])


{

    int N = uniforms.signalLength;
    int t = N/8;//get_global_size(0); // number of threads
    
    int i = gid.x % t;// get_global_id(0); // current thread
    
    int p = uniforms.fftStage;
    int k = i & (p-1); // index in input sequence, in 0..P-1
    
    // Inputs indices are I+{0,1,2,3,4,5,6,7}*T
    
    int rowIndex = gid.y * N;
    
    int inIndex = rowIndex + i;
    
    // Output indices are J+{0,1,2,3,4,5,6,7}*P, where
    
    // J is I with three 0 bits inserted at bit log2(P)
    
    int outIndex = rowIndex + ((i-k)<<3) + k;
    
    
    
    // Load and twiddle inputs
    
    // Twiddling factors are exp(_I*PI*{0,1,2,3,4,5,6,7}*K/4P)
    
    float alpha = -PI*(float)k/(float)(4*p);
    
    
    
    // Load and twiddle
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    float2 u0 = inBuffer[inIndex + 0];
    float2 u1 = inBuffer[inIndex + t];
    float2 u2 = inBuffer[inIndex + 2*t];
    float2 u3 = inBuffer[inIndex + 3*t];
    float2 u4 = inBuffer[inIndex + 4*t];
    float2 u5 = inBuffer[inIndex + 5*t];
    float2 u6 = inBuffer[inIndex + 6*t];
    float2 u7 = inBuffer[inIndex + 7*t];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    
    //if( 1 )
    //{
    float2 twiddle;
    exp_alpha_1(alpha, twiddle); // W
    
    
    mul_1(twiddle,u1);
    
    mul_1(twiddle,u3);
    
    mul_1(twiddle,u5);
    
    mul_1(twiddle,u7);
    
    /*twiddle = */sqr_1(twiddle); // W^2
    
    /*u2 = */mul_1(twiddle,u2);
    
    /*u3 = */mul_1(twiddle,u3);
    
    /*u6 = */mul_1(twiddle,u6);
    
    /*u7 = */mul_1(twiddle,u7);
    
    /*twiddle = */sqr_1(twiddle); // W^4
    
    /*u4 = */mul_1(twiddle,u4);
    
    /*u5 = */mul_1(twiddle,u5);
    
    /*u6 = */mul_1(twiddle,u6);
    
    /*u7 = */mul_1(twiddle,u7);
        
    //}
        
        // 4x in-place DFT2 and twiddle
        
        float2 v0 = u0 + u4;
        
        float2 v4 = mul_p0q4(u0 - u4);
        
        float2 v1 = u1 + u5;
        
        float2 v5 = u1 - u5;
        mul_p1q4(v5);
        
        float2 v2 = u2 + u6;
        
        float2 v6 = u2 - u6;
         mul_p2q4(v6);
        
        float2 v3 = u3 + u7;
        
        float2 v7 = u3 - u7;
        mul_p3q4(v7);
        
        
        
        // 4x in-place DFT2 and twiddle
        
        u0 = v0 + v2;
        
        u2 = mul_p0q2(v0 - v2);
        
        u1 = v1 + v3;
        
        u3 = v1 - v3;
        mul_p1q2(u3);
        
        u4 = v4 + v6;
        
        u6 = mul_p0q2(v4 - v6);
        
        u5 = v5 + v7;
        
        u7 = v5 - v7;
        mul_p1q2(u7);
        
        
        
        // 4x DFT2 and store (reverse binary permutation)
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        outBuffer[outIndex + 0]   = u0 + u1;
        
        outBuffer[outIndex + p]   = u4 + u5;
        
        outBuffer[outIndex + 2*p] = u2 + u3;
        
        outBuffer[outIndex + 3*p] = u6 + u7;
        
        outBuffer[outIndex + 4*p] = u0 - u1;
        
        outBuffer[outIndex + 5*p] = u4 - u5;
        
        outBuffer[outIndex + 6*p] = u2 - u3;
        
        outBuffer[outIndex + 7*p] = u6 - u7;
        threadgroup_barrier(mem_flags::mem_threadgroup);

    }


kernel void fft_radix8(      device float2 *inBuffer [[ buffer(0) ]],
                       /*device float2 *outBuffer [[ buffer(1) ]],*/
                       device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                       uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                       uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                       uint2 gid [[thread_position_in_grid]])


{
    
    int N = uniforms.signalLength;
    int t = N/8;//get_global_size(0); // number of threads
    
    int i = gid.x % t;// get_global_id(0); // current thread
    
    int p = uniforms.fftStage;
    int k = i & (p-1); // index in input sequence, in 0..P-1
    
    // Inputs indices are I+{0,1,2,3,4,5,6,7}*T
    
    int rowIndex = gid.y * N;
    
    int inIndex = rowIndex + i;
    
    // Output indices are J+{0,1,2,3,4,5,6,7}*P, where
    
    // J is I with three 0 bits inserted at bit log2(P)
    
    int outIndex = rowIndex + ((i-k)<<3) + k;
    
    
    
    // Load and twiddle inputs
    
    // Twiddling factors are exp(_I*PI*{0,1,2,3,4,5,6,7}*K/4P)
    
    float alpha = -PI*(float)k/(float)(4*p);
    
    
    
    // Load and twiddle
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    float2 u0 = inBuffer[inIndex + 0];
    float2 u1 =  inBuffer[inIndex+t];
    float2 u2 = inBuffer[inIndex + 2*t];
    float2 u3 = inBuffer[inIndex + 3*t];
    float2 u4 = inBuffer[inIndex + 4*t];
    float2 u5 = inBuffer[inIndex+5*t];
    float2 u6 = inBuffer[inIndex+6*t];
    float2 u7 = inBuffer[inIndex+7*t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    
    //if( 1 )
    //{
    float2 twiddle;
    exp_alpha_1(alpha, twiddle); // W
    
    
    mul_1(twiddle,u1);
    
    mul_1(twiddle,u3);
    
    mul_1(twiddle,u5);
    
    mul_1(twiddle,u7);
    
    /*twiddle = */sqr_1(twiddle); // W^2
    
    /*u2 = */mul_1(twiddle,u2);
    
    /*u3 = */mul_1(twiddle,u3);
    
    /*u6 = */mul_1(twiddle,u6);
    
    /*u7 = */mul_1(twiddle,u7);
    
    /*twiddle = */sqr_1(twiddle); // W^4
    
    /*u4 = */mul_1(twiddle,u4);
    
    /*u5 = */mul_1(twiddle,u5);
    
    /*u6 = */mul_1(twiddle,u6);
    
    /*u7 = */mul_1(twiddle,u7);
    
    //}
    
    // 4x in-place DFT2 and twiddle
    
    float2 v0 = u0 + u4;
    
    float2 v4 = mul_p0q4(u0 - u4);
    
    float2 v1 = u1 + u5;
    
    float2 v5 = u1 - u5;
    mul_p1q4(v5);
    
    float2 v2 = u2 + u6;
    
    float2 v6 = u2 - u6;
    mul_p2q4(v6);
    
    float2 v3 = u3 + u7;
    
    float2 v7 = u3 - u7;
    mul_p3q4(v7);
    
    
    
    // 4x in-place DFT2 and twiddle
    
    u0 = v0 + v2;
    
    u2 = mul_p0q2(v0 - v2);
    
    u1 = v1 + v3;
    
    u3 = v1 - v3;
    mul_p1q2(u3);
    
    u4 = v4 + v6;
    
    u6 = mul_p0q2(v4 - v6);
    
    u5 = v5 + v7;
    
    u7 = v5 - v7;
    mul_p1q2(u7);
    
    
    
    // 4x DFT2 and store (reverse binary permutation)
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    inBuffer[outIndex + 0]   = u0 + u1;
    
    inBuffer[outIndex + p]   = u4 + u5;
    
    inBuffer[outIndex + 2*p] = u2 + u3;
    
    inBuffer[outIndex + 3*p] = u6 + u7;
    
    inBuffer[outIndex + 4*p] = u0 - u1;
    
    inBuffer[outIndex + 5*p] = u4 - u5;
    
    inBuffer[outIndex + 6*p] = u2 - u3;
    
    inBuffer[outIndex + 7*p] = u6 - u7;
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
}


kernel void fft_radix8_2(      device float2 *evenBuffer [[ buffer(0) ]],
                       /*device float2 *oddBuffer [[ buffer(1) ]],*/
                       device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                       /*uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],*/
                       /*uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],*/
                       uint2 gid [[thread_position_in_grid]])


{
    
    int N = uniforms.signalLength/2;
    int t = N/8;//get_global_size(0); // number of threads
    
    int i = gid.x % t;// get_global_id(0); // current thread
    
    int p = uniforms.fftStage;
    int k = i & (p-1); // index in input sequence, in 0..P-1
    
    // Inputs indices are I+{0,1,2,3,4,5,6,7}*T
    
    int rowIndex = gid.y * uniforms.signalLength;
    
    int inIndex = rowIndex + i;
    
    // Output indices are J+{0,1,2,3,4,5,6,7}*P, where
    
    // J is I with three 0 bits inserted at bit log2(P)
    
    int outIndex = rowIndex + ((i-k)<<3) + k;
    
    
    
    // Load and twiddle inputs
    
    // Twiddling factors are exp(_I*PI*{0,1,2,3,4,5,6,7}*K/4P)
    
    float alpha = -PI*(float)k/(float)(4*p);
    
    
    
    // Load and twiddle
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    float2 u0 = evenBuffer[inIndex + 0];
    float2 u1 = evenBuffer[inIndex+t];
    float2 u2 = evenBuffer[inIndex + 2*t];
    float2 u3 = evenBuffer[inIndex + 3*t];
    float2 u4 = evenBuffer[inIndex + 4*t];
    float2 u5 = evenBuffer[inIndex+5*t];
    float2 u6 = evenBuffer[inIndex+6*t];
    float2 u7 = evenBuffer[inIndex+7*t];
    
    /*
    float2 w0 = oddBuffer[inIndex + 0];
    float2 w1 =  oddBuffer[inIndex+t];
    float2 w2 = oddBuffer[inIndex + 2*t];
    float2 w3 = oddBuffer[inIndex + 3*t];
    float2 w4 = oddBuffer[inIndex + 4*t];
    float2 w5 = oddBuffer[inIndex+5*t];
    float2 w6 = oddBuffer[inIndex+6*t];
    float2 w7 = oddBuffer[inIndex+7*t];
    */
    
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    
    //Perform the complex Twiddles
    //if( 1 )
    //{
        float2 twiddle;
        exp_alpha_1(alpha, twiddle); // W
    
        //even twiddles
        
        mul_1(twiddle,u1);
        
        mul_1(twiddle,u3);
        
        mul_1(twiddle,u5);
        
        mul_1(twiddle,u7);
        
        /*twiddle = */sqr_1(twiddle); // W^2
        
        /*u2 = */mul_1(twiddle,u2);
        
        /*u3 = */mul_1(twiddle,u3);
        
        /*u6 = */mul_1(twiddle,u6);
        
        /*u7 = */mul_1(twiddle,u7);
        
        /*twiddle = */sqr_1(twiddle); // W^4
        
        /*u4 = */mul_1(twiddle,u4);
        
        /*u5 = */mul_1(twiddle,u5);
        
        /*u6 = */mul_1(twiddle,u6);
        
        /*u7 = */mul_1(twiddle,u7);
    
    /*
    //odd twiddles
    exp_alpha_1(alpha, twiddle); // W

    mul_1(twiddle,w1);
    
    mul_1(twiddle,w3);
    
    mul_1(twiddle,w5);
    
    mul_1(twiddle,w7);
    
    sqr_1(twiddle); // W^2
    
    mul_1(twiddle,w2);
    
    mul_1(twiddle,w3);
    
    mul_1(twiddle,w6);
    
    mul_1(twiddle,w7);
    
    sqr_1(twiddle); // W^4
    
    mul_1(twiddle,w4);
    
    mul_1(twiddle,w5);
    
    mul_1(twiddle,w6);
    
    mul_1(twiddle,w7);
    */
    //}
    
    // Even 4x in-place DFT2 and twiddle
    
    float2 v0 = u0 + u4;
    
    float2 v4 = mul_p0q4(u0 - u4);
    
    float2 v1 = u1 + u5;
    
    float2 v5 = u1 - u5;
    mul_p1q4(v5);
    
    float2 v2 = u2 + u6;
    
    float2 v6 = u2 - u6;
    mul_p2q4(v6);
    
    float2 v3 = u3 + u7;
    
    float2 v7 = u3 - u7;
    mul_p3q4(v7);

/*
    // Odd 4x in-place DFT2 and twiddle
    
    float2 y0 = w0 + w4;
    
    float2 y4 = mul_p0q4(w0 - w4);
    
    float2 y1 = w1 + w5;
    
    float2 y5 = w1 - w5;
    mul_p1q4(v5);
    
    float2 y2 = w2 + w6;
    
    float2 y6 = w2 - w6;
    mul_p2q4(v6);
    
    float2 y3 = w3 + w7;
    
    float2 y7 = w3 - w7;
    mul_p3q4(y7);
   */
    
    // Even 4x in-place DFT2 and twiddle
    
    u0 = v0 + v2;
    
    u2 = mul_p0q2(v0 - v2);
    
    u1 = v1 + v3;
    
    u3 = v1 - v3;
    mul_p1q2(u3);
    
    u4 = v4 + v6;
    
    u6 = mul_p0q2(v4 - v6);
    
    u5 = v5 + v7;
    
    u7 = v5 - v7;
    mul_p1q2(u7);
    
/*
    // Odd 4x in-place DFT2 and twiddle
    
    w0 = y0 + y2;
    
    w2 = mul_p0q2(y0 - y2);
    
    w1 = y1 + y3;
    
    w3 = y1 - y3;
    mul_p1q2(y3);
    
    w4 = y4 + y6;
    
    w6 = mul_p0q2(y4 - y6);
    
    w5 = y5 + y7;
    
    w7 = y5 - y7;
    mul_p1q2(w7);
*/
    
    
    // 4x DFT2 and store (reverse binary permutation)
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    float2 out = float2( 1, 0 );
    evenBuffer[outIndex]   = u0 + u1;
    
    evenBuffer[outIndex + p]   = u4 + u5;
    
    evenBuffer[outIndex + 2*p] = u2 + u3;
    
    evenBuffer[outIndex + 3*p] = u6 + u7;
    
    evenBuffer[outIndex + 4*p] = u0 - u1;
    
    evenBuffer[outIndex + 5*p] = u4 - u5;
    
    evenBuffer[outIndex + 6*p] = u2 - u3;
    
    evenBuffer[outIndex + 7*p] = u6 - u7;
    
    /*
    
    oddBuffer[outIndex]   = w0 + w1;
    
    oddBuffer[outIndex + p]   = w4 + w5;
    
    oddBuffer[outIndex + 2*p] = w2 + w3;
    
    oddBuffer[outIndex + 3*p] = w6 + w7;
    
    oddBuffer[outIndex + 4*p] = w0 - w1;
    
    oddBuffer[outIndex + 5*p] = w4 - w5;
    
    oddBuffer[outIndex + 6*p] = w2 - w3;
    
    oddBuffer[outIndex + 7*p] = w6 - w7;
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    */
}


kernel void FFT_8_KERNEL(      device float2 *inBuffer [[ buffer(0) ]],
                               /*device float2 *outBuffer [[ buffer(1) ]],*/
                               device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                               uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                               uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                               uint2 gid [[thread_position_in_grid]])
{


    uint N = uniforms.signalLength;     //FFT 1D Signal Length (power of 2)
    uint K_W = 8;//uniforms.kernelSize;     //kernel size
    uint Ns = uniforms.fftStage;        //fft stage index = 2^num_completed_radix-2_fft_stages; 8 point kernel is always the bottom (first) recursive stage
    uint gSize = N/8;                   //512/8= 64; number of threads operating on point kernels per row simultaneously
    //uint numKernels =

    uint columnOffset = gid.x;
    uint rowIndex = gid.y * N;
    
    //uint numThreads = N/K_W; //64 threads
    //threadgroup block size / kernelSize + thread offset into threadgroup
    //K_W = 8; fftSize = 512; 512/8 = 64 (num threadgroups per row)
    uint gId = gid.x % gSize;//*/( columnOffset - (columnOffset  % gSize) ) / K_W;///*(threadIndexInThreadGroup) + */((columnOffset - threadIndexInThreadGroup  ) / K_W);

    //uint gId = (columnOffset - (columnOffset  % gSize) ) / K_W;
    //int gId = get_global_id(0);
    //uint tps = N/K_W;
    //int input_offset = (gId / tps) * N;
    //gId = gId % tps;
    uint Idin = gId + rowIndex;
    

    //uint Idin = rowIndex+gId;

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    float2 in0, in1, in2, in3, in4, in5, in6, in7;
    in0 = inBuffer[ (0*gSize)+Idin];
    in1 = inBuffer[ (1*gSize)+Idin];
    in2 = inBuffer[ (2*gSize)+Idin];
    in3 = inBuffer[ (3*gSize)+Idin];
    in4 = inBuffer[ (4*gSize)+Idin];
    in5 = inBuffer[ (5*gSize)+Idin];
    in6 = inBuffer[ (6*gSize)+Idin];
    in7 = inBuffer[ (7*gSize)+Idin];

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    int k = gId & (Ns-1); // index in input sequence, in 0..P-1

    if( Ns != 1 )
    {
        //float t1 = gId%Ns;
        //float t2 = Ns*K_W;
        //float angle0 = -2*PI*(t1/t2);//(gId%Ns)/(Ns*K_W);
        
        float angle0 = -PI*(float)k/(float)(4*Ns);

        //float angle1 = 2*-2*PI*( threadIndex%Ns)/(Ns*8);
        //float angle2 = 3*-2*PI*( threadIndex%Ns)/(Ns*8);
        //float angle3 = 4*-2*PI*( threadIndex%Ns)/(Ns*8);
        //float angle4 = 5*-2*PI*( threadIndex%Ns)/(Ns*8);
        //float angle5 = 6*-2*PI*( threadIndex%Ns)/(Ns*8);
        //float angle6 = 7*-2*PI*( threadIndex%Ns)/(Ns*8);
        //float angle7 = 8*-2*PI*( threadIndex%Ns)/(Ns*8);

        TWIDDLE_FACTOR(1, angle0, in1);
        TWIDDLE_FACTOR(2, angle0, in2);
        TWIDDLE_FACTOR(3, angle0, in3);
        TWIDDLE_FACTOR(4, angle0, in4);
        TWIDDLE_FACTOR(5, angle0, in5);
        TWIDDLE_FACTOR(6, angle0, in6);
        TWIDDLE_FACTOR(7, angle0, in7);
    }
    
    
    FFT_8(in0, in1, in2, in3, in4, in5, in6, in7);


    //float threadIDFloat = (float)threadIndex;
    //float2 outColor(threadIDFloat/64.f,0);
    /*
    if( (threadIndex/8)%2 == 0 )
    {
        outColor = float2(1,0);
        //inBuffer[rowIndex+columnOffset] = outColor;
        
        //if( threadIndexInThreadGroup == 0 )
        //    outColor = float2(0.5, 0);
        
        //if( threadIndex == 62 )
        //{
        //    outColor = float2(0.5, 0);
        //}
    }
    */
    
    //inBuffer[rowIndex+columnOffset] = outColor;
    
    /*
    int Idout = (threadIndex/Ns)*Ns*K_W+(threadIndex%Ns);
    inBuffer[rowIndex + (0*Ns)+Idout] = outColor;
    inBuffer[rowIndex + (1*Ns)+Idout] = outColor;
    inBuffer[rowIndex + (2*Ns)+Idout] = outColor;
    inBuffer[rowIndex + (3*Ns)+Idout] = outColor;
    inBuffer[rowIndex + (4*Ns)+Idout] = outColor;
    inBuffer[rowIndex + (5*Ns)+Idout] = outColor;
    inBuffer[rowIndex + (6*Ns)+Idout] = outColor;
    inBuffer[rowIndex + (7*Ns)+Idout] = outColor;
    
    */
    

    //uint Idout = (threadIndex/Ns)*Ns*K_W+(threadIndex%Ns);
    //float t1 = (float)gId/(float)Ns;
    //float t2 = (float)Ns*(float)K_W;
    
    uint Idout = ((gId-k)<<3) + k + rowIndex;///t1*t2+(gId %Ns) + rowIndex;//rowIndex;
    //Idout += rowIndex;
    
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    inBuffer[ (0*Ns)+Idout] = in0;
    inBuffer[ (1*Ns)+Idout] = in1;
    inBuffer[ (2*Ns)+Idout] = in2;
    inBuffer[ (3*Ns)+Idout] = in3;
    inBuffer[ (4*Ns)+Idout] = in4;
    inBuffer[ (5*Ns)+Idout] = in5;
    inBuffer[ (6*Ns)+Idout] = in6;
    inBuffer[ (7*Ns)+Idout] = in7;
    
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    

    //float2 outColor = float2(float(threadIndex)/float(64),0);
    

    //if( uniforms.fftStage == 0 )
    //    outColor = float2(1,0);
    
    /*
    float2 outColor = float2(0,0);
    
    if( gSize == 64 )
        outColor = float2(1, 0 );
    
    inBuffer[ rowIndex + columnOffset ] = outColor;
    */
    
/*
    int N=8;

    //int gId = get_global_id(0);
    int gSize = N/8;
    float2 in0, in1, in2, in3, in4, in5, in6, in7;
    
    in0 = fftTexture.read(gid);

    in1 = datain[(1*gSize)+gId];
    in2 = datain[(2*gSize)+gId];
    in3 = datain[(3*gSize)+gId];
    
    //if (Ns!=1)
    //{
        
        float angle = -2*PI*(gId)/(N);
        twidle_factor(1, angle, in1);
        twidle_factor(2, angle, in2);
        twidle_factor(3, angle, in3);
    //}
    
    FFT8(in0, in1, in2, in3, in4, in5, in6, in7);

    int Idout = (gId/Ns)*Ns*K_W+(gId%Ns);
    dataout[(0*Ns)+Idout] = in0;
    dataout[(1*Ns)+Idout] = in1;
    dataout[(2*Ns)+Idout] = in2;
    dataout[(3*Ns)+Idout] = in3;
    dataout[(4*Ns)+Idout] = in4;
    dataout[(5*Ns)+Idout] = in5;
    dataout[(6*Ns)+Idout] = in6;
    dataout[(7*Ns)+Idout] = in7;
*/
}


#pragma mark -- 4 Point FFT Kernel

//define a radix-4 Cooley-Tukey FFT function
#define FFT_4(in0, in1, in2, in3) {\
float2 v0, v1, v2, v3;\
v0 = in0 + in2;\
v1 = in1 + in3;\
v2 = in0;\
v2 -= in2;\
v3.x = in1.y;\
v3.x -= in3.y;\
v3.y = in3.x;\
v3.y -= in1.x;\
in0 = v0 + v1;\
in2 = v0;\
in2 = in2 - v1;\
in1 = v2 + v3;\
in3 = v2;\
in3 = in3 - v3;\
}


kernel void FFT_4_KERNEL(      device float2 *inBuffer [[ buffer(0) ]],
                         device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                         uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                         uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                         uint2 gid [[thread_position_in_grid]])
{
    //int gId = get_global_id(0);
    //int gSize = N/4;
    
    uint N = uniforms.signalLength;     //FFT 1D Signal Length (power of 2)
    uint K_W = 4;//uniforms.kernelSize;     //kernel size
    uint Ns = uniforms.fftStage;        //fft stage index = 2^num_completed_radix-2_fft_stages; 8 point kernel is always the bottom (first) recursive stage
    uint gSize = N/K_W;                   //512/8= 64; number of threads operating on point kernels per row simultaneously
    //uint numKernels =
    
    uint columnOffset = gid.x;
    uint rowIndex = gid.y * N;
    

    uint gId = columnOffset % gSize;
    
    uint Idin = rowIndex+gId;
    
    float2 in0, in1, in2, in3;
    in0 = inBuffer[ (0*gSize)+Idin];
    in1 = inBuffer[ (1*gSize)+Idin];
    in2 = inBuffer[ (2*gSize)+Idin];
    in3 = inBuffer[ (3*gSize)+Idin];
    
    
    if (Ns!=1)
    {
        float t1 = gId%Ns;
        float t2 = Ns*K_W;
        float angle = -2*PI*t1/t2;
        //float angle = -2*PI*(gId)/(N);
        TWIDDLE_FACTOR(1, angle, in1);
        TWIDDLE_FACTOR(2, angle, in2);
        TWIDDLE_FACTOR(3, angle, in3);
    }
    
    FFT_4(in0, in1, in2, in3);
    
    /*
    int Idout = (gId/Ns)*Ns*K_W+(gId%Ns);
    dataout[(0*Ns)+Idout] = in0;
    dataout[(1*Ns)+Idout] = in1;
    dataout[(2*Ns)+Idout] = in2;
    dataout[(3*Ns)+Idout] = in3;
    */
    
    float t1 = (float)gId/(float)Ns;
    float t2 = (float)Ns*(float)K_W;
    uint Idout = t1*t2+(gId %Ns) + rowIndex;
    //Idout += rowIndex;
    
    inBuffer[ (0*Ns)+Idout] = in0;
    inBuffer[ (1*Ns)+Idout] = in1;
    inBuffer[ (2*Ns)+Idout] = in2;
    inBuffer[ (3*Ns)+Idout] = in3;

}

#pragma mark -- 2 Point FFT Kernel

//define a radix-2 Cooley-Tukey FFT function
#define FFT_2(in0, in1) {\
float2 v0;\
v0 = in0;\
in0 = v0 + in1;\
in1 = v0 - in1;\
}

kernel void FFT_2_KERNEL(      device float2 *inBuffer [[ buffer(0) ]],
                         device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                         uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                         uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                         uint2 gid [[thread_position_in_grid]])
{
    //int gId = get_global_id(0);
    //int gSize = N/4;

    uint N = uniforms.signalLength;     //FFT 1D Signal Length (power of 2)
    uint K_W = 2;//uniforms.kernelSize;     //kernel size
    uint Ns = uniforms.fftStage;        //fft stage index = 2^num_completed_radix-2_fft_stages; 8 point kernel is always the bottom (first) recursive stage
    uint gSize = N/K_W;                   //512/8= 64; number of threads operating on point kernels per row simultaneously
    //uint numKernels =
    
    uint columnOffset = gid.x;
    uint rowIndex = gid.y * N;
    
    
    uint gId = columnOffset % gSize;
    
    uint Idin = rowIndex+gId;
    
    float2 in0, in1;
    
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    in0 = inBuffer[ (0*gSize)+Idin];
    in1 = inBuffer[ (1*gSize)+Idin];
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);


    
    if (Ns!=1)
    {
        float t1 = gId%Ns;
        float t2 = Ns*K_W;
        float angle = -2*PI*t1/t2;
        //float angle = -2*PI*(gId)/(N);
        TWIDDLE_FACTOR(1, angle, in1);
    }
    
    FFT_2(in0, in1);

    
    /*
     int Idout = (gId/Ns)*Ns*K_W+(gId%Ns);
     dataout[(0*Ns)+Idout] = in0;
     dataout[(1*Ns)+Idout] = in1;

     */
    
    float t1 = (float)gId/(float)Ns;
    float t2 = (float)Ns*(float)K_W;
    uint Idout = t1*t2+(gId %Ns) + rowIndex;
    //Idout += rowIndex;
    
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    inBuffer[ (0*Ns)+Idout] = in0;
    inBuffer[ (1*Ns)+Idout] = in1;
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    
}


kernel void FFT_1D_TWIST_INTERLEAVED(device float2 *inBuffer [[ buffer(0) ]],
                                     constant FFTImageFilterUniforms &uniforms [[buffer(1)]],
                                     uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                                     uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                                     uint2 gid [[thread_position_in_grid]])
{

    uint N = uniforms.signalLength;     //FFT 1D Signal Length (power of 2)
    uint K_W = uniforms.kernelSize;     //kernel size
    uint Ns = uniforms.fftStage;        //fft stage index; 8 point kernel is always the bottom (first) recursive stage
    uint gSize = N/K_W;                 //number of threads operating on point kernels per row simultaneously

    uint columnOffset = gid.x;
    uint rowIndex = gid.y * N;
    uint threadIndex =  threadIndexInThreadGroup + ((columnOffset ) / K_W);

    for (uint i=0; i<K_W/2; i++ )
    {
        //in0 = inBuffer[rowIndex + (0*gSize)+threadIndex];
    
    }
    

}

//define the Metal GPGPU kernel function for calculating real and complex fft output on an MTLTexture
kernel void FFT_2D_Image_Filter(                /*texture2d<float, access::read> inTexture [[texture(0)]],*/
                                                texture2d<float, access::read> fftTexture [[texture(0)]],
                                                texture2d<float, access::write> outTexture [[texture(1)]],
                                                constant FFTImageFilterUniforms &uniforms [[buffer(0)]],
                                                uint2 gid [[thread_position_in_grid]],
                                                uint2 lid [[thread_position_in_threadgroup]],
                                                uint threadIndexInThreadGroup [[thread_index_in_threadgroup]],
                                                uint2 threadGroupPositionInGrid [[threadgroup_position_in_grid]],
                                                uint2 threadGroupsPerGrid [[threadgroups_per_grid]] )
{
    //convert the 32 bit RGBA image to grayscale
    
    //int fftSize = uniforms.fftDimension;
    //uint fftSize = fftTexture.get_width();
    //float4 image_color = inTexture.read(gid);//tex.sample(s_quad, in.uv);
    //float grayColor = dot(image_color.rgb, float3(0.30h, 0.59h, 0.11h));
    //float4 outColor(grayColor, grayColor, grayColor, 1.0);
    
    //fftTexture.write(outColor, gid);
    //constexpr sampler s_quad(filter::linear);
    //half4 image_color = tex.sample(s_quad, in.uv);

    //Pack_Cmplx( inTexture, fftTexture);

    //float4 fftColor = fftTexture.read(gid);
    //float4 outColor = float4( fftColor.r, fftColor.r, fftColor.r, 1);
    //float out = float(threadGroupIndex)/255.0;
    
    

    //uint threadGroupIndex = tgBuffer[0];
    //threadgroup uint threadGroupIndex;

    //threadgroup_barrier(mem_flags::mem_threadgroup);

    //if( gid.x ==0  && gid.y == 0 )
    //{
    //    threadGroupIndex = 0;
    //}

    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    uint threadGroupIndex = 0;
    bool secondQuadX = false;
    bool secondQuadY = false;
    if( threadGroupPositionInGrid.x >= (threadGroupsPerGrid.x / 2) )
    {
        secondQuadX = true;
    }
    
    if( threadGroupPositionInGrid.y >= (threadGroupsPerGrid.y / 2) )
    {
            secondQuadY = true;
    }
    
    if( secondQuadX == true && secondQuadY == true )
    {
            threadGroupIndex = 3;
    }
    else if( secondQuadX )
    {
            threadGroupIndex = 1;
    }
    else if( secondQuadY )
    {
        threadGroupIndex = 2;
    }
    //uint threadGroupIndex = threadGroupPositionInGrid;//tgBuffer->threadGroupIndex;

    threadGroupIndex = threadGroupPositionInGrid.x/255.0;


    float4 outColor = float4(1, 1, 1, 1);
    
    if( threadGroupIndex == 0 )
    {
        outColor = float4(1, 0, 0, 1);
    }
    else if( threadGroupIndex == 1 )
    {
        outColor = float4(0, 0, 1, 1);

    }
    else if( threadGroupIndex == 2 )
    {
        outColor = float4(0, 1, 0, 1);
        
    }
    else if( threadGroupIndex == 3 )
    {
        outColor = float4(0, 0, 0, 1);
        
    }

    //if( threadGroupIndex % 4 == 0 )
    //    threadGroupIndex = 0;

    /*
    else if( threadGroupIndex % 4 == 2 )
    {
        outColor = float4(0, 0, 1, 1);
        threadGroupIndex += 1;

    }
    else if( threadGroupIndex % 4 == 3 )
    {
        outColor = float4(0, 0, 0, 1);
        threadGroupIndex += 1;
        
    }
    */
    //threadgroup_barrier(mem_flags::mem_threadgroup);

    //threadgroup_barrier(mem_flags::mem_threadgroup);

    //threadGroupIndex +=1 ;
    
    //threadgroup_barrier(mem_flags::mem_device_and_threadgroup);


    outTexture.write(outColor, gid);
    
    /*
    for (uint row = 0; row < fftSize; ++row)
    {
        
        uint2 textureCoordIndex(gid.x + 0, gid.y + row);

        //float4 * fftRow = fftTexture.read(textureCoordIndex);
    }
     */


    // inTexture:  the image to process
    // outTexture: the filtered version of the inTexture
    // fftTexture: an RG32 64-bit texture for storing real and complex FFT output

    //pack the real and complex data in the fftTexture
    
    /*
    int X = get_global_id(0);
    float2  V;
    V.x = Data_In [X];
    V.y = Data_In [X+N];
    Data_Out  [X] = V;
     */
}

//define the Metal GPGPU kernel function for calculating radix-2 FFT
/*
kernel void FFT_Texture_Magnitude_Visualization(texture2d<float, access::read> inTexture [[texture(0)]],
                             texture2d<float, access::write> outTexture [[texture(1)]],
                             uint2 gid [[thread_position_in_grid]])
{
    float4 inColor = inTexture.read(gid);
    float value = dot(inColor.rgb, float3(0.299, 0.587, 0.114));
    float4 grayColor(value, value, value, 1.0);
    float4 outColor = mix(grayColor, inColor, uniforms.saturationFactor);
    outTexture.write(outColor, gid);
}
*/

# pragma mark -- Pack/Unpack Buffers before/after FFT processing


/*
 * Pack the complex float2 buffer for FFT calculation
 */
 kernel void Pack_Complex_Buffer(
                                device float2 *complexBuffer [[ buffer(0) ]],
                                texture2d<float, access::read> inTexture [[texture(0)]],
                                constant FFTImageFilterUniforms &uniforms [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]] )
{
    /*
     int X = get_global_id(0);
     float2 V;
     V.x = Data_In [X];
     V.y = Data_In [X+N];
     Data_Out [X] = V;
     */
    
    //convert to grayscale
    
    uint columnOffset = gid.x;
    //uint rowIndex = gid.y * inTexture.get_width();
    uint complexRowIndex = gid.y * uniforms.signalLength;
    
    if( gid.x < inTexture.get_width() && gid.y < inTexture.get_height() )
    {
    float4 image_color = inTexture.read(uint2(gid.x, gid.y));//tex.sample(s_quad, in.uv);
    float grayColor = dot(image_color.rgb, float3(0.30h, 0.59h, 0.11h));
    
    //our complexTexture is of type RG32 (float2), but we can only write float4 using .write function
    float2 outColor(grayColor, 0);
    
    
    //if( columnOffset > inTexture.get_width()-1 )
    //    outColor = float2(1, 0);
    //else
    complexBuffer[complexRowIndex+columnOffset] = outColor;
    
    }
    else
        complexBuffer[complexRowIndex + columnOffset] = float2(0,0);
    
    /*
     if( columnOffset == inTexture.get_width()-1 )
     {
         for( unsigned int i = columnOffset+1; i< uniforms.signalLength; i++ )
         {
         
         outColor = float2(0,0);
         complexBuffer[complexRowIndex+i] = outColor;
         }
     }
     
     if( gid.y == inTexture.get_height()-1 )
     {
         for( unsigned int i = gid.y+1; i< uniforms.signalLength; i++ )
         {
         
         outColor = float2(0,0);
         complexBuffer[i*uniforms.signalLength+columnOffset] = outColor;
         }
     }
     
     if( gid.x == inTexture.get_width()-1 && gid.y == inTexture.get_height()-1 )
     {
         for( unsigned int row = gid.y+1; row<uniforms.signalLength; row++)
         {
         complexRowIndex = row * uniforms.signalLength;
             for (unsigned int column = gid.x+1; column<uniforms.signalLength; column++)
             {
             complexBuffer[complexRowIndex + column] = float2(0,0);
             }
         }
     }
    
    */
}

/*
 * Unpack the complex float2 buffer after FFT calculation
 */
kernel void Unpack_Complex_Buffer( /*texture2d<float, access::read> inTexture [[texture(0)]],*/
                                  texture2d<float, access::write> outTexture [[texture(0)]],
                                  device float2 *complexBuffer [[ buffer(0) ]],
                                  constant FFTImageFilterUniforms &uniforms [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]] )
{
    /*
     int X = get_global_id(0);
     float2 V;
     V.x = Data_In [X];
     V.y = Data_In [X+N];
     Data_Out [X] = V;
     */
    /*
     for ( uint col=0; col<352; col++)
     {
     for( uint row=0; row<288; row++ )
     {
     uint complexRowIndex = row * uniforms.signalLength;
     float2 cout = complexBuffer[complexRowIndex+col];
     float4 outColor = float4( cout.x, cout.x, cout.x, 1);
     outTexture.write( outColor, gid );
     
     }
     }
     */
    
    uint lastRowIndex = uniforms.signalLength - 1;
    uint lastColumnIndex = uniforms.signalLength - 1;
    
    uint lastTextureIndex = uniforms.signalLength * lastRowIndex + lastColumnIndex;
    float2 minMax = complexBuffer[lastTextureIndex];
    
    float minVal = minMax.x;
    float maxVal = minMax.y;
    
    /*
     //normalize the values in the complex buffer
     for ( uint row = 0; row < outTexture.get_height(); row++)
     {
     for( uint column = 0; column < outTexture.get_width(); column++)
     {
     float2 cout = complexBuffer[ row+column];
     
     
     if( cout.x > maxVal )
     maxVal = cout.x;
     else if( cout.x < minVal )
     minVal = cout.x;
     }
     }
     */
    
    uint columnOffset = gid.x;
    //uint rowIndex = gid.y * outTexture.get_width();
    uint complexRowIndex = gid.y * uniforms.signalLength;
    
    
    //real value from complex buffer
    float2 cout = complexBuffer[complexRowIndex+columnOffset];
    
    float normalizedVal = cout.x;//(cout.x/maxVal);
    float4 outColor = float4( normalizedVal, normalizedVal, normalizedVal, 1);
    
    //if( gid.x < 353 && gid.y < 289 )
    
    //if( columnOffset > 351 || gid.y > 287)
    //    outColor = float4(1, 1, 1, 0);
    
    outTexture.write(outColor, gid/*uint2(512 - gid.x, 512 - gid.y)*/);
    
}



/*
 * Pack the complex float2 buffer for FFT calculation
 */
kernel void Pack_Complex_Buffer2(texture2d<float, access::read> inTexture [[texture(0)]],
                                device float2 *evenComplexBuffer [[ buffer(0) ]],
                                device float2 *oddComplexBuffer [[ buffer(1) ]],
                                constant FFTImageFilterUniforms &uniforms [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]] )
{
    /*
     int X = get_global_id(0);
     float2 V;
     V.x = Data_In [X];
     V.y = Data_In [X+N];
     Data_Out [X] = V;
     */
    
    //convert to grayscale
    float4 image_color = inTexture.read(uint2(gid.x, gid.y));//tex.sample(s_quad, in.uv);
    float grayColor = dot(image_color.rgb, float3(0.30h, 0.59h, 0.11h));
    
    //our complexTexture is of type RG32 (float2), but we can only write float4 using .write function
    float2 outColor(grayColor, 0);
    
    uint columnOffset = gid.x;
    //uint rowIndex = gid.y * inTexture.get_width();
    uint complexRowIndex = gid.y * uniforms.signalLength/2;
    
    
    //if( columnOffset > inTexture.get_width()-1 )
    //    outColor = float2(1, 0);
    //else
    
    if( columnOffset % 2 == 0 )
        evenComplexBuffer[ complexRowIndex + columnOffset/2 ] = outColor;
    else
        oddComplexBuffer[ complexRowIndex + ( columnOffset - 1 )/2 ] = outColor;

}

/*
 * Interleave the complex float2 buffers after FFT calculation
 */
kernel void Interleave_Complex_2(
                                   device float2 *evenComplexBuffer [[ buffer(0) ]],
                                   device float2 *oddComplexBuffer [[ buffer(1) ]],
                                   device float2 *outComplexBuffer [[buffer(2)]],
                                   constant FFTImageFilterUniforms &uniforms [[buffer(3)]],
                                   uint2 gid [[thread_position_in_grid]] )
{
    
    uint shortBufferColumnOffset = gid.x;
    uint shortBufferRowIndex = gid.y * uniforms.signalLength;
    uint shortBufferIndex = shortBufferRowIndex + shortBufferColumnOffset;

    uint longBufferColumnOffset = shortBufferColumnOffset * 2;
    uint longBufferRowIndex = shortBufferRowIndex * 2;
    uint longBufferIndex = longBufferColumnOffset + longBufferRowIndex;
    
    outComplexBuffer[ longBufferIndex ] = evenComplexBuffer[ shortBufferIndex ];
    outComplexBuffer[ longBufferIndex + 1 ] = oddComplexBuffer[ shortBufferIndex ];
    
    
    
    /*
    uint longBufferCol = gid.x;
    uint longBufferRow = gid.y * uniforms.signalLength;
        
    float2 out;

    if( longBufferCol % 2 == 0 )
    {
        out = evenComplexBuffer[ longBufferRow/2 + longBufferCol/2];
    }
    else
    {
        out = oddComplexBuffer[ longBufferRow/2 + (longBufferCol-1)/2];
        
    }
    
    outComplexBuffer[ longBufferRow + longBufferCol ] = out;
    */
}

/*
 * Deinterleave the complex float2 buffers after FFT calculation
 */
kernel void Deinterleave_Complex_2(
                                 device float2 *evenComplexBuffer [[ buffer(0) ]],
                                 device float2 *oddComplexBuffer [[ buffer(1) ]],
                                 device float2 *outComplexBuffer [[buffer(2)]],
                                 constant FFTImageFilterUniforms &uniforms [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]] )
{
    
    uint columnOffset = gid.x;
    //uint rowIndex = gid.y * inTexture.get_width();
    uint complexRowIndex = gid.y * uniforms.signalLength;
    
    float2 outVal = outComplexBuffer[complexRowIndex+columnOffset];
    
    complexRowIndex /= 2;
    
    if( columnOffset % 2 == 0 )
        evenComplexBuffer[ complexRowIndex + columnOffset/2 ] = outVal;
    else
        oddComplexBuffer[ complexRowIndex + ( columnOffset - 1 )/2 ] = outVal;

}


#pragma mark -- Complex Buffer Kernel Functions

/*
 * Copy one complex float2 buffer to another
 */
kernel void Copy_Complex_Buffer(
                                  device float2 *inBuffer [[ buffer(0) ]],
                                  device float2 *outBuffer [[ buffer(1) ]],
                                  constant FFTImageFilterUniforms &uniforms [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]] )
{

    uint rowIndex = gid.y * uniforms.signalLength;
    uint colIndex = gid.x;
    uint index = rowIndex + colIndex;
    outBuffer[ index ] = inBuffer[ index ];
}

/*
 * Transpose the values in the complex buffer
 */
kernel void Transpose_Complex_Buffer(
                                     device float2 *complexBuffer [[ buffer(0) ]],
                                     device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                                     uint2 gid [[thread_position_in_grid]] )

// width = N (signal length)
// height = batch_size (number of signals in a batch)

{
    
    uint row = gid.y;
    uint column = gid.x;
    
    uint N = uniforms.signalLength;
    uint rowIndex = row * N + column;
    uint columnIndex = column*N + row;
    
    float2 tmp;
    if( row < N-1 )
    {
        if( column > row && column < N ) //from rowIndex+1 to num_cols-1
        {
            //swap A(n,m) with A(m,n)
            tmp = complexBuffer[rowIndex];
            complexBuffer[rowIndex] = complexBuffer[columnIndex];
            complexBuffer[columnIndex] = tmp;
            
        }
    }
    
    
    //for n = 0 to N - 2
    //    for m = n + 1 to N - 1
    
    // read the matrix tile into shared memory
    
    /*
     unsigned int xIndex = gid.x;
     unsigned int yIndex = gid.y;
     
     if((xIndex < uniforms.signalLength) && (yIndex < uniforms.signalLength))
     {
     unsigned int index_in = yIndex * uniforms.signalLength + xIndex;
     int Idin = get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0);
     block[Idin]=  datain[index_in];
     }
     
     barrier(CLK_LOCAL_MEM_FENCE);
     
     // write the transposed matrix tile to global memory
     
     xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
     yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
     
     if((xIndex < height) && (yIndex < width))
     {
     unsigned int index_out = yIndex * height + xIndex;
     int Idout = get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1);
     dataout[index_out] = block[Idout];
     }
     */
    
}

/*
 * Transpose the values in the complex buffer out-of-place
 */
kernel void Transpose_Complex_Buffer_OOP(
                                     device float2 *complexBuffer [[ buffer(0) ]],
                                     device float2 *outBuffer [[ buffer(1) ]],
                                     device FFTImageFilterUniforms &uniforms [[buffer(2)]],
                                     uint2 gid [[thread_position_in_grid]] )

// width = N (signal length)
// height = batch_size (number of signals in a batch)

{
    
    uint row = gid.y;
    uint column = gid.x;
    
    uint N = uniforms.signalLength;
    uint rowIndex = row * N + column;
    uint columnIndex = column*N + row;
    
    //threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 tmp;
    if( row < N-1 )
    {
        if( column > row && column < N ) //from rowIndex+1 to num_cols-1
        {
            //swap A(n,m) with A(m,n)
            tmp = complexBuffer[rowIndex];
            complexBuffer[rowIndex] = complexBuffer[columnIndex];
            complexBuffer[columnIndex] = tmp;
            
        }
    }
    
    //threadgroup_barrier(mem_flags::mem_threadgroup);

    //outBuffer[rowIndex] = complexBuffer[rowIndex];
    //outBuffer[columnIndex] = complexBuffer[columnIndex];
    //threadgroup_barrier(mem_flags::mem_threadgroup);

    //for n = 0 to N - 2
    //    for m = n + 1 to N - 1
    
    // read the matrix tile into shared memory
    
    /*
     unsigned int xIndex = gid.x;
     unsigned int yIndex = gid.y;
     
     if((xIndex < uniforms.signalLength) && (yIndex < uniforms.signalLength))
     {
     unsigned int index_in = yIndex * uniforms.signalLength + xIndex;
     int Idin = get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0);
     block[Idin]=  datain[index_in];
     }
     
     barrier(CLK_LOCAL_MEM_FENCE);
     
     // write the transposed matrix tile to global memory
     
     xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
     yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
     
     if((xIndex < height) && (yIndex < width))
     {
     unsigned int index_out = yIndex * height + xIndex;
     int Idout = get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1);
     dataout[index_out] = block[Idout];
     }
     */
    
}


/*
 * Transpose the values in the complex buffer
 */
kernel void Transpose_Complex_Buffer_Texture(
                                    texture2d<float, access::write> outTexture [[texture(0)]],
                                     device float2 *complexBuffer [[ buffer(0) ]],
                                     device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                                     uint2 gid [[thread_position_in_grid]] )

// width = N (signal length)
// height = batch_size (number of signals in a batch)

{
    
    uint row = gid.y;
    uint column = gid.x;
    
    uint N = uniforms.signalLength;
    uint rowIndex = row * N + column;
    uint columnIndex = column*N + row;
    
    float2 tmp;
    if( row < N-1 )
    {
        if( column > row && column < N ) //from rowIndex+1 to num_cols-1
        {
            //swap A(n,m) with A(m,n)
            tmp = complexBuffer[rowIndex];
            complexBuffer[rowIndex] = complexBuffer[columnIndex];
            complexBuffer[columnIndex] = tmp;
            
            //outTexture.write( float4(complexBuffer[columnIndex].x, complexBuffer[columnIndex].x, complexBuffer[columnIndex].x, 1), gid);
            //outTexture.write( float4(complexBuffer[columnIndex].x, complexBuffer[columnIndex].x, complexBuffer[columnIndex].x, 1), uint2(row, column));
            
        }
    }
    
    //float out = complexBuffer[rowIndex].x;
    //float outC = complexBuffer[rowIndex].y;
    

    //calculate magnitude, log transform, and normalize
    
    float magLog = log( length( complexBuffer[rowIndex] ) + 1.0f );
    //if( gid.x > 0 && gid.y > 0 )
        magLog = magLog/ log( length( complexBuffer[0] + 1.0f ) );
    
    //reposition the quads
    uint Ndiv2 = N/2;
    uint x = gid.x;
    uint y = gid.y;
    if( gid.x >= Ndiv2 )
        x -= Ndiv2;
    else
        x += Ndiv2;
    
    if( gid.y >= Ndiv2 )
        y -= Ndiv2;
    else
        y += Ndiv2;

    outTexture.write( float4(magLog, magLog, magLog, 1), uint2(x,y));
    //outTexture.write( float4( 1,0,0,1), gid);
    //for n = 0 to N - 2
    //    for m = n + 1 to N - 1
    
    // read the matrix tile into shared memory
    
    /*
     unsigned int xIndex = gid.x;
     unsigned int yIndex = gid.y;
     
     if((xIndex < uniforms.signalLength) && (yIndex < uniforms.signalLength))
     {
     unsigned int index_in = yIndex * uniforms.signalLength + xIndex;
     int Idin = get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0);
     block[Idin]=  datain[index_in];
     }
     
     barrier(CLK_LOCAL_MEM_FENCE);
     
     // write the transposed matrix tile to global memory
     
     xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
     yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
     
     if((xIndex < height) && (yIndex < width))
     {
     unsigned int index_out = yIndex * height + xIndex;
     int Idout = get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1);
     dataout[index_out] = block[Idout];
     }
     */
    
}


/*
 * Take the magnitude of each complex pair in the complex buffer
 */
kernel void Magnitude_Complex_Buffer(
                                     device float2 *complexBuffer [[ buffer(0) ]],
                                     device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                                     uint2 gid [[thread_position_in_grid]] )

// width = N (signal length)
// height = batch_size (number of signals in a batch)

{
    
    uint row = gid.y * uniforms.signalLength;
    uint column = gid.x;
    
    float2 complexValue = complexBuffer[ row+column ];
    complexBuffer[ row + column] = length(complexValue);
    
}

/*
 * Take the log of the real values of each complex pair in the complex buffer
 */
kernel void Log_Complex_Buffer(
                               device float2 *complexBuffer [[ buffer(0) ]],
                               device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]] )

// width = N (signal length)
// height = batch_size (number of signals in a batch)

{
    
    uint row = gid.y * uniforms.signalLength;
    uint column = gid.x;
    
    float2 complexValue = complexBuffer[ row+column ];
    complexBuffer[ row + column] = log(complexValue);
    
}

/*
 * Take the magnitude of each complex pair in the complex buffer
 * Then take the log of the real values of each complex pair in the complex buffer
 * Place the output in the real values of each complex value pair
 */
kernel void Mag_Log_Complex_Buffer(
                                   device float2 *complexBuffer [[ buffer(0) ]],
                                   device FFTImageFilterUniforms &uniforms [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]] )

// width = N (signal length)
// height = batch_size (number of signals in a batch)

{

/*
    uint lastRowIndex = uniforms.signalLength - 1;
    uint lastColumnIndex = uniforms.signalLength - 1;
    
 
    uint lastTextureIndex = uniforms.signalLength * lastRowIndex + lastColumnIndex;
    
    if( gid.x == 0 && gid.y == 0 )
    {
        //if this is our first time in the kernel,
        //calculate the magnitude and log of the last fft pixel value
        //so we can use it to find and store the min and max values
        //as we iterate over each pixel via the kernel calls
        float2 complexValue = complexBuffer[ lastTextureIndex ];
        
        float magLog = log(length(complexValue) + 1.0f);
        float2 mLog = float2( magLog, magLog);
        complexBuffer[ lastTextureIndex ] = mLog;
        
    }
    
    float2 minMax = complexBuffer[lastTextureIndex];
    
    float minVal = minMax.x;
    float maxVal = minMax.y;
    


    if( gid.x >= lastColumnIndex && gid.y >= lastRowIndex )
    {
        
    }
    else
    {
 
    
        uint row = gid.y * uniforms.signalLength;
        uint column = gid.x;
        
        float2 complexValue = complexBuffer[ row+column ];
        
        float magLog = log(length(complexValue) + 1.0f);
        complexBuffer[ row + column ].x = magLog;
    
        
        if( magLog > maxVal )
        {
            float2 minMax = complexBuffer[lastTextureIndex];
            minMax.y = magLog;
            complexBuffer[ lastTextureIndex ] = minMax;
        }
        if( magLog < minVal )
        {
            float2 minMax = complexBuffer[lastTextureIndex];
            minMax.x = magLog;
            complexBuffer[ lastTextureIndex ] = minMax;
        }
     
    }
*/
}

# pragma mark -- Pack/Unpack Texture before/after FFT processing

/*
 * Pack the complex rg32 texture for FFT calculation
 *
 */
kernel void Pack_Complex_Texture(texture2d<float, access::read> inTexture [[texture(0)]],
                                 /*texture2d<float, access::read> outTexture [[texture(1)]],*/
                                 texture2d<float, access::write>  complexTexture [[texture(1)]],
                                 uint2 gid [[thread_position_in_grid]] )
{
    /*
     int X = get_global_id(0);
     float2 V;
     V.x = Data_In [X];
     V.y = Data_In [X+N];
     Data_Out [X] = V;
     */
    
    //convert to grayscale
    float4 image_color = inTexture.read(gid);//tex.sample(s_quad, in.uv);
    float grayColor = dot(image_color.rgb, float3(0.30h, 0.59h, 0.11h));
    
    //our complexTexture is of type RG32 (float2), but we can only write float4 using .write function
    float4 outColor(grayColor, 0, 0, 1);
    complexTexture.write(outColor, gid);
    
}

