//
//  MetalFFTImageFilter.m
//  FFTMetal
//
//  Created by MACMaster on 12/17/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import "MetalFFTImageFilter.h"
#import "FFTMath.h"

struct FFTImageFilterUniforms
{
    uint signalLength;      // N
    uint kernelSize;        // K_W
    uint fftStageIndex;     // Ns
    uint sourceWidth;
    uint sourceHeight;
};


static const int maxComplexBuffers = 2;

@interface MetalFFTImageFilter()
{
    unsigned int _fftDimension;

    id <MTLBuffer> _complexBuffers[maxComplexBuffers];

}

//define some properties to store kernel shader lookups
@property (nonatomic, retain) id<MTLComputePipelineState> packComplexBufferKernel;
@property (nonatomic, retain) id<MTLComputePipelineState> transposeComplexBufferKernel;
@property (nonatomic, retain) id<MTLComputePipelineState> transposeComplexBufferToTextureKernel;

@property (nonatomic) MTLSize maxThreadsPerThreadGroup;
@property (nonatomic) uint maxThreadWidth;
@property (nonatomic) uint fftDimension;                        //we require processing on an NxN buffer/image where N is a power of 2 in order to perform radix-2 FFT

@property (nonatomic) uint numFFTStages;                        //numFFTStages = log(fftDimension); fftDimension = 2^^numFFTStages; not the same as fftStages.count
@property (nonatomic, retain) NSMutableArray * fftPipelineStages;       //contains the pipeline stages (kernel functions) for the "recursive" radix-2 fft routine
@property (nonatomic, retain) NSMutableArray * fftPipelineUniforms;

//@property (nonatomic, strong) id<MTLBuffer> complexBuffer;/

@property (nonatomic, strong) id<MTLBuffer> fftUniformReuseBuffer;
@property (nonatomic, strong) id<MTLBuffer> complexBuffer2;


@property (nonatomic) uint kernelSize;                                  //the largest fft kernel size we will use
@property (nonatomic) uint finalFFTStageKernelSize;                     // the kernel size we will use for the final fft stage

@end

@implementation MetalFFTImageFilter



+ (instancetype)filterWithContext:(MetalGPGPUContext *)context
{
    return [[self alloc] initWithContext:context];
}

- (instancetype)initWithContext:(MetalGPGPUContext *)context
{
    //call the parent MetalImageFilter constructor, which requires a single kernel shader
    //though we won't use it
    if ((self = [super initWithFunctionName:@"Pack_Complex_Buffer" context:context]))
    {
        //configure our FFT textures based on the size of the input image set via the texture provider in our base class
        
        //add additional pipeline stages to chain kernel functions
        
        [self createKernelShaderReferences];
        
        //[self updateOutputTextureSize];
    }
    return self;
}

-(void)createKernelShaderReferences
{
    NSError * error = nil;

    /*
     *  Pack the complex buffer with values from the source texture
     */
    id<MTLFunction> packComplexKernelFunction = [self.context.library newFunctionWithName:@"Pack_Complex_Buffer"];
    id<MTLComputePipelineState> packComplexPipelineStage = [self.context.device newComputePipelineStateWithFunction:packComplexKernelFunction error:&error];
    
    if (!packComplexPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", packComplexKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;// nil;
    }
    else
    {
        //[self.pipelineStages addObject:packComplexPipelineStage];
        self.packComplexBufferKernel = packComplexPipelineStage;
    }

    /*
     *  Transpose the image after performing 1D FFT on the rows once
     */
    id<MTLFunction> transposeComplexKernelFunction = [self.context.library newFunctionWithName:@"Transpose_Complex_Buffer"];
    id<MTLComputePipelineState> transposeComplexPipelineStage = [self.context.device newComputePipelineStateWithFunction:transposeComplexKernelFunction error:&error];
    
    if (!transposeComplexPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", transposeComplexKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;// nil;
    }
    else
    {
        [self.pipelineStages addObject:transposeComplexPipelineStage];
        self.transposeComplexBufferKernel = transposeComplexPipelineStage;
    }
    
    /*
     *  Take the magnitude and log of the complex FFT buffer for visualization purposes
     */
    id<MTLFunction> magLogComplexKernelFunction = [self.context.library newFunctionWithName:@"Mag_Log_Complex_Buffer"];
    id<MTLComputePipelineState> magLogComplexPipelineStage = [self.context.device newComputePipelineStateWithFunction:magLogComplexKernelFunction error:&error];
    
    if (!magLogComplexPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", magLogComplexKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;// nil;
    }
    else
        [self.pipelineStages addObject:magLogComplexPipelineStage];
    
    
    /*
     *  Unpack the complex buffer into a Metal Texture
     */
    id<MTLFunction> unpackComplexKernelFunction = [self.context.library newFunctionWithName:@"Unpack_Complex_Buffer"];
    id<MTLComputePipelineState> unpackComplexPipelineStage = [self.context.device newComputePipelineStateWithFunction:unpackComplexKernelFunction error:&error];
    
    if (!unpackComplexPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", unpackComplexKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;// nil;
    }
    else
        [self.pipelineStages addObject:unpackComplexPipelineStage];
    
    /*
     *  Transpose the image after performing 1D FFT on the rows once
     */
    id<MTLFunction> transposeComplexTextureKernelFunction = [self.context.library newFunctionWithName:@"Transpose_Complex_Buffer_Texture"];
    id<MTLComputePipelineState> transposeComplexTexturePipelineStage = [self.context.device newComputePipelineStateWithFunction:transposeComplexTextureKernelFunction error:&error];
    
    if (!transposeComplexTexturePipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", transposeComplexTextureKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;// nil;
    }
    else
    {
        [self.pipelineStages addObject:transposeComplexTexturePipelineStage];
        self.transposeComplexBufferToTextureKernel = transposeComplexTexturePipelineStage;
    }
    
    /*
     *  Transpose the image out of place
     */
    id<MTLFunction> transposeComplexBufferOOPKernelFunction = [self.context.library newFunctionWithName:@"Transpose_Complex_Buffer_OOP"];
    id<MTLComputePipelineState> transposeComplexBufferOOPPipelineStage = [self.context.device newComputePipelineStateWithFunction:transposeComplexBufferOOPKernelFunction error:&error];
    
    if (!transposeComplexBufferOOPPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", transposeComplexBufferOOPKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;// nil;
    }
    else
        [self.pipelineStages addObject:transposeComplexBufferOOPPipelineStage];
}

- (void)updateOutputTextureSize
{

    //first, get the largest image dimension
    CGFloat largestDim = self.provider.texture.width;
    
    if( largestDim < self.provider.texture.height )
        largestDim = self.provider.texture.height;
    
    int largestDimInt = (int)largestDim;
    
    _fftDimension = getNextPowerOfTwo(largestDimInt);
    
    //NSLog(@"FFT Dimension:  %d", _fftDimension);
    //assume the input texture is the same size as that of the output texture
    //since that is what is implemented in our base class
    
    //create o;ur output texture if needed
    //need a square image that is a power of 2 pixels
    
    
    /*
     *  Update the base class output texture to be the size of fft dimension NxN
     *  Also, generate an FFT plan now that we know the FFT dimension
     */
    if (!self.internalTexture ||
        [self.internalTexture width] != _fftDimension ||
        [self.internalTexture height] != _fftDimension)
    {

        id<MTLTexture> inputTexture = self.provider.texture;
        
        MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:[inputTexture pixelFormat]
                                                                                                     width:_fftDimension
                                                                                                    height:_fftDimension
                                                                                                 mipmapped:NO];
        self.internalTexture = [self.context.device newTextureWithDescriptor:textureDescriptor];
    
        NSLog(@"Created Output Texture with FFT Dimension:   %d", _fftDimension);
        
        self.maxThreadsPerThreadGroup = self.context.device.maxThreadsPerThreadgroup;
        self.maxThreadWidth = (uint)self.maxThreadsPerThreadGroup.width;
        
        [self generateFFTPlan];

    }
    

    
    //NSLog(@"FFT Texture Dimension:  %d", _fftDimension);
}

/*
- (void)setThreadgroupSize
{
    //configure Metal threadgrouping to dispatch a single thread per row in the texture
    //that is, each thread will call the kernel function for a row in the image
    
    //query the max threadgroup size
    
    MTLSize maxThreadsPerThreadGroup = self.context.device.maxThreadsPerThreadgroup;
    
    NSLog(@"Max Thread Width:   %lu", (unsigned long)maxThreadsPerThreadGroup.width);
    NSLog(@"Max Thread Height:   %lu", (unsigned long)maxThreadsPerThreadGroup.height);
    NSLog(@"Max Thread Depth:   %lu", (unsigned long)maxThreadsPerThreadGroup.depth);
    
    NSUInteger rowSize = _fftDimension;
    
    //_numRowsPerThreadGroup = _fftDimension
    
    //self.threadgroupSize = MTLSize( maxThreadsPerThreadGroup.width

}
*/

/*
 * Pack the complex buffer(s) with the source texture
 */
-(void)packComplexBufferWithCommandEncoder:commandEncoder
{
    
    //commit multiple kernel functions to chain them
    //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:self.packComplexBufferKernel];
    
    
    /*
     *  Pack the source texture into an FFT Texture (or Buffer) containing Real and Complex planes (each 32 bit float)
     */
    //[self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];            //bind the complex buffer(s) we will use for in-place fft calculation
    [self bindComplexBuffer:0 atIndex:0 withCommandEncoder:commandEncoder];
    [commandEncoder setTexture:self.provider.texture atIndex:0];                     //bind the input texture to the kernel shader
    
    if( !self.fftUniformReuseBuffer )
    {
        struct FFTImageFilterUniforms uniforms;
        uniforms.signalLength = _fftDimension;
        uniforms.kernelSize = 0;
        uniforms.fftStageIndex = 0;
        //uniforms.threadGroupIndex = 0;
        
        
        self.fftUniformReuseBuffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceStorageModeShared];
        
        /*
         *  Update constant uniform buffer memory
         */
        memcpy([self.fftUniformReuseBuffer contents], &uniforms, sizeof(uniforms));
        
    }
    
    /*
     *  Attach the constant uniform buffer for passing to kernel function
     */
    [commandEncoder setBuffer:self.fftUniformReuseBuffer offset:0 atIndex:1];
    
    //configure GPU threading for the kernel function
    MTLSize kThreadgroupSize = MTLSizeMake(16, 16, 1);
    NSUInteger nThreadCountW = (_fftDimension /*self.provider.texture.width*/ + kThreadgroupSize.width - 1 ) / kThreadgroupSize.width;
    NSUInteger nThreadCountH = (_fftDimension /*self.provider.texture.height*/ + kThreadgroupSize.height - 1) / kThreadgroupSize.height;
    MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
    //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
    
    [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];

}

/*
 * Unpack the Complex Buffer for visual rendering or other use when we are finished with FFT processing
 */
-(void)unpackComplexBufferWithCommandEncoder:commandEncoder
{

    /*
     *  Take the magnitude and log of the complex buffer after performing 2D FFT
     */
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)[self.pipelineStages objectAtIndex:2];
    
    //commit multiple kernel functions to chain them
    //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:pipeline];
    
    //[commandEncoder setTexture:self.internalTexture atIndex:0];                     //bind the output texture to the kernel shader
    //[commandEncoder setTexture:self.internalTexture atIndex:1];                     //bind the output texture to the kernel shader
    //[self bindFFTTextureWithCommandEncoder:commandEncoder atIndex:0];                 //bind the packed FFT input texture ready to be processed
    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];
    
    struct FFTImageFilterUniforms uniforms;
    uniforms.signalLength = _fftDimension;
    uniforms.kernelSize = 0;
    uniforms.fftStageIndex = 0;//pow(2, self.numFFTStages);
    //uniforms.threadGroupIndex = 0;
    
    id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];

    /*
     *  Update constant uniform buffer memory
     */
    memcpy([buffer contents], &uniforms, sizeof(uniforms));
    
    
    /*
     *  Attach the constant uniform buffer for passing to kernel function
     */
    [commandEncoder setBuffer:buffer offset:0 atIndex:1];
    
    
    
    //configure GPU threading for the kernel function
    MTLSize kThreadgroupSize = MTLSizeMake(1, 1, 1);
    MTLSize kThreadgroups = MTLSizeMake((_fftDimension + kThreadgroupSize.width - 1) / kThreadgroupSize.width, (_fftDimension + kThreadgroupSize.height - 1) / kThreadgroupSize.height, 1);
    
    //NSLog(@"self.internalTexture w: %d, h:  %d", [self.internalTexture width], [self.internalTexture height]);
    [commandEncoder dispatchThreadgroups:kThreadgroups threadsPerThreadgroup:kThreadgroupSize];
    
    

    /*
     *  Unpack the complex buffer into a visual texture
     */
    /*id<MTLComputePipelineState>*/ pipeline = (id<MTLComputePipelineState>)[self.pipelineStages objectAtIndex:3];
    
    //commit multiple kernel functions to chain them
    //commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:pipeline];
    
    [commandEncoder setTexture:self.internalTexture atIndex:0];                     //bind the output texture to the kernel shader
    //[commandEncoder setTexture:self.internalTexture atIndex:1];                     //bind the output texture to the kernel shader
    //[self bindFFTTextureWithCommandEncoder:commandEncoder atIndex:0];                 //bind the packed FFT input texture ready to be processed
    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];
    
    /*
     struct FFTImageFilterUniforms uniforms;
     uniforms.signalLength = _fftDimension;
     uniforms.kernelSize = 0;
     uniforms.fftStageIndex = 0;
     //uniforms.threadGroupIndex = 0;
     */
    
    //id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];

    
    /*
     *  Update constant uniform buffer memory
     */
    //memcpy([buffer contents], &uniforms, sizeof(uniforms));
    
    
    /*
     *  Attach the constant uniform buffer for passing to kernel function
     */
    [commandEncoder setBuffer:buffer offset:0 atIndex:1];
    
    
    
    //configure GPU threading for the kernel function
     kThreadgroupSize = MTLSizeMake(1, 1, 1);
     kThreadgroups = MTLSizeMake(([self.internalTexture width] + kThreadgroupSize.width - 1) / kThreadgroupSize.width, ([self.internalTexture height] + kThreadgroupSize.height - 1) / kThreadgroupSize.height, 1);
    
    //NSLog(@"self.internalTexture w: %d, h:  %d", [self.internalTexture width], [self.internalTexture height]);
    [commandEncoder dispatchThreadgroups:kThreadgroups threadsPerThreadgroup:kThreadgroupSize];
    

}

/*
 * Unpack the complex buffer into multiple complex buffers
 */
-(void)deinterleaveComplexBufferWithCommandEncoder:commandEncoder
{
    NSError * error;
    id<MTLFunction> packComplexKernelFunction = [self.context.library newFunctionWithName:@"Deinterleave_Complex_2"];
    id<MTLComputePipelineState> packComplexPipelineStage = [self.context.device newComputePipelineStateWithFunction:packComplexKernelFunction error:&error];
    
    if (!packComplexPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", packComplexKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;
    }
    
    //commit multiple kernel functions to chain them
    //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:packComplexPipelineStage];
    
    
    /*
     *  Pack the source texture into an FFT Texture (or Buffer) containing Real and Complex planes (each 32 bit float)
     */
    [commandEncoder setTexture:self.provider.texture atIndex:0];                     //bind the input texture to the kernel shader
    //[self bindComplexBuffersAtIndex:0 withCommandEncoder:commandEncoder];            //bind the complex buffer(s) we will use for in-place fft calculation
    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:2];
    
    struct FFTImageFilterUniforms uniforms;
    uniforms.signalLength = _fftDimension;//self.maxThreadWidth;
    uniforms.kernelSize = 0;
    uniforms.fftStageIndex = 0;
    //uniforms.threadGroupIndex = 0;
    
    id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];
    
    /*
     *  Update constant uniform buffer memory
     */
    memcpy([buffer contents], &uniforms, sizeof(uniforms));
    
    
    /*
     *  Attach the constant uniform buffer for passing to kernel function
     */
    [commandEncoder setBuffer:buffer offset:0 atIndex:3];
    
    //configure GPU threading for the kernel function
    MTLSize kThreadgroupSize = MTLSizeMake(1, 1, 1);
    NSUInteger nThreadCountW = (_fftDimension + kThreadgroupSize.width - 1 ) / kThreadgroupSize.width;
    NSUInteger nThreadCountH = (_fftDimension + kThreadgroupSize.height - 1) / kThreadgroupSize.height;
    MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
    //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
    
    [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
    
}

-(void)interleaveComplexBufferWithCommandEncoder:commandEncoder
{
    NSError * error;
    /*
     *  Merge Even/Odd Complex Buffers back into a single Complex Buffer
     */
    id<MTLFunction> mergeKernelFunction = [self.context.library newFunctionWithName:@"Interleave_Complex_2"];
    
    id<MTLComputePipelineState> mergeComplexPipelineStage = [self.context.device newComputePipelineStateWithFunction:mergeKernelFunction error:&error];
    
    if (!mergeComplexPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", mergeKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;
    }
    
    //commit multiple kernel functions to chain them
    //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:mergeComplexPipelineStage];
    
    //[self bindComplexBuffersAtIndex:0 withCommandEncoder:commandEncoder];            //bind the packed fft input buffer ready to be processed
    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:2];
    
    /*
     *  Bind a constant uniform buffer defining paramers used for 1D twist kernel at each subsequent stage
     */
    
    //create an FFT buffer with parameters to pass to the fft kernel being executed
    struct FFTImageFilterUniforms uniforms;
    uniforms.signalLength = self.maxThreadWidth;
    //uniforms.kernelSize = kSize;
    
    //float ncsFloat = pow(2.f, (float)numCompletedStages);
    //NSLog(@"Ns Value:   %f", ncsFloat);
    //uniforms.fftStageIndex = (uint)ncsFloat;
    
    //NSLog(@"uniforms.fftStage = %u", uniforms.fftStage);
    //uniforms.threadGroupIndex = 0;
    
    id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];
    
    /*
     *  Update constant uniform buffer memory
     */
    memcpy([buffer contents], &uniforms, sizeof(uniforms));
    
    /*
     *  Attach the constant uniform buffer for passing to kernel function
     */
    [commandEncoder setBuffer:buffer offset:0 atIndex:3];
    
    //configure GPU threading for the kernel function for size of the fft complex strips to be interleaved
    MTLSize kThreadgroupSize = MTLSizeMake(self.maxThreadWidth, 1, 1);
    NSUInteger nThreadCountW = (self.maxThreadWidth + kThreadgroupSize.width - 1 ) / kThreadgroupSize.width;
    NSUInteger nThreadCountH = (_fftDimension + kThreadgroupSize.height - 1) / kThreadgroupSize.height;
    MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
    //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
    
    [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
}


/*
-(void)renderPipelineStages
{
    //populate the metal command buffer
    id<MTLCommandBuffer> commandBuffer = [self.context.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];


    //1:  Pack the complex buffer(s) with the source texture
    [self packComplexBufferWithCommandEncoder:commandEncoder];
    

    [self render1DFFTAndTranspose2:commandEncoder];
    
    //[self deinterleaveComplexBufferWithCommandEncoder:commandEncoder];
    //[self render1DFFTAndTranspose2:commandEncoder];

    
    [self unpackComplexBufferWithCommandEncoder:commandEncoder];

    //Finished Defining Pipeline Kernel Stages: End Encoding
    [commandEncoder endEncoding];
    
    //commit the command queue to Metal Land
    [commandBuffer commit];
    //synchronously wait until processing has completed
    [commandBuffer waitUntilCompleted];
}
*/


-(void)renderPipelineStages
{
    id<MTLCommandBuffer> commandBuffer = [self.context.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    
    //unpack the values from the source input texture, and put them in a complex float2 buffer
    [self packComplexBufferWithCommandEncoder:commandEncoder];

    //perform the 1D FFT stages and transpose after each FFT pass
    [self render1DFFTAndTranspose:commandEncoder outputToTexture:NO];
    [self render1DFFTAndTranspose:commandEncoder outputToTexture:YES];
    
    //Finished Defining Pipeline Kernel Stages: End Encoding
    [commandEncoder endEncoding];
    
    
    //commit the command queue to Metal Land
    [commandBuffer commit];
    //synchronously wait until processing has completed
    [commandBuffer waitUntilCompleted];
}

-(void)renderPipelineStagesOLD
{
    
    //populate the metal command buffer
    id<MTLCommandBuffer> commandBuffer = [self.context.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    
    //All filters set a single pipeline stage containing a single kernel function on initialization,
    //but child classes can modify the order and insert into pipeline stages to chain kernel functions if necessary
    for (int i=0; i<1/*self.pipelineStages.count*/; i++)
    {
        id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)[self.pipelineStages objectAtIndex:i];
        
        //commit multiple kernel functions to chain them
        //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        [commandEncoder setComputePipelineState:pipeline];
        
        //first kernel stage:  pack the source texture into an FFT Texture (or Buffer) containing Real and Complex planes (each 32 bit float)
        if( i== 0 )
        {
            [commandEncoder setTexture:self.provider.texture atIndex:0];                     //bind the input texture to the kernel shader
            [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];
            
            
            
            if( !self.fftUniformReuseBuffer )
            {
                struct FFTImageFilterUniforms uniforms;
                uniforms.signalLength = _fftDimension;
                uniforms.kernelSize = 0;
                uniforms.fftStageIndex = 0;
                //uniforms.threadGroupIndex = 0;
                
                
                self.fftUniformReuseBuffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceStorageModeShared];
                
                /*
                 *  Update constant uniform buffer memory
                 */
                memcpy([self.fftUniformReuseBuffer contents], &uniforms, sizeof(uniforms));
            
            }
            
            /*
             *  Attach the constant uniform buffer for passing to kernel function
             */
            [commandEncoder setBuffer:self.fftUniformReuseBuffer offset:0 atIndex:1];
            
            //configure GPU threading for the kernel function
            MTLSize kThreadgroupSize = MTLSizeMake(8, 8, 1);
            NSUInteger nThreadCountW = (self.provider.texture.width + kThreadgroupSize.width - 1 ) / kThreadgroupSize.width;
            NSUInteger nThreadCountH = (self.provider.texture.height + kThreadgroupSize.height - 1) / kThreadgroupSize.height;
            MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
            //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
            
            [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
            
        }
        else if (i == 1 )
        {
            
            
            //[self bindThreadGroupBufferWithCommandEncoder:commandEncoder];
            //[self bindFFTTextureWithCommandEncoder:commandEncoder atIndex:0];                 //bind the packed FFT input texture ready to be processed
            //[commandEncoder setTexture:self.provider.texture atIndex:0];                     //bind the input texture to the kernel shader
            //[commandEncoder setTexture:self.internalTexture atIndex:1];                     // bind the output texture to the kernel shader
            
            //[self bindUniformBufferWithCommandEncoder:commandEncoder];
            //[self bindThreadGroupBufferWithCommandEncoder:commandEncoder];
            
            //configure GPU threading for the kernel function
            MTLSize kThreadgroupSize = MTLSizeMake(2, 2, 1);
            NSUInteger nThreadCountW = (self.provider.texture.width + kThreadgroupSize.width - 1 ) / kThreadgroupSize.width;
            NSUInteger nThreadCountH = (self.provider.texture.height + kThreadgroupSize.height - 1) / kThreadgroupSize.height;
            MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
            //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
            
            [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:MTLSizeMake(2,2,1)];
            //[commandEncoder dispatchThreadgroupsWithIndirectBuffer:self.uniformBuffer indirectBufferOffset:0 threadsPerThreadgroup:kThreadgroupSize];
            
            //do any custom configuration for the filter by calling the virtual overloaded functions from child class implementations
            //[self configureArgumentTableWithCommandEncoder:commandEncoder];
        }
        
        //[commandEncoder endEncoding];
        
    }
    
    [self render1DFFTAndTranspose:commandEncoder outputToTexture:NO];
    [self render1DFFTAndTranspose:commandEncoder outputToTexture:YES];
    
    
    
    //configure GPU threading for the kernel function
    //MTLSize kThreadgroupSize = MTLSizeMake(1, 1, 1);
    //MTLSize kThreadgroups = MTLSizeMake((self.provider.texture.width + kThreadgroupSize.width - 1) / kThreadgroupSize.width, (self.provider.texture.height + kThreadgroupSize.height - 1) / kThreadgroupSize.height, 1);
    
    //NSLog(@"self.internalTexture w: %d, h:  %d", [self.internalTexture width], [self.internalTexture height]);
    //[commandEncoder dispatchThreadgroups:kThreadgroups threadsPerThreadgroup:kThreadgroupSize];
    

    
    //Finished Defining Pipeline Kernel Stages: End Encoding
    [commandEncoder endEncoding];
    
    
    //commit the command queue to Metal Land
    [commandBuffer commit];
    //synchronously wait until processing has completed
    [commandBuffer waitUntilCompleted];
    
    
}




-(void)render1DFFTAndTranspose:(id<MTLComputeCommandEncoder>)commandEncoder outputToTexture:(BOOL)outputToTexture
{
    unsigned int kSize = self.kernelSize;
    unsigned int numCompletedStages = 0;
    unsigned int numRemainingStages = self.numFFTStages;
    
    //outputToTexture == true indicates that this is our second fft/transpose pass
    //if this is the case, and fftPipeline stages is odd, we must start with the secondary
    //complex buffer on the second fft/transpose pass
    int inBufferIndex = 0;
    int outBufferIndex = 1;
    
    if( outputToTexture && ( self.fftPipelineStages.count %2 == 1 ) )
    {
        inBufferIndex = 1;
        outBufferIndex = 0;
    }
    
    for (int i=0; i<self.fftPipelineStages.count; i++)
    {
        
        id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)[self.fftPipelineStages objectAtIndex:i];
        
        //commit multiple kernel functions to chain them
        [commandEncoder setComputePipelineState:pipeline];
        
        if( i != (self.fftPipelineStages.count-1) )
        {
            
            if( i%2 == 0 )
            {
                /*
                if( outputToTexture )
                {
                    [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
                    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be
                }
                
                else
                {
                [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
                [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be
                }
                */
                
                [self bindComplexBuffer:inBufferIndex atIndex:0 withCommandEncoder:commandEncoder];
                [self bindComplexBuffer:outBufferIndex atIndex:1 withCommandEncoder:commandEncoder];

            }
            else
            {
                /*
                if( outputToTexture )
                {
                    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
                    [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be
                }
                else
                {
                [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be
                [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be processed
                }
                */
                
                [self bindComplexBuffer:outBufferIndex atIndex:0 withCommandEncoder:commandEncoder];
                [self bindComplexBuffer:inBufferIndex atIndex:1 withCommandEncoder:commandEncoder];
            }
            
            /*
             *  Bind a constant uniform buffer defining paramers used for 1D twist kernel at each subsequent stage
             */
            
            id<MTLBuffer> buffer = [self.fftPipelineUniforms objectAtIndex:i];
            
            
            /*
             *  Attach the constant uniform buffer for passing to kernel function
             */
            [commandEncoder setBuffer:buffer offset:0 atIndex:2];
            
            //configure GPU threading for the kernel function
            
            MTLSize kThreadgroupSize = MTLSizeMake(512, 1, 1);
            
            
            //NSLog(@"Thread Execution Width:   %lu", (unsigned long)maxThreadsPerThreadGroup);
            
            NSUInteger nThreadCountW = ( _fftDimension /*+ kThreadgroupSize.width - 1*/ ) / kThreadgroupSize.width;
            NSUInteger nThreadCountH = ( _fftDimension /*+ kThreadgroupSize.height - 1*/ ) / kThreadgroupSize.height;
            MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
            //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
            
            [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
            
            
            //NSLog(@"Total Number of FFT_8_KERNEL threads:   %d", nThreadCountW);
            //[commandEncoder dispatchThreadgroupsWithIndirectBuffer:self.uniformBuffer indirectBufferOffset:0 threadsPerThreadgroup:kThreadgroupSize];
            
            //do any custom configuration for the filter by calling the virtual overloaded functions from child class implementations
            //[self configureArgumentTableWithCommandEncoder:commandEncoder];
            
            
            if( kSize == 8 )
            {
                numCompletedStages += 3;
                numRemainingStages -= 3;
            }
            else if( kSize == 4 )
            {
                numCompletedStages += 2;
                numRemainingStages -= 2;
            }
            else if ( kSize ==  2 )
            {
                numCompletedStages += 1;
                numRemainingStages -= 1;
            }
            
            
            
            //if( [pipeline.computeFunction.name localizedCompare:[@"FFT_8_KERNEL" ]]
            
        }
        else //last stage
        {
            kSize = self.finalFFTStageKernelSize;
            
            /*
             *  Bind the interleaved Real/Complex Buffer containing our in-place FFT calculation
             */
            
            //[self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];
            
            if( i%2 == 0 )
            {
                /*
                 if( outputToTexture )
                 {
                 [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
                 [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be
                 }
                 
                 else
                 {
                 [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
                 [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be
                 }
                 */
                
                [self bindComplexBuffer:inBufferIndex atIndex:0 withCommandEncoder:commandEncoder];
                [self bindComplexBuffer:outBufferIndex atIndex:1 withCommandEncoder:commandEncoder];
                
            }
            else
            {
                /*
                 if( outputToTexture )
                 {
                 [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
                 [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be
                 }
                 else
                 {
                 [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be
                 [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be processed
                 }
                 */
                
                [self bindComplexBuffer:outBufferIndex atIndex:0 withCommandEncoder:commandEncoder];
                [self bindComplexBuffer:inBufferIndex atIndex:1 withCommandEncoder:commandEncoder];
            }
            
            /*
             *  Bind a constant uniform buffer defining paramers used for 1D twist kernel at each subsequent stage
             */
            
            id<MTLBuffer> buffer = [self.fftPipelineUniforms objectAtIndex:i];
            
            
            /*
             *  Attach the constant uniform buffer for passing to kernel function
             */
            [commandEncoder setBuffer:buffer offset:0 atIndex:2];

            
            
            /*
             *  Configure GPU threading for the kernel function
             */
            MTLSize kThreadgroupSize = MTLSizeMake(512, 1, 1);
            NSUInteger nThreadCountW = ( _fftDimension /*+ kThreadgroupSize.width - 1*/ ) / kThreadgroupSize.width;
            NSUInteger nThreadCountH = ( _fftDimension /*+ kThreadgroupSize.height - 1*/ ) / kThreadgroupSize.height;
            MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
            //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
            
            [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
            
            
            if( self.finalFFTStageKernelSize == 2 )
            {
                numCompletedStages += 1;
                numRemainingStages -= 1;
            }
            else if( self.finalFFTStageKernelSize == 4 )
            {
                numCompletedStages += 2;
                numRemainingStages -= 2;
            }
            else if( self.finalFFTStageKernelSize == 8 )
            {
                numCompletedStages += 3;
                numRemainingStages -= 3;
            }
            
        }
        
        
        //[commandEncoder endEncoding];
        
    }
    
    //after we are done performing fft stages, prepare for transpose
    //if num fft stages is odd, perform the transpose with the secondary complex buffer
    //as the input buffer (so swap the indices here)
    int transposeBufferIndex = inBufferIndex;
    if( self.fftPipelineStages.count %2 == 1  )
    {
        transposeBufferIndex = outBufferIndex;
    }
    
    /*
     *  Transpose the image after performing 1D FFT on the rows once
     */

    id<MTLComputePipelineState> pipeline;// = (id<MTLComputePipelineState>)[self.pipelineStages objectAtIndex:pipelineIndex];

    //if outputting to texture, ie this is our second fft/transpose pass
    //use the appropriate kernel shader
    if( outputToTexture )
        pipeline = self.transposeComplexBufferToTextureKernel;
    else
        pipeline = self.transposeComplexBufferKernel;

    //set the appropriate transpose kernel, always use an in-place tranpose
    //but we might output to visual texture as well if this is the last pass
    [commandEncoder setComputePipelineState:pipeline];
    
    if( outputToTexture )
    {
        //bind the output texture to the kernel shader
        [commandEncoder setTexture:self.internalTexture atIndex:0];
    }

    //bind the appropriate complex buffer containing fft output for in place tranpose
    [self bindComplexBuffer:transposeBufferIndex atIndex:0 withCommandEncoder:commandEncoder];

    if( !self.fftUniformReuseBuffer )
    {
        struct FFTImageFilterUniforms uniforms;
        uniforms.signalLength = _fftDimension;
        uniforms.kernelSize = 0;
        uniforms.fftStageIndex = 0;
        //uniforms.threadGroupIndex = 0;
        
        
        self.fftUniformReuseBuffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceStorageModeShared];
        
        /*
         *  Update constant uniform buffer memory
         */
        memcpy([self.fftUniformReuseBuffer contents], &uniforms, sizeof(uniforms));
        
    }
    

    /*
     *  Attach the constant uniform buffer for passing to kernel function
     */
    //bind the fft uniform reuse buffer (it will only be used to tell the kernel shader the fft signal size i.e. image width )
    [commandEncoder setBuffer:self.fftUniformReuseBuffer offset:0 atIndex:1];

    
    //configure GPU threading for the kernel function
    MTLSize kThreadgroupSize = MTLSizeMake(512, 1, 1);
    MTLSize kThreadgroups = MTLSizeMake((_fftDimension /*+ kThreadgroupSize.width - 1*/) / kThreadgroupSize.width, (_fftDimension /*+ kThreadgroupSize.height - 1*/) / kThreadgroupSize.height, 1);
    
    //NSLog(@"self.internalTexture w: %d, h:  %d", [self.internalTexture width], [self.internalTexture height]);
    [commandEncoder dispatchThreadgroups:kThreadgroups threadsPerThreadgroup:kThreadgroupSize];
    
    
    
    
    
}

-(void)render1DFFTAndTransposeKernel2:(id<MTLComputeCommandEncoder>)commandEncoder
{
    unsigned int kSize = 2;
    unsigned int numCompletedStages = 0;
    unsigned int maxThreadGroupPowerOf2 = 8;
    
    for (int i=0; i<self.fftPipelineStages.count; i++)
    {
        
        
        id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)[self.fftPipelineStages objectAtIndex:i];
        
        //commit multiple kernel functions to chain them
        //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        [commandEncoder setComputePipelineState:pipeline];
        
        //NSLog(@"Executing Stage %d", i+1);
        if( i != (self.fftPipelineStages.count-1) )
        {
            
            //[self bindThreadGroupBufferWithCommandEncoder:commandEncoder];
            [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
            //[self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be processed
            
            //[self bindFFTTextureWithCommandEncoder:commandEncoder atIndex:0];                 //bind the packed FFT input texture ready to be processed
            
            /*
             *  Bind a constant uniform buffer defining paramers used for 1D twist kernel at each subsequent stage
             */
            //create an FFT buffer with parameters to pass to the kernel
            struct FFTImageFilterUniforms uniforms;
            uniforms.signalLength = _fftDimension;
            uniforms.kernelSize = kSize;
            
            float ncsFloat = pow(2.f, (float)numCompletedStages);
            //NSLog(@"Ns Value:   %f", ncsFloat);
            uniforms.fftStageIndex = (uint)ncsFloat;
            
            //NSLog(@"uniforms.fftStage = %u", uniforms.fftStage);
            //uniforms.threadGroupIndex = 0;
            
            id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];
            
            /*
             *  Update constant uniform buffer memory
             */
            memcpy([buffer contents], &uniforms, sizeof(uniforms));
            
            
            /*
             *  Attach the constant uniform buffer for passing to kernel function
             */
            [commandEncoder setBuffer:buffer offset:0 atIndex:1];
            //configure GPU threading for the kernel function
            
            NSUInteger threadgroupWidth = 512;//pow(2.f, (float)maxThreadGroupPowerOf2);
            MTLSize kThreadgroupSize = MTLSizeMake(threadgroupWidth, 1, 1);
            
            
            //NSLog(@"Thread Execution Width:   %lu", (unsigned long)maxThreadsPerThreadGroup);
            
            NSUInteger nThreadCountW = ( _fftDimension /*+ kThreadgroupSize.width - 1*/ ) / kThreadgroupSize.width;
            NSUInteger nThreadCountH = ( _fftDimension /*+ kThreadgroupSize.height - 1*/ ) / kThreadgroupSize.height;
            MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
            //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
            
            [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
            
            
            //NSLog(@"Total Number of FFT_8_KERNEL threads:   %d", nThreadCountW);
            //[commandEncoder dispatchThreadgroupsWithIndirectBuffer:self.uniformBuffer indirectBufferOffset:0 threadsPerThreadgroup:kThreadgroupSize];
            
            //do any custom configuration for the filter by calling the virtual overloaded functions from child class implementations
            //[self configureArgumentTableWithCommandEncoder:commandEncoder];
            
            numCompletedStages += 1;
            maxThreadGroupPowerOf2 -= 1;
            
            //if( [pipeline.computeFunction.name localizedCompare:[@"FFT_8_KERNEL" ]]
            
        }
        else //last stage
        {
            kSize = self.finalFFTStageKernelSize;
            //kSize *= 2;  //kernel size doubles at each stage until we reach N/2 kernel size
            
            /*
             *  Bind the interleaved Real/Complex Buffer containing our in-place FFT calculation
             */
            
            //if( i%2 == 1 )
            //{
            //    [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
            //    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be processed
            
            //}
            //else
            // {
            [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];            //bind the packed fft input buffer ready to be processed
            //[self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:1];            //bind the packed fft input buffer ready to be processed
            //   }
            /*
             *  Bind a constant uniform buffer defining paramers used for 1D twist kernel at each subsequent stage
             */
            //create an FFT buffer with parameters to pass to the kernel
            struct FFTImageFilterUniforms uniforms;
            uniforms.signalLength = _fftDimension;
            uniforms.kernelSize = kSize;
            float ncsFloat = pow(2.f, (float)numCompletedStages);
            //NSLog(@"Ns Value:   %f", ncsFloat);
            uniforms.fftStageIndex = (uint)ncsFloat;
            
            //uniforms.threadGroupIndex = 0;
            
            id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];
            
            /*
             *  Update constant uniform buffer memory
             */
            memcpy([buffer contents], &uniforms, sizeof(uniforms));
            
            
            /*
             *  Attach the constant uniform buffer for passing to kernel function
             */
            [commandEncoder setBuffer:buffer offset:0 atIndex:1];
            /*
             *  Configure GPU threading for the kernel function
             */
            NSUInteger threadgroupWidth = 512;//pow(2.f, (float)maxThreadGroupPowerOf2);
            MTLSize kThreadgroupSize = MTLSizeMake(threadgroupWidth, 1, 1);
            NSUInteger nThreadCountW = ( _fftDimension /*+ kThreadgroupSize.width - 1*/ ) / kThreadgroupSize.width;
            NSUInteger nThreadCountH = ( _fftDimension /*+ kThreadgroupSize.height - 1*/ ) / kThreadgroupSize.height;
            MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
            //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
            
            [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
            
            
            if( self.finalFFTStageKernelSize == 2 )
                numCompletedStages += 1;
            else if( self.finalFFTStageKernelSize == 4 )
                numCompletedStages += 2;
            else
                numCompletedStages += 3;
            
            maxThreadGroupPowerOf2 -= 1;

        }
        
        
        //[commandEncoder endEncoding];
        
    }
    
    /*
     *  Copy ComplexBuffer2 to ComplexBuffer
     */
    //id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)[self.pipelineStages objectAtIndex:1];
    /*
     id<MTLFunction> kernelFunction = [self.context.library newFunctionWithName:@"Copy_Complex_Buffer"];
     id<MTLComputePipelineState> pipeline = [self.context.device newComputePipelineStateWithFunction:kernelFunction error:nil];
     
     
     //commit multiple kernel functions to chain them
     //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
     [commandEncoder setComputePipelineState:pipeline];
     
     //[commandEncoder setTexture:self.internalTexture atIndex:0];                     //bind the output texture to the kernel shader
     //[commandEncoder setTexture:self.internalTexture atIndex:1];                     //bind the output texture to the kernel shader
     //[self bindFFTTextureWithCommandEncoder:commandEncoder atIndex:0];                 //bind the packed FFT input texture ready to be processed
     [self bindComplexBuffer2WithCommandEncoder:commandEncoder atIndex:0];
     [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:1];
     
     struct FFTImageFilterUniforms uniforms;
     uniforms.signalLength = _fftDimension;
     uniforms.kernelSize = kSize;
     uniforms.fftStage = pow(2, self.fftStages.count+3);
     //uniforms.threadGroupIndex = 0;
     
     if( !self.uniformBuffer )
     {
     self.uniformBuffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];
     }
     
     
     //  Update constant uniform buffer memory
     memcpy([self.uniformBuffer contents], &uniforms, sizeof(uniforms));
     
     //  Attach the constant uniform buffer for passing to kernel function
     
     [commandEncoder setBuffer:self.uniformBuffer offset:0 atIndex:2];
     */
    
    
    //configure GPU threading for the kernel function
    MTLSize kThreadgroupSize = MTLSizeMake(1, 1, 1);
    MTLSize kThreadgroups = MTLSizeMake((_fftDimension + kThreadgroupSize.width - 1) / kThreadgroupSize.width, (_fftDimension + kThreadgroupSize.height - 1) / kThreadgroupSize.height, 1);
    
    //NSLog(@"self.internalTexture w: %d, h:  %d", [self.internalTexture width], [self.internalTexture height]);
    //[commandEncoder dispatchThreadgroups:kThreadgroups threadsPerThreadgroup:kThreadgroupSize];
    
    
    /*
     *  Transpose the image after performing 1D FFT on the rows once
     */
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)[self.pipelineStages objectAtIndex:1];
    
    //commit multiple kernel functions to chain them
    //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:pipeline];
    
    //[commandEncoder setTexture:self.internalTexture atIndex:0];                     //bind the output texture to the kernel shader
    //[commandEncoder setTexture:self.internalTexture atIndex:1];                     //bind the output texture to the kernel shader
    //[self bindFFTTextureWithCommandEncoder:commandEncoder atIndex:0];                 //bind the packed FFT input texture ready to be processed
    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];
    
    
    
    //uniforms.threadGroupIndex = 0;
    
    if( !self.uniformBuffer )
    {
        
        NSLog(@"creating uniform buffer");
        
        struct FFTImageFilterUniforms uniforms;
        uniforms.signalLength = _fftDimension;
        uniforms.kernelSize = kSize;
        uniforms.fftStageIndex = pow(2, self.numFFTStages);
        
        self.uniformBuffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];
        
        /*
         *  Update constant uniform buffer memory
         */
        memcpy([self.uniformBuffer contents], &uniforms, sizeof(uniforms));
        
    }
    
    
    
    /*
     *  Attach the constant uniform buffer for passing to kernel function
     */
    [commandEncoder setBuffer:self.uniformBuffer offset:0 atIndex:1];
    
    
    
    //configure GPU threading for the kernel function
    kThreadgroupSize = MTLSizeMake(1, 1, 1);
    kThreadgroups = MTLSizeMake((_fftDimension + kThreadgroupSize.width - 1) / kThreadgroupSize.width, (_fftDimension + kThreadgroupSize.height - 1) / kThreadgroupSize.height, 1);
    
    //NSLog(@"self.internalTexture w: %d, h:  %d", [self.internalTexture width], [self.internalTexture height]);
    [commandEncoder dispatchThreadgroups:kThreadgroups threadsPerThreadgroup:kThreadgroupSize];
    
    
    
    
    
}


-(void)render1DFFTAndTranspose2:(id<MTLComputeCommandEncoder>)commandEncoder
{
    unsigned int kSize = self.kernelSize;
    unsigned int numCompletedStages = 0;
    unsigned int numRemainingStages = self.numFFTStages;
    
    NSError * error;

    /*
     *  Loop Over And Execute The Planned FFT Kernel Stages
     */
    for (int i=0; i<self.fftPipelineStages.count - 1; i++)
    {
        
        if( i == self.fftPipelineStages.count -1 )
            kSize = self.finalFFTStageKernelSize;
        
        /*
        id<MTLFunction> fftKernelFunction;
        
        if( kSize == 8 )
            fftKernelFunction = [self.context.library newFunctionWithName:@"fft_radix8_2"];
        else if ( kSize == 4 )
            fftKernelFunction = [self.context.library newFunctionWithName:@"fft_radix4_2"];
        else if ( kSize == 2 )
            fftKernelFunction = [self.context.library newFunctionWithName:@"fft_radix2_2"];

        
        id<MTLComputePipelineState> fftComplexPipelineStage = [self.context.device newComputePipelineStateWithFunction:fftKernelFunction error:&error];
        
        if (!fftComplexPipelineStage)
        {
            NSLog(@"Error occurred when building compute pipeline for function %@", fftKernelFunction);
            NSLog(@"Here is the error:  %@", [error localizedDescription]);
            return;
        }
         */
       // NSLog(@"Executing FFT Stage %d", i+1);

        id<MTLComputePipelineState> fftComplexPipelineStage = (id<MTLComputePipelineState>)[self.fftPipelineStages objectAtIndex:i];

        //commit multiple kernel functions to chain them
        //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        [commandEncoder setComputePipelineState:fftComplexPipelineStage];
         
        //[self bindComplexBuffersAtIndex:0 withCommandEncoder:commandEncoder];            //bind the packed fft input buffer ready to be processed
        
        [self bindComplexBuffersWithIndex:0 atIndex:0 withCommandEncoder:commandEncoder];
        
        
        /*
         *  Bind a constant uniform buffer defining paramers used for 1D twist kernel at each subsequent stage
         */
            
            //create an FFT buffer with parameters to pass to the fft kernel being executed
            struct FFTImageFilterUniforms uniforms;
            uniforms.signalLength = _fftDimension;
            uniforms.kernelSize = kSize;
            
            float ncsFloat = pow(2.f, (float)numCompletedStages);
            //NSLog(@"Ns Value:   %f", ncsFloat);
            uniforms.fftStageIndex = (uint)ncsFloat;
            
            //NSLog(@"uniforms.fftStage = %u", uniforms.fftStage);
            //uniforms.threadGroupIndex = 0;
            
            id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];
            
            /*
             *  Update constant uniform buffer memory
             */
            memcpy([buffer contents], &uniforms, sizeof(uniforms));
            
            
            /*
             *  Attach the constant uniform buffer for passing to kernel function
             */
            [commandEncoder setBuffer:buffer offset:0 atIndex:1];
            //configure GPU threading for the kernel function
            
            MTLSize kThreadgroupSize = MTLSizeMake(self.maxThreadWidth, 1, 1);
        
            //NSLog(@"Thread Execution Width:   %lu", (unsigned long)maxThreadsPerThreadGroup);
            
            NSUInteger nThreadCountW = (self.maxThreadWidth + kThreadgroupSize.width - 1 ) / kThreadgroupSize.width;
            NSUInteger nThreadCountH = (_fftDimension + kThreadgroupSize.height - 1) / kThreadgroupSize.height;
            MTLSize kThreadgroupsCount = MTLSizeMake(nThreadCountW, nThreadCountH, 1);
            //[commandEncoder setThreadgroupMemoryLength:sizeof(struct ThreadGroupBuffer) atIndex:0];
            
            [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
        
        
        [commandEncoder setComputePipelineState:fftComplexPipelineStage];
        
        
        [self bindComplexBuffersWithIndex:1 atIndex:0 withCommandEncoder:commandEncoder];
        
        
        /*
         *  Attach the constant uniform buffer for passing to kernel function
         */
        [commandEncoder setBuffer:buffer offset:0 atIndex:1];
        //configure GPU threading for the kernel function
        
        
        [commandEncoder dispatchThreadgroups:kThreadgroupsCount threadsPerThreadgroup:kThreadgroupSize];
        
        
            if( kSize == 8 )
            {
                numCompletedStages += 3;
                numRemainingStages -= 3;
            }
            else if( kSize == 4 )
            {
                numCompletedStages += 2;
                numRemainingStages -= 2;
            }
            else if ( kSize ==  2 )
            {
                numCompletedStages += 1;
                numRemainingStages -= 1;
            }
        
        
    
    }
    
    
    //configure GPU threading for the kernel function
    MTLSize kThreadgroupSize = MTLSizeMake(1, 1, 1);
    MTLSize kThreadgroups = MTLSizeMake((_fftDimension + kThreadgroupSize.width - 1) / kThreadgroupSize.width, (_fftDimension + kThreadgroupSize.height - 1) / kThreadgroupSize.height, 1);
    
    //NSLog(@"self.internalTexture w: %d, h:  %d", [self.internalTexture width], [self.internalTexture height]);
    //[commandEncoder dispatchThreadgroups:kThreadgroups threadsPerThreadgroup:kThreadgroupSize];
    
    /*
     *  Merge Even/Odd Complex Buffers back into a single Complex Buffer
     */
    [self interleaveComplexBufferWithCommandEncoder:commandEncoder];
    

    /*
     *  Perform the final FFT Stage after recombination, if necessary
     */
    
    
    
    /*
     *  Transpose the image/buffer after performing 1D FFT on the rows once
     */
    id<MTLFunction> transposeKernelFunction = [self.context.library newFunctionWithName:@"Transpose_Complex_Buffer"];
    
    id<MTLComputePipelineState> transposeComplexPipelineStage = [self.context.device newComputePipelineStateWithFunction:transposeKernelFunction error:&error];
    
    if (!transposeComplexPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", transposeKernelFunction);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;
    }
    
    //commit multiple kernel functions to chain them
    //id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:transposeComplexPipelineStage];
    
    //[commandEncoder setTexture:self.internalTexture atIndex:0];                     //bind the output texture to the kernel shader
    //[commandEncoder setTexture:self.internalTexture atIndex:1];                     //bind the output texture to the kernel shader
    //[self bindFFTTextureWithCommandEncoder:commandEncoder atIndex:0];               //bind the packed FFT input texture ready to be processed
    [self bindComplexBufferWithCommandEncoder:commandEncoder atIndex:0];
    

    struct FFTImageFilterUniforms uniforms;
    uniforms.signalLength = _fftDimension;
    uniforms.kernelSize = kSize;
    uniforms.fftStageIndex = pow(2, self.numFFTStages);
    
    id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceOptionCPUCacheModeDefault];

    memcpy([buffer contents], &uniforms, sizeof(uniforms));

    
    /*
     *  Attach the constant uniform buffer for passing to kernel function
     */
    [commandEncoder setBuffer:buffer offset:0 atIndex:1];
    
    
    
    //configure GPU threading for the kernel function
    kThreadgroupSize = MTLSizeMake(1, 1, 1);
    kThreadgroups = MTLSizeMake((_fftDimension + kThreadgroupSize.width - 1) / kThreadgroupSize.width, (_fftDimension + kThreadgroupSize.height - 1) / kThreadgroupSize.height, 1);
    
    //NSLog(@"self.internalTexture w: %d, h:  %d", [self.internalTexture width], [self.internalTexture height]);
    [commandEncoder dispatchThreadgroups:kThreadgroups threadsPerThreadgroup:kThreadgroupSize];
    
    
    
    
}



-(void)bindThreadGroupBufferWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
{
    struct ThreadGroupBuffer uniforms;
    uniforms.threadGroupIndex = 0;
    uniforms.threadGroupParam1 = 1;
    uniforms.threadGroupParam2 = 2;
    uniforms.threadGroupParam3 = 3;
    
    if (!self.threadGroupBuffer)
    {
        self.threadGroupBuffer = [self.context.device newBufferWithLength:sizeof(uniforms)
                                                              options:MTLResourceOptionCPUCacheModeDefault];
    }
    
    memcpy([self.threadGroupBuffer contents], &uniforms, sizeof(uniforms));
    
    
    [commandEncoder setBuffer:self.threadGroupBuffer offset:0 atIndex:1];

}

-(void)bindFFTUniformBufferWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder atIndex:(NSUInteger)bindIndex
{
    struct FFTImageFilterUniforms uniforms;
    uniforms.signalLength = _fftDimension;
    uniforms.kernelSize = 8;
    uniforms.fftStageIndex = 1;
    
    if (!self.uniformBuffer)
    {
        self.uniformBuffer = [self.context.device newBufferWithLength:sizeof(uniforms)
                                                              options:MTLResourceOptionCPUCacheModeDefault];
    }
    
    memcpy([self.uniformBuffer contents], &uniforms, sizeof(uniforms));
    
    
    [commandEncoder setBuffer:self.uniformBuffer offset:0 atIndex:bindIndex];
}


-(void)bindUniformBufferWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
{
    struct FFTImageFilterUniforms uniforms;
    uniforms.signalLength = _fftDimension;
    //uniforms.threadGroupIndex = 0;
    
    if (!self.uniformBuffer)
    {
        self.uniformBuffer = [self.context.device newBufferWithLength:sizeof(uniforms)
                                                              options:MTLResourceOptionCPUCacheModeDefault];
    }
    
    memcpy([self.uniformBuffer contents], &uniforms, sizeof(uniforms));
    
    
    [commandEncoder setBuffer:self.uniformBuffer offset:0 atIndex:0];
}

-(void)bindComplexBufferWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder atIndex:(NSUInteger)bindIndex
{

    uint bufferSize = _fftDimension*_fftDimension*2*sizeof(float);
    
    if (!self.complexBuffer)
    {
    
        //void * buffer;
        //uint alignment = 0x4000; // 16K aligned
        //uint size = bufferSize;// bufferSize == your buffer size
        
        //posix_memalign(&buffer, alignment, bufferSize);
    
        self.complexBuffer = [self.context.device newBufferWithLength:bufferSize
                                                              options:MTLResourceStorageModeShared];

        //self.complexBuffer = [self.context.device newBufferWithBytesNoCopy:buffer length:bufferSize options:0 deallocator:nil];
        //memset([self.complexBuffer contents], 0, bufferSize);
        
        //[self generateFFTPlan];


    }
    //else
    //    memset([self.complexBuffer contents], 0, bufferSize);
    
    
    
    [commandEncoder setBuffer:self.complexBuffer offset:0 atIndex:bindIndex];
}

-(void)bindComplexBuffer2WithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder atIndex:(NSUInteger)bindIndex
{
    
    uint bufferSize = _fftDimension*_fftDimension*2*sizeof(float);
    
    if (!self.complexBuffer2)
    {
        
        //void * buffer;
        //uint alignment = 0x4000; // 16K aligned
        //uint size = bufferSize;// bufferSize == your buffer size
        
        //posix_memalign(&buffer, alignment, bufferSize);
        
        self.complexBuffer2 = [self.context.device newBufferWithLength:bufferSize
                                                              options:MTLResourceStorageModeShared];
        
        //self.complexBuffer = [self.context.device newBufferWithBytesNoCopy:buffer length:bufferSize options:0 deallocator:nil];
        //memset([self.complexBuffer contents], 0, bufferSize);
        
        //[self generateFFTPlan];
        
        
    }
    //else
    //    memset([self.complexBuffer contents], 0, bufferSize);
    
    
    
    [commandEncoder setBuffer:self.complexBuffer2 offset:0 atIndex:bindIndex];
}

//when the texture is requested, apply the filter on the GPU and synchronously return
- (id<MTLBuffer>)buffer
{
    if (self.isDirty)
    {
       // NSLog(@"Apply Complex Buffer Filter");
        //[self applyFilter];
        [self updateOutputTextureSize];
        [self renderPipelineStages];
    }
    
    return self.complexBuffer;
}

-(void)bindComplexBuffer:(uint)bufferIndex atIndex:(uint)bindIndex withCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
{

    if( bufferIndex >= maxComplexBuffers )
    {
        NSLog(@"Error:  Complex Buffer Array Out of Bounds!");
        assert(0);
    }
    
    uint bufferSize = _fftDimension*_fftDimension*2*sizeof(float);

    if( !_complexBuffers[bufferIndex] )
    {
    
        
            _complexBuffers[bufferIndex] = [self.context.device newBufferWithLength:bufferSize
                                                     options:MTLResourceStorageModeShared];
        
            //we will pad with zeros within the shader for speed
            //memset([_complexBuffers[bufferIndex] contents], 0, bufferSize);
            
    }
    //else
    //    memset([_complexBuffers[bufferIndex] contents], 0, bufferSize);

    [commandEncoder setBuffer:_complexBuffers[bufferIndex] offset:0 atIndex:bindIndex];

        
    
    
    
}


-(void)bindComplexBuffersWithIndex:(uint)bufferIndex atIndex:(uint)bindIndex withCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
{
    uint numBuffers = _fftDimension/self.maxThreadWidth;
    uint bufferSize = self.maxThreadWidth*_fftDimension*2*sizeof(float);
    
    //NSLog(@"Creating %d FFT Complex Strip Buffers", numBuffers);
    
    if( bufferIndex < numBuffers )
    {
        if( !_complexBuffers[bufferIndex] )
        {
            _complexBuffers[bufferIndex] = [self.context.device newBufferWithLength:bufferSize
                                                                 options:MTLResourceStorageModeShared];
            
            memset([_complexBuffers[bufferIndex] contents], 0, bufferSize);
            
        }
        //else
        //    memset(complexBuffers[i] contents], 1, bufferSize);
        
        [commandEncoder setBuffer:_complexBuffers[bufferIndex] offset:0 atIndex:bindIndex];


    }
    else
    {
            NSLog(@"Error:  Complex Buffer Array Out of Bounds!");
            assert(0);
            
    }
    

    
    
}


-(void)bindFFTTextureWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder atIndex:(NSUInteger)bindIndex
{
    //create or recreate the FFT Metal 2D Texture as necessary
    if (!self.internalTexture || [self.internalTexture width] != _fftDimension || [self.internalTexture height] != _fftDimension)
    {
        [self generateComplexFourierTexture];
        //[self generateFFTPlan];

    }
    
    [commandEncoder setTexture:self.fftTexture atIndex:bindIndex];


}


- (void)configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
{
    
    //update and bind the uniform buffer to be passed to the pipeline stage kernel function
    [self bindUniformBufferWithCommandEncoder:commandEncoder];
    
    
    //if the fft image size has updated as a result of the base class input image size being updated
    //recreate teh fft metal texture
    if (!self.fftTexture || [self.fftTexture width] != _fftDimension || [self.fftTexture height] != _fftDimension)
    {
       [self generateComplexFourierTexture];
       [self generateFFTPlan];
    }
    
    //Optionally, the fftTexture may be packed with values from the input texture here on the CPU rather than on the GPU
    /*
    void * inputPixelBuffer;
    NSUInteger bytesPerRow;

    MTLRegion region = MTLRegionMake2D(0, 0, _imageSize.width, _imageSize.height);
    [self.provider.texture getBytes:inputPixelBuffer bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
    
    [self.fftTexture replaceRegion:region mipmapLevel:0 withBytes:inputPixelBuffer bytesPerRow:sizeof(float) * (int)_imageSize.height];
    
    //should the pixel buffer from the metal input texture be freed?
    free(inputPixelBuffer);
    */
    [commandEncoder setTexture:self.fftTexture atIndex:2];
}

- (void)generateComplexFourierTexture
{
    
    NSLog(@"Generate Complex Fourier Texture");
    
    //Use an RG32 64 bit Metal Texture:  The Red Channel will store the Real FFT Output and the Green Channel will store the Complex FFT output
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                                 width:_fftDimension
                                                                                                height:_fftDimension
                                                                                             mipmapped:NO];
    
    self.internalTexture = [self.context.device newTextureWithDescriptor:textureDescriptor];
    
    
    //MTLRegion region = MTLRegionMake2D(0, 0, _fftDimension, _fftDimension);
    //[self.fftTexture replaceRegion:region mipmapLevel:0 withBytes:weights bytesPerRow:sizeof(float) * size];
    
    //free(weights);
}

-(void)generateFFTPlan
{
    NSLog(@"Generate FFT Plan");
    //decompose each row FFT signal into 8 Point FFT Kernels
    //the initial 8 point FFT stage counts for 3 stages when completed
    uint numFFTStages = log2(_fftDimension);
    self.numFFTStages = numFFTStages;       //store this so we can use it when rendering the kernel functions

    self.fftPipelineStages = [[NSMutableArray alloc] init];
    self.fftPipelineUniforms = [[NSMutableArray alloc] init];

    self.kernelSize = 8;

    NSLog(@"FFT Kernel Size:    %d", self.kernelSize);
    NSLog(@"Num FFT Stages:     %d", numFFTStages);

    uint kSize = 8;
    unsigned int numCompletedStages = 0;

    if( self.kernelSize == 8 )
    {
        kSize = 8;
        
        //plan as many as 8 point kernel stages as possible
        int stageCount = 0;
        while(numFFTStages >= 3)
        {
            NSLog(@"Stage %d", stageCount+1);
            stageCount++;
            NSError * error = nil;

            id<MTLFunction> kernelFunction = [self.context.library newFunctionWithName:@"fft_radix8_OOP"];
            id<MTLComputePipelineState> pipelineStage = [self.context.device newComputePipelineStateWithFunction:kernelFunction error:&error];
            
            if (!pipelineStage)
            {
                NSLog(@"Error occurred when building compute pipeline for function %@", pipelineStage);
                NSLog(@"Here is the error:  %@", [error localizedDescription]);
                return;
            }
            else
            {
                [self.fftPipelineStages addObject:pipelineStage];
            
                //create an FFT buffer with parameters to pass to the fft kernel being executed
                struct FFTImageFilterUniforms uniforms;
                uniforms.signalLength = _fftDimension;
                uniforms.kernelSize = kSize;
                
                float ncsFloat = pow(2.f, (float)numCompletedStages);
                //NSLog(@"Ns Value:   %f", ncsFloat);
                uniforms.fftStageIndex = (uint)ncsFloat;
                
                //NSLog(@"uniforms.fftStage = %u", uniforms.fftStage);
                //uniforms.threadGroupIndex = 0;
                
                id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceStorageModeShared];
                
                /*
                 *  Update constant uniform buffer memory
                 */
                memcpy([buffer contents], &uniforms, sizeof(uniforms));
                
                
                [self.fftPipelineUniforms addObject:buffer];
            
            }
            
            numFFTStages -= 3;
            numCompletedStages += 3;
        }
        
        self.finalFFTStageKernelSize = 8;
        
        
        //plan the final FFT stage as a 4 point FFT kernel if we are left with 2 stages remaining after planning 8 point kernels
        if( numFFTStages == 2 )
        {
            
            NSLog(@"Final Stage is 4 point kernel");
            NSError * error = nil;
            
            id<MTLFunction> kernelFunction = [self.context.library newFunctionWithName:@"FFT_4_KERNEL"];
            id<MTLComputePipelineState> pipelineStage = [self.context.device newComputePipelineStateWithFunction:kernelFunction error:&error];
            
            if (!pipelineStage)
            {
                NSLog(@"Error occurred when building compute pipeline for function %@", pipelineStage);
                NSLog(@"Here is the error:  %@", [error localizedDescription]);
                return;
            }
            else
            {
                [self.fftPipelineStages addObject:pipelineStage];
            
                //create an FFT buffer with parameters to pass to the fft kernel being executed
                struct FFTImageFilterUniforms uniforms;
                uniforms.signalLength = _fftDimension;
                uniforms.kernelSize = kSize;
                
                float ncsFloat = pow(2.f, (float)numCompletedStages);
                //NSLog(@"Ns Value:   %f", ncsFloat);
                uniforms.fftStageIndex = (uint)ncsFloat;
                
                //NSLog(@"uniforms.fftStage = %u", uniforms.fftStage);
                //uniforms.threadGroupIndex = 0;
                
                id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceStorageModeShared];
                
                /*
                 *  Update constant uniform buffer memory
                 */
                memcpy([buffer contents], &uniforms, sizeof(uniforms));
                
                
                [self.fftPipelineUniforms addObject:buffer];
                
            }
            
            numFFTStages -= 2;
            numCompletedStages += 2;
            self.finalFFTStageKernelSize = 4;

        }
        
        //plan the final FFT stage as a 2 point FFT kernel if we are left with only 1 stage remaining after planning 8 point kernels
        else if( numFFTStages == 1 )
        {
            kSize =2;
            NSLog(@"Final Stage is 2 point kernel");

            NSError * error = nil;
            
            id<MTLFunction> kernelFunction = [self.context.library newFunctionWithName:@"fft_radix2_OOP"];
            id<MTLComputePipelineState> pipelineStage = [self.context.device newComputePipelineStateWithFunction:kernelFunction error:&error];
            
            if (!pipelineStage)
            {
                NSLog(@"Error occurred when building compute pipeline for function %@", pipelineStage);
                NSLog(@"Here is the error:  %@", [error localizedDescription]);
                return;
            }
            else
            {
                [self.fftPipelineStages addObject:pipelineStage];
            
            
                //create an FFT buffer with parameters to pass to the fft kernel being executed
                struct FFTImageFilterUniforms uniforms;
                uniforms.signalLength = _fftDimension;
                uniforms.kernelSize = kSize;
                
                float ncsFloat = pow(2.f, (float)numCompletedStages);
                //NSLog(@"Ns Value:   %f", ncsFloat);
                uniforms.fftStageIndex = (uint)ncsFloat;
                
                //NSLog(@"uniforms.fftStage = %u", uniforms.fftStage);
                //uniforms.threadGroupIndex = 0;
                
                id<MTLBuffer> buffer = [self.context.device newBufferWithLength:sizeof(uniforms) options:MTLResourceStorageModeShared];
                
                /*
                 *  Update constant uniform buffer memory
                 */
                memcpy([buffer contents], &uniforms, sizeof(uniforms));
                
                
                [self.fftPipelineUniforms addObject:buffer];
                
            }
            
            numFFTStages -= 1;
            numCompletedStages += 1;
            self.finalFFTStageKernelSize = 2;

        }
    
    }
    else if( self.kernelSize == 4 )
    {
    
    }
    else if ( self.kernelSize == 2 )
    {
    
        while ( numFFTStages > 0 )
        {
            NSError * error = nil;
            
            id<MTLFunction> kernelFunction = [self.context.library newFunctionWithName:@"fft_radix2"];
            id<MTLComputePipelineState> pipelineStage = [self.context.device newComputePipelineStateWithFunction:kernelFunction error:&error];
            
            if (!pipelineStage)
            {
                NSLog(@"Error occurred when building compute pipeline for function %@", pipelineStage);
                NSLog(@"Here is the error:  %@", [error localizedDescription]);
                return;
            }
            else
                [self.fftPipelineStages addObject:pipelineStage];
            
            numFFTStages -= 1;
            
            
        }
        
        self.finalFFTStageKernelSize = 2;
        

    }
    else
    {
        NSLog(@">> ERROR: Using an Undefined FFT Kernel Size.");
        assert(0);
    }

    NSLog(@"Size of FFT Pipeline Stages Array:   %lu", (unsigned long)self.fftPipelineStages.count);

    if( numFFTStages != 0 )
    {
        NSLog(@">> ERROR: Num FFT Stages did not sum to 0.");
        assert(0);
    }
    

/*
    NSError * error = nil;
    id<MTLFunction> initialStageKernelFunction = [self.context.library newFunctionWithName:@"FFT_8_KERNEL"];
    id<MTLComputePipelineState> initialFFTPipelineStage = [self.context.device newComputePipelineStateWithFunction:initialStageKernelFunction error:&error];
    
    if (!initialFFTPipelineStage)
    {
        NSLog(@"Error occurred when building compute pipeline for function %@", initialFFTPipelineStage);
        NSLog(@"Here is the error:  %@", [error localizedDescription]);
        return;
    }
    else
        [self.fftStages addObject:initialFFTPipelineStage];


    for( int i = 0; i < numFFTStages-1; i++)
    {
        id<MTLFunction> kernelFunction = [self.context.library newFunctionWithName:@"FFT_8_KERNEL"];
        id<MTLComputePipelineState> pipelineStage = [self.context.device newComputePipelineStateWithFunction:kernelFunction error:&error];
        
        if (!pipelineStage)
        {
            NSLog(@"Error occurred when building compute pipeline for function %@", pipelineStage);
            NSLog(@"Here is the error:  %@", [error localizedDescription]);
            return;
        }
        else
            [self.fftStages addObject:pipelineStage];

    }
*/
    //[self.pipelineStages addObjectsFromArray:_fftStages];

}




@end
