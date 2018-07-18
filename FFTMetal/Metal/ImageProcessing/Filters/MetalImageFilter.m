//
//  MetalImageFilter.m
//  FFTMetal
//
//  Created by MACMaster on 12/12/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import "MetalImageFilter.h"
#import "MetalGPGPUContext.h"
#import <Metal/Metal.h>

@interface MetalImageFilter ()
{
    MTLSize _threadgroupSize;
    MTLSize _threadgroups;
}

@property (nonatomic, strong) id<MTLFunction> kernelFunction;
//@property (nonatomic, strong) id<MTLTexture> texture;
@end

@implementation MetalImageFilter

@synthesize dirty=_dirty;
@synthesize provider=_provider;

- (instancetype)initWithFunctionName:(NSString *)functionName context:(MetalGPGPUContext *)context;
{
    if ((self = [super init]))
    {
        NSError *error = nil;
        _context = context;
        _kernelFunction = [_context.library newFunctionWithName:functionName];
        _pipeline = [_context.device newComputePipelineStateWithFunction:_kernelFunction error:&error];
        _pipelineStages = [[NSMutableArray alloc] initWithObjects:_pipeline, nil];
        
        _threadgroupSize = MTLSizeMake(8, 8, 1);

        
        if (!_pipeline)
        {
            NSLog(@"Error occurred when building compute pipeline for function %@", functionName);
            return nil;
        }
        _dirty = YES;
    }
    
    return self;
}

- (instancetype)initWithFunctionName:(NSString *)functionName outTextureSize:(CGSize)textureSize context:(MetalGPGPUContext *)context;
{
    if ((self = [super init]))
    {
        NSError *error = nil;
        _context = context;
        _kernelFunction = [_context.library newFunctionWithName:functionName];
        _pipeline = [_context.device newComputePipelineStateWithFunction:_kernelFunction error:&error];
        _pipelineStages = [[NSMutableArray alloc] initWithObjects:_pipeline, nil];

        if (!_pipeline)
        {
            NSLog(@"Error occurred when building compute pipeline for function %@", functionName);
            return nil;
        }
        _dirty = YES;
    }
    
    return self;
}


#pragma mark -- virtual overloaded functions for extension classes
- (void)configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
{

}

- (void)updateOutputTextureSize
{

    id<MTLTexture> inputTexture = self.provider.texture;

    if (!self.internalTexture ||
        [self.internalTexture width] != [inputTexture width] ||
        [self.internalTexture height] != [inputTexture height])
    {
        MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:[inputTexture pixelFormat]
                                                                                                     width:[inputTexture width]
                                                                                                    height:[inputTexture height]
                                                                                                 mipmapped:NO];
        self.internalTexture = [self.context.device newTextureWithDescriptor:textureDescriptor];
        
        //call the virtual overloaded functions from child class implementations
        //[self updateOutputTextureSize];
        [self setThreadgroupSize];
        
    }
}

-(void)setThreadgroupSize
{
    _threadgroupSize = MTLSizeMake(8, 8, 1);
    _threadgroups = MTLSizeMake([self.internalTexture width] / _threadgroupSize.width, [self.internalTexture height] / _threadgroupSize.height, 1);

}

-(void)renderPipelineStages
{
    
    //populate the metal command buffer
    id<MTLCommandBuffer> commandBuffer = [self.context.commandQueue commandBuffer];
    
    //All filters set a single pipeline stage containing a single kernel function on initialization,
    //but child classes can modify the order and insert into pipeline stages to chain kernel functions if necessary
    for (int i=0; i<_pipelineStages.count; i++)
    {
        id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)[_pipelineStages objectAtIndex:i];
        
        //commit multiple kernel functions to chain them
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        [commandEncoder setComputePipelineState:pipeline];
        [commandEncoder setTexture:self.provider.texture atIndex:0];                     //bind the input texture to the kernel shader
        [commandEncoder setTexture:self.internalTexture atIndex:1];             //bind the output texture with the same pixel dimensions to the kernel shader
        
        //do any custom configuration for the filter by calling the virtual overloaded functions from child class implementations
        [self configureArgumentTableWithCommandEncoder:commandEncoder];
        [commandEncoder dispatchThreadgroups:_threadgroups threadsPerThreadgroup:_threadgroupSize];
        [commandEncoder endEncoding];
        
    }
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

}

#pragma mark -- render the filter
- (void)applyFilter
{
    [self updateOutputTextureSize];
    [self renderPipelineStages];
}

//when the texture is requested, apply the filter on the GPU and synchronously return
- (id<MTLTexture>)texture
{
    if (self.isDirty)
    {
        [self applyFilter];
    }
    
    return self.internalTexture;
}

@end
