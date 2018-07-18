//
//  MetalImageFilter.h
//  FFTMetal
//
//  Created by MACMaster on 12/12/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MetalTextureProvider.h"
#import "MetalTextureConsumer.h"
#import "MetalGPGPUContext.h"

struct ThreadGroupBuffer
{
    unsigned int threadGroupIndex;
    unsigned int threadGroupParam1;
    unsigned int threadGroupParam2;
    unsigned int threadGroupParam3;
    
};

@protocol MTLTexture, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState;

@interface MetalImageFilter : NSObject <MetalTextureProvider, MetalTextureConsumer>

@property (nonatomic, strong) MetalGPGPUContext *context;
@property (nonatomic, strong) id<MTLBuffer> uniformBuffer;
@property (nonatomic, strong) id<MTLBuffer> threadGroupBuffer;
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property (nonatomic, retain) NSMutableArray * pipelineStages;
@property (nonatomic, strong) id<MTLTexture> internalTexture;
@property (nonatomic, assign, getter=isDirty) BOOL dirty;

- (instancetype)initWithFunctionName:(NSString *)functionName context:(MetalGPGPUContext *)context;

- (void)configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder;
- (void)updateOutputTextureSize;
- (void)setThreadgroupSize;
- (void)renderPipelineStages;
- (void)applyFilter;

@end
