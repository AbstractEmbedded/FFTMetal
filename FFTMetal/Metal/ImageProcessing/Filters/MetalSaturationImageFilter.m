//
//  MetalSaturationImageFilter.m
//  FFTMetal
//
//  Created by MACMaster on 12/13/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import "MetalSaturationImageFilter.h"

struct AdjustSaturationUniforms
{
    float saturationFactor;
};

@implementation MetalSaturationImageFilter

@synthesize saturationFactor=_saturationFactor;

+ (instancetype)filterWithSaturationFactor:(float)saturation context:(MetalGPGPUContext *)context
{
    return [[self alloc] initWithSaturationFactor:saturation context:context];
}

- (instancetype)initWithSaturationFactor:(float)saturation context:(MetalGPGPUContext *)context
{
    if ((self = [super initWithFunctionName:@"adjust_saturation" context:context]))
    {
        _saturationFactor = saturation;
    }
    return self;
}

- (void)setSaturationFactor:(float)saturationFactor
{
    self.dirty = YES;
    _saturationFactor = saturationFactor;
}

- (void)configureArgumentTableWithCommandEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
{
    struct AdjustSaturationUniforms uniforms;
    uniforms.saturationFactor = self.saturationFactor;
    
    if (!self.uniformBuffer)
    {
        self.uniformBuffer = [self.context.device newBufferWithLength:sizeof(uniforms)
                                                              options:MTLResourceOptionCPUCacheModeDefault];
    }
    
    memcpy([self.uniformBuffer contents], &uniforms, sizeof(uniforms));
    
    [commandEncoder setBuffer:self.uniformBuffer offset:0 atIndex:0];
}

@end
