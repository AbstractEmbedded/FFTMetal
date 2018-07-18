//
//  MetalFFTImageFilter.h
//  FFTMetal
//
//  Created by MACMaster on 12/17/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MetalImageFilter.h"

@interface MetalFFTImageFilter : MetalImageFilter

//@property (nonatomic, assign) float saturationFactor;
+ (instancetype)filterWithContext:(MetalGPGPUContext *)context;

@property (nonatomic, strong) id<MTLTexture> fftTexture;
@property (nonatomic, strong) id<MTLBuffer> complexBuffer;
- (id<MTLBuffer>)buffer;

@end
