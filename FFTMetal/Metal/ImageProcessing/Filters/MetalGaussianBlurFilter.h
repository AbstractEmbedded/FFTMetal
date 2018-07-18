//
//  MetalGaussianBlurFilter.h
//  FFTMetal
//
//  Created by MACMaster on 12/13/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import "MetalImageFilter.h"

@interface MetalGaussianBlurFilter : MetalImageFilter

@property (nonatomic, assign) float radius;
@property (nonatomic, assign) float sigma;

+ (instancetype)filterWithRadius:(float)radius context:(MetalGPGPUContext *)context;

@end