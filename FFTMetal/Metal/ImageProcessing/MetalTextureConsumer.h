//
//  MetalTextureConsumer.h
//  FFTMetal
//
//  Created by MACMaster on 12/11/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#ifndef MetalTextureConsumer_h
#define MetalTextureConsumer_h

@protocol MetalTextureProvider;

@protocol MetalTextureConsumer <NSObject>

@property (nonatomic, strong) id<MetalTextureProvider> provider;

@end

#endif /* MetalTextureConsumer_h */
