//
//  MetalTextureProvider.h
//  An Obj-c protocol for an object to provide a texture
//
//  Created by MACMaster on 12/11/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#ifndef MetalTextureProvider_h
#define MetalTextureProvider_h

#import <Foundation/Foundation.h>

//impolement this protocol on a class to add a metal texture property you can populate for processing

@protocol MTLTexture;

@protocol MetalTextureProvider <NSObject>

@property (nonatomic, readonly) id<MTLTexture> texture;

@end


#endif