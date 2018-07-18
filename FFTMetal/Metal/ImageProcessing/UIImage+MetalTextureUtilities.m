//
//  UIImage+MBETextureUtilities.m
//  ImageProcessing
//
//  Created by Warren Moore on 10/8/14.
//  Copyright (c) 2014 Metal By Example. All rights reserved.
//

#import "UIImage+MetalTextureUtilities.h"
#import <Metal/Metal.h>

static void MetalReleaseDataCallback(void *info, const void *data, size_t size)
{
    free((void *)data);
}

@implementation UIImage (MBETextureUtilities)

+ (UIImage *)imageWithMTLTexture:(id<MTLTexture>)texture
{
    //NSLog(@"pixel format %d", [texture pixelFormat]);
    //NSAssert([texture pixelFormat] == MTLPixelFormatR8Unorm, @"Pixel format of texture must be MTLPixelFormatBGRA8Unorm to create UIImage");
    
    
    CGSize imageSize = CGSizeMake([texture width], [texture height]);

    
    if( imageSize.width <= 0 || imageSize.height <= 0 )
    {
        NSLog(@"Invalid image size");
    
        return nil;
    }
    size_t imageByteCount = imageSize.width * imageSize.height * 4;
    void *imageBytes = malloc(imageByteCount);
    NSUInteger bytesPerRow = imageSize.width * 4;
    MTLRegion region = MTLRegionMake2D(0, 0, imageSize.width, imageSize.height);
    [texture getBytes:imageBytes bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, imageBytes, imageByteCount, MetalReleaseDataCallback);
    int bitsPerComponent = 8;
    int bitsPerPixel = 32;
    CGColorSpaceRef colorSpaceRef = CGColorSpaceCreateDeviceRGB();
    
    //if converting BGRA pixels to ios image format, need to use little endian format and premultiply the alpha channel first
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little;
    //CGBitmapInfo bitmapInfo = kCGImageAlphaNone;


    CGColorRenderingIntent renderingIntent = kCGRenderingIntentDefault;
    CGImageRef imageRef = CGImageCreate(imageSize.width,
                                        imageSize.height,
                                        bitsPerComponent,
                                        bitsPerPixel,
                                        bytesPerRow,
                                        colorSpaceRef,
                                        bitmapInfo,
                                        provider,
                                        NULL,
                                        false,
                                        renderingIntent);
    
    UIImage *image = [UIImage imageWithCGImage:imageRef scale:0.0 orientation:UIImageOrientationDownMirrored];
    
    CFRelease(provider);
    CFRelease(colorSpaceRef);
    CFRelease(imageRef);
    
    return image;
}

@end
