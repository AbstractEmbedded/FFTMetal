//
//  ScannerViewController.h
//  MC HW
//
//  Created by MACMaster on 9/13/15.
//
//

#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import "MetalTextureProvider.h"

@protocol LiveCameraViewControllerDelegate <NSObject>
@optional
- (void) dismissedWithScannedImage:(UIImage*)scannedImage;
@end

@interface LiveCameraViewController : UIViewController <AVCaptureMetadataOutputObjectsDelegate, AVCaptureVideoDataOutputSampleBufferDelegate, MetalTextureProvider>

@property (nonatomic, weak) id <LiveCameraViewControllerDelegate> delegate;

@end
