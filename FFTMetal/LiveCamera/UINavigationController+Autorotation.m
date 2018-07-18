//
//  UINavigationController+Autorotation.m
//  FFTMetal
//
//  Created by MACMaster on 12/24/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import "UINavigationController+Autorotation.h"

@implementation UINavigationController (Autorotation)

- (BOOL) shouldAutorotate
{
    return [[self topViewController] shouldAutorotate];
}

- (NSUInteger) supportedInterfaceOrientations
{
    return [[self topViewController] supportedInterfaceOrientations];
}

@end
