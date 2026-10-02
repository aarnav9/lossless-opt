// Read-only Metal capabilities. No command queue, buffers, or kernel dispatch.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(void) {
  @autoreleasepool {
    NSMutableArray *devices = [NSMutableArray array];
    for (id<MTLDevice> d in MTLCopyAllDevices()) {
      MTLSize t = d.maxThreadsPerThreadgroup;
      NSMutableArray *families = [NSMutableArray array];
      // Stable MTLGPUFamily enum values; unsupported values return NO.
      NSDictionary *known = @{@1001:@"Apple1", @1002:@"Apple2", @1003:@"Apple3",
        @1004:@"Apple4", @1005:@"Apple5", @1006:@"Apple6", @1007:@"Apple7",
        @1008:@"Apple8", @1009:@"Apple9", @1010:@"Apple10",
        @2002:@"Mac2", @3001:@"Common1", @3002:@"Common2", @3003:@"Common3",
        @5001:@"Metal3", @5002:@"Metal4"};
      for (NSNumber *family in [[known allKeys] sortedArrayUsingSelector:@selector(compare:)])
        if ([d supportsFamily:(MTLGPUFamily)family.integerValue]) [families addObject:known[family]];
      NSMutableDictionary *info = [@{
        @"source": @"MTLDevice public API", @"device_name": d.name,
        @"unified_memory": @(d.hasUnifiedMemory), @"low_power": @(d.isLowPower),
        @"max_buffer_length_bytes": @(d.maxBufferLength),
        @"recommended_working_set_bytes": @(d.recommendedMaxWorkingSetSize),
        @"max_threadgroup_memory_bytes": @(d.maxThreadgroupMemoryLength),
        @"max_threadgroup_dimensions": @[@(t.width), @(t.height), @(t.depth)],
        @"supported_families": families,
        @"physical_memory_channel_mapping": [NSNull null],
        @"gpu_l2_cache_bytes": [NSNull null]
      } mutableCopy];
      NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
        @"kernel void probe(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = x[i] + 1.0f; }";
      NSError *error = nil;
      id<MTLLibrary> library = [d newLibraryWithSource:source options:nil error:&error];
      id<MTLComputePipelineState> pipeline = nil;
      if (library) pipeline = [d newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
      if (pipeline) {
        info[@"probe_pipeline"] = @{
          @"status": @"ok", @"source": source,
          @"thread_execution_width": @(pipeline.threadExecutionWidth),
          @"max_total_threads_per_threadgroup": @(pipeline.maxTotalThreadsPerThreadgroup),
          @"static_threadgroup_memory_bytes": @(pipeline.staticThreadgroupMemoryLength),
          @"scope": @"Trivial compiled probe only; query every actual candidate pipeline separately. No dispatch."};
      } else info[@"probe_pipeline"] = @{@"status":@"unavailable", @"reason":@"pipeline_compile_failed"};
      [devices addObject:info];
    }
    NSDictionary *result = @{@"status": devices.count ? @"ok" : @"unavailable",
                              @"devices": devices,
                              @"scope": @"Recommended working set is advisory; device dimensions are not candidate-specific pipeline limits."};
    NSData *json = [NSJSONSerialization dataWithJSONObject:result options:0 error:nil];
    puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
  }
  return 0;
}
