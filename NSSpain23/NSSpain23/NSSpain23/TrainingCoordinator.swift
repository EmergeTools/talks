//
//  TrainingCoordinator.swift
//  NSSpain23
//
//  Created by Noah Martin on 9/4/23.
//

import Foundation
import SwiftUI
import MetalPerformanceShadersGraph

class TrainingState: ObservableObject {
  @Published var loss = [Float]()
  @Published var images = [UIImage]()
}

class TrainingCoordinator {
  let resnet50 = Resnet50()
  let dataSet = CIFARDataSet(device: gDevice)
  let trainingState = TrainingState()

  func startTraining() {
    dataSet.prepare()
    runBatch()
  }

  func runBatch() {
    let input = MPSNDArray(device: gDevice, descriptor: .init(dataType: .float32, shape: [64, 32, 32, 3]))
    let labels = MPSNDArray(device: gDevice, descriptor: .init(dataType: .float32, shape: [64, 10]))
    dataSet.getInputData(tensor: input)
    dataSet.getLabels(tensor: labels)
    var inputFloats = [Float](repeating: 0, count: 64 * 32 * 32 * 3)
    input.readBytes(&inputFloats, strideBytes: nil)
    var images = [UIImage]()
    for i in 0..<4 {
      let imageData = inputFloats[i*32*32*3..<i*32*32*3 + 32*32*3].map { UInt8($0 * 255) }
      let image = imageFromARGB32Bitmap(pixels: imageData, width: 32, height: 32)!
      images.append(image)
    }
    trainingState.images = images
    resnet50.train(
      with: MPSGraphTensorData(input),
      batchLabels: MPSGraphTensorData(labels)) { [weak self] loss in
        DispatchQueue.main.async {
          print("done \(loss)")
          self?.trainingState.loss.append(loss)
          self?.dataSet.advance()
          self?.runBatch()
        }
      }
  }
}

struct PixelData {
  var r: UInt8
  var g: UInt8
  var b: UInt8
}

func imageFromARGB32Bitmap(pixels: [UInt8], width: Int, height: Int) -> UIImage? {
    guard width > 0 && height > 0 else { return nil }
    guard pixels.count == width * height * 3 else { return nil }

    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
    let bitsPerComponent = 8
    let bitsPerPixel = 24

    var data = pixels // Copy to mutable []
    guard let providerRef = CGDataProvider(data: NSData(bytes: &data,
                            length: data.count * MemoryLayout<UInt8>.size)
        )
        else { return nil }

    guard let cgim = CGImage(
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bitsPerPixel: bitsPerPixel,
        bytesPerRow: width * MemoryLayout<UInt8>.size * 3,
        space: rgbColorSpace,
        bitmapInfo: bitmapInfo,
        provider: providerRef,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
        )
        else { return nil }

    return UIImage(cgImage: cgim)
}
