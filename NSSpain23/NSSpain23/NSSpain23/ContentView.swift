//
//  ContentView.swift
//  NSSpain23
//
//  Created by Noah Martin on 8/30/23.
//

import SwiftUI
import MetalPerformanceShadersGraph
import Charts

let trainingCoordinator = TrainingCoordinator()

struct ContentView: View {

  @ObservedObject var trainingState = trainingCoordinator.trainingState

    var body: some View {
        VStack {
          if trainingState.images.count == 4 {
            HStack {
              Image(uiImage: trainingState.images[0]).resizable().frame(width: 64, height: 64)
              Image(uiImage: trainingState.images[1]).resizable().frame(width: 64, height: 64)
              Image(uiImage: trainingState.images[2]).resizable().frame(width: 64, height: 64)
              Image(uiImage: trainingState.images[3]).resizable().frame(width: 64, height: 64)
            }
          }
          ProgressChart(loss: trainingState.loss)
          Button("Train") {
            trainingCoordinator.startTraining()
          }
        }
        .padding()
    }
}

struct ProgressChart: View {

  let loss: [Float]

  var body: some View {
    Chart {
      ForEach(Array(loss.enumerated()), id: \.element) { index, trainingUpdate in
        LineMark(
          x: .value("Iteration", index),
          y: .value("Loss", trainingUpdate))
        .foregroundStyle(by: .value("Value", "Loss"))
      }
    }
    .chartYAxis {
        AxisMarks(position: .leading)
    }
    .frame(height: 300)
  }
}
