import SwiftUI
import ScreenCaptureKit
import AVFoundation
import Foundation
import ScreenCaptureKit
import CoreImage
import CoreGraphics
import ImageIO

struct ContentView: View {
    @State private var status: String = "Ready"
    @State var audio = AudioRecorder()

    var body: some View {
        VStack(spacing: 20) {
            Text(status)
            Button("Start Capture") {
                print("Starting capture")
                startAud()
                startCap()
            }
            Button("Stop Capture") {
                stopAud()
                stopCap()
                print("Stopping capture")
            }
        }
        .frame(width: 300, height: 200)
        .padding()
    }
}
