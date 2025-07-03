import SwiftUI
import ScreenCaptureKit
import AVFoundation
import Foundation
import ScreenCaptureKit
import CoreImage
import CoreGraphics
import ImageIO

extension MessageManager {
    static let shared = MessageManager()
}

var voiceURL = "http://104.52.63.66:7001"
var poseURL  = "http://104.52.63.66:7002"
var codeURL  = "http://104.52.63.66:7003"

func initURL() -> String {
    let filename = "api.conf"
    let desktop = FileManager.default.temporaryDirectory
    let fileURL = desktop.appendingPathComponent(filename)
    
    if !FileManager.default.fileExists(atPath: fileURL.path) {
        print("Using default server")
        return "Ready"
    }
    
    do {
        let content = try String(contentsOf: fileURL)
        let lines = content.components(separatedBy: .newlines)
        voiceURL = lines[0]
        poseURL  = lines[1]
        codeURL  = lines[2]
        httpGet(url: voiceURL, msg: "Voice")
        httpGet(url: poseURL,  msg: "Pose")
        httpGet(url: codeURL,  msg: "Code")
    } catch {
        print("Custom server: Error reading config \(fileURL.path): \(error)")
        return "ERROR: Failed custom server init"
    }
    
    print("Custom server URL from \(fileURL.path) applied")
    return "Custom Server Ready"
}

class MessageManager: ObservableObject {
    @Published var audioStatus: String = "Idle"
    @Published var codeStatus: String = "Idle"
    @Published var mainStatus: String = "Ready"
    @Published var poseStatus: String = "Idle"
    @Published var buttonStatus: String = "Start Capture"
    
    func audio(_ status: String) {
        DispatchQueue.main.async {
            self.audioStatus = status
        }
    }
    
    func code(_ status: String) {
        DispatchQueue.main.async {
            self.codeStatus = status
        }
    }
    
    func main(_ status: String) {
        DispatchQueue.main.async {
            self.mainStatus = status
        }
    }
    
    func pose(_ status: String) {
        DispatchQueue.main.async {
            self.poseStatus = status
        }
    }
    
    func button(_ status: String) {
        DispatchQueue.main.async {
            self.buttonStatus = status
        }
    }
    
    func start() {
        self.audio("Starting...")
        self.code("Starting...")
        self.pose("Starting...")
        self.main("Recording Active")
        self.button("Stop Capture")
    }
    
    func stop() {
        self.main("Stopping...")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.audio("Idle")
            self.code("Idle")
            self.pose("Idle")
            self.main("Ready")
            self.button("Start Capture")
        }
    }
}

struct ContentView: View {
    @ObservedObject private var mm = MessageManager.shared
    @State var audio = AudioRecorder()
    @State private var floatingManager = FloatingWindowManager()
    @State private var isCap: Bool = false
    
    init() {
        mm.main(initURL())
    }
    
    var body: some View {
        VStack(spacing: 40) {
            // Main status display
            Text(mm.mainStatus)
                .font(.system(size: 28, weight: .bold))
                .foregroundColor(.primary)
            
            // Status text boxes
            VStack(spacing: 24) {
                StatusBox(title: "Verbal Communication Analysis", status: mm.audioStatus, color: .blue)
                StatusBox(title: "Typing Pattern Analysis", status: mm.codeStatus, color: .orange)
                StatusBox(title: "Head Pose Analysis", status: mm.poseStatus, color: .green)
            }
            
            // Control buttons
            HStack(spacing: 30) {
                Button() {
                    floatingManager.showFloatingPicker()
                } label: {
                    Text("Window Picker")
                        .padding(7)
                }
                .font(.system(size: 22, weight: .semibold))
                
                Button() {
                    if isCap {
                        stopAud()
                        stopCap()
                        mm.stop()
                        isCap = false
                        print("Stopping capture")
                    } else {
                        print("Starting capture")
                        mm.start()
                        
                        startAud() { ret in
                            mm.audio(ret)
                        }
                        
                        /*
                        let codeWindow = floatingManager.getCodeWindow()
                        let poseWindow = floatingManager.getPoseWindow()
                        mm.code(codeWindow != nil ? "Active" : "Idle")
                        mm.pose(poseWindow != nil ? "Active" : "Idle")
                         */
                        isCap = true
                        startCap()
                    }
                } label: {
                    Text(mm.buttonStatus)
                        .padding(7)
                }
                .buttonStyle(.borderedProminent)
                .font(.system(size: 22, weight: .semibold))
            }
        }
        .frame(width: 600, height: 480)
        .padding()
        .onAppear {
            // Make window always on top
            if let window = NSApplication.shared.windows.first {
                window.level = .floating
                window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            }
        }
        .onReceive(mm.$mainStatus) { _ in }
    }
}

struct StatusBox: View {
    let title: String
    let status: String
    let color: Color
    
    var body: some View {
        HStack {
            Circle()
                .fill(color)
                .frame(width: 16, height: 16)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 22, design: .monospaced))
                    .foregroundColor(.secondary)
                Text(status)
                    .font(.system(size: 22, design: .monospaced))
                    .foregroundColor(.primary)
            }
            
            Spacer()
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .background(Color(.controlBackgroundColor))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color(.separatorColor), lineWidth: 1)
        )
    }
}
