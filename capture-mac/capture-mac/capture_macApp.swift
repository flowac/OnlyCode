//
//  capture_macApp.swift
//  capture-mac
//

import SwiftUI
import ScreenCaptureKit
import AVFoundation

@main
struct capture_macApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
