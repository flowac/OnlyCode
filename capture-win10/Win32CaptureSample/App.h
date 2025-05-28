#pragma once
#include "SimpleCapture.h"
#include "audio.h"

class App
{
public:
    App(winrt::Windows::UI::Composition::ContainerVisual root);
    ~App() {}

    winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Graphics::Capture::GraphicsCaptureItem> StartCaptureWithPickerAsync();
    winrt::Windows::Foundation::IAsyncOperation<bool> TakeSnapshotAsync();
    void StopCapture();
    void InitializeWithWindow(HWND window);

private:
    void StartCaptureFromItem(winrt::Windows::Graphics::Capture::GraphicsCaptureItem item);
    void InitializeObjectWithWindowHandle(winrt::Windows::Foundation::IUnknown const& object);
    void OnTick(winrt::Windows::Foundation::IInspectable const &sender, winrt::Windows::Foundation::IInspectable const &event);

    winrt::Windows::System::DispatcherQueue m_mainThread{ nullptr };
    winrt::Windows::UI::Composition::Compositor m_compositor{ nullptr };
    winrt::Windows::UI::Composition::ContainerVisual m_root{ nullptr };
    winrt::Windows::UI::Composition::SpriteVisual m_content{ nullptr };
    winrt::Windows::UI::Composition::CompositionSurfaceBrush m_brush{ nullptr };

    HWND m_mainWindow{ nullptr };
    winrt::Windows::Storage::StorageFolder pwd{ nullptr };
    AudioCapture m_record;
    bool audio_ready{ false };

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };
    std::unique_ptr<SimpleCapture> m_capture{ nullptr };
	winrt::Windows::System::DispatcherQueueTimer m_timer{ nullptr };
    winrt::Windows::Graphics::DirectX::DirectXPixelFormat m_pixelFormat{ winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized };
};