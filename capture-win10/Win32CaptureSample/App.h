#pragma once
#include <atomic>
#include "SimpleCapture.h"
#include "Audio.h"

static enum CaptureIndex : size_t {
    POSE,
    CODE,
    CAPTURE_LEN
};
WCHAR const* const CaptureName[CaptureIndex::CAPTURE_LEN]{
    L"Pose",
    L"Code"
};

class App
{
public:
    App(winrt::Windows::UI::Composition::ContainerVisual root);
    winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Graphics::Capture::GraphicsCaptureItem> StartCaptureWithPickerAsync(CaptureIndex idx);
    winrt::Windows::Foundation::IAsyncOperation<bool> TakeSnapshotAsync(CaptureIndex idx);
    void StopCapture(CaptureIndex idx);
    void InitializeWithWindow(HWND window);
	void SetResultPtr(HWND pose, HWND code, HWND voice);

private:
    void StartCaptureFromItem(winrt::Windows::Graphics::Capture::GraphicsCaptureItem item, CaptureIndex idx);
    void InitializeObjectWithWindowHandle(winrt::Windows::Foundation::IUnknown const& object);
    void OnTickPose(winrt::Windows::Foundation::IInspectable const& sender, winrt::Windows::Foundation::IInspectable const& event);
    void OnTickCount(winrt::Windows::Foundation::IInspectable const& sender, winrt::Windows::Foundation::IInspectable const& event);

    winrt::Windows::System::DispatcherQueue m_mainThread{ nullptr };

    HWND m_mainWindow{ nullptr };
    HWND m_pose { nullptr };
    HWND m_code { nullptr };
    HWND m_voice{ nullptr };
    winrt::Windows::Storage::StorageFolder pwd{ nullptr };
    AudioCapture m_record;
    bool audio_ready{ false };

    bool m_prevCodeSus{ false };
	int m_prevCodeCount{ 0 };
    int m_totalCode{ 1 };
    int m_susCode{ 0 };
    bool m_prevPoseSus{ false };
	int m_totalPose{ 1 };
	int m_susPose{ 0 };

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };
    std::unique_ptr<SimpleCapture>             m_capture[CaptureIndex::CAPTURE_LEN]{ 0,0 };
    winrt::Windows::System::DispatcherQueueTimer m_timer[CaptureIndex::CAPTURE_LEN]{ 0,0 };
    winrt::Windows::Graphics::DirectX::DirectXPixelFormat m_pixelFormat{ winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized };
};
