#pragma once
#include <robmikh.common/DesktopWindow.h>

class App;
class WindowList;

struct SampleWindow : robmikh::common::desktop::DesktopWindow<SampleWindow>
{
    static const std::wstring ClassName;

    SampleWindow(int width, int height, std::shared_ptr<App> app);
    ~SampleWindow();

    LRESULT MessageHandler(UINT const message, WPARAM const wparam, LPARAM const lparam);

private:
    static void RegisterWindowClass();
    void CreateControls(HINSTANCE instance);
    void SetSubTitle();
    winrt::fire_and_forget OnPickerButtonClicked(CaptureIndex idx);
    void StopCapture(CaptureIndex idx);
    void OnCaptureItemClosedPose(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const&, winrt::Windows::Foundation::IInspectable const&);
    void OnCaptureItemClosedCount(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const&, winrt::Windows::Foundation::IInspectable const&);
    void OnCaptureStarted(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item, CaptureIndex idx);

    bool m_isCap[CaptureIndex::CAPTURE_LEN]{};
    HWND m_poseButton = nullptr;
    HWND m_countButton = nullptr;
    HWND m_stopButton = nullptr;
    HWND m_label = nullptr;
    HWND m_result = nullptr;
    std::shared_ptr<App> m_app;
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem::Closed_revoker m_itemClosedRevoker[CaptureIndex::CAPTURE_LEN];
};