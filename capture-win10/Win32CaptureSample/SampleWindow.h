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
    void SetSubTitle(std::wstring const& text);
    winrt::fire_and_forget OnPickerButtonClicked();
    winrt::fire_and_forget OnSnapshotButtonClicked();
    void StopCapture();
    void OnCaptureItemClosed(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const&, winrt::Windows::Foundation::IInspectable const&);
    void OnCaptureStarted(
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item);

private:
    HWND m_pickerButton = nullptr;
    HWND m_stopButton = nullptr;
    HWND m_snapshotButton = nullptr;
    std::shared_ptr<App> m_app;
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem::Closed_revoker m_itemClosedRevoker;
};