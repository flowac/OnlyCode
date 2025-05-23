#include "pch.h"
#include "App.h"
#include "SampleWindow.h"

namespace winrt
{
    using namespace Windows::Foundation;
    using namespace Windows::Foundation::Metadata;
    using namespace Windows::Graphics::Capture;
    using namespace Windows::Graphics::DirectX;
    using namespace Windows::System;
    using namespace Windows::UI;
    using namespace Windows::UI::Composition;
    using namespace Windows::UI::Composition::Desktop;
}

namespace util
{
    using namespace robmikh::common::desktop::controls;
}

const std::wstring SampleWindow::ClassName = L"Cap.SampleWindow";
std::once_flag SampleWindowClassRegistration;

const std::wstring WindowsUniversalContract = L"Windows.Foundation.UniversalApiContract";

void SampleWindow::RegisterWindowClass()
{
    auto instance = winrt::check_pointer(GetModuleHandleW(nullptr));
    WNDCLASSEX wcex = {};
    wcex.cbSize = sizeof(wcex);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.hInstance = instance;
    wcex.hIcon = LoadIconW(instance, IDI_APPLICATION);
    wcex.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszClassName = ClassName.c_str();
    wcex.hIconSm = LoadIconW(wcex.hInstance, IDI_APPLICATION);
    winrt::check_bool(RegisterClassExW(&wcex));
}

SampleWindow::SampleWindow(int width, int height, std::shared_ptr<App> app)
{
    auto instance = winrt::check_pointer(GetModuleHandleW(nullptr));

    std::call_once(SampleWindowClassRegistration, []() { RegisterWindowClass(); });

    auto exStyle = WS_EX_LAYERED;
    auto style = WS_OVERLAPPEDWINDOW;

    RECT rect = { 0, 0, width, height };
    winrt::check_bool(AdjustWindowRectEx(&rect, style, false, exStyle));
    auto adjustedWidth = rect.right - rect.left;
    auto adjustedHeight = rect.bottom - rect.top;

    winrt::check_bool(CreateWindowExW(exStyle, ClassName.c_str(), L"Cap", style,
        CW_USEDEFAULT, CW_USEDEFAULT, adjustedWidth, adjustedHeight, nullptr, nullptr, instance, this));
    WINRT_ASSERT(m_window);

    SetLayeredWindowAttributes(m_window, RGB(200, 200, 200), 200, LWA_ALPHA);

    auto isAllDisplaysPresent = winrt::ApiInformation::IsApiContractPresent(WindowsUniversalContract, 9);

    m_app = app;
    m_app->InitializeWithWindow(m_window);

    CreateControls(instance);
    winrt::check_bool(SetWindowDisplayAffinity(m_window, WDA_EXCLUDEFROMCAPTURE));

    ShowWindow(m_window, SW_SHOW);
    UpdateWindow(m_window);
}

SampleWindow::~SampleWindow()
{}

LRESULT SampleWindow::MessageHandler(UINT const message, WPARAM const wparam, LPARAM const lparam)
{
    switch (message)
    {
    case WM_COMMAND:
    {
        auto command = HIWORD(wparam);
        auto hwnd = (HWND)lparam;
        switch (command)
        {
        case BN_CLICKED:
            {
                if (hwnd == m_pickerButton)
                {
                    OnPickerButtonClicked();
                }
                else if (hwnd == m_stopButton)
                {
                    StopCapture();
                }
                else if (hwnd == m_snapshotButton)
                {
                    OnSnapshotButtonClicked();
                }
            }
            break;
        }
    }
    break;
    case WM_CTLCOLORSTATIC:
        return util::StaticControlColorMessageHandler(wparam, lparam);
    default:
        return base_type::MessageHandler(message, wparam, lparam);
    }

    return 0;
}

void SampleWindow::OnCaptureStarted(winrt::GraphicsCaptureItem const& item)
{
    m_itemClosedRevoker.revoke();
    m_itemClosedRevoker = item.Closed(winrt::auto_revoke, { this, &SampleWindow::OnCaptureItemClosed });
    SetSubTitle(std::wstring(item.DisplayName()));
    EnableWindow(m_stopButton, true);
    EnableWindow(m_snapshotButton, true);
}

winrt::fire_and_forget SampleWindow::OnPickerButtonClicked()
{
    auto selectedItem = co_await m_app->StartCaptureWithPickerAsync();

    if (selectedItem)
    {
        OnCaptureStarted(selectedItem);
    }
}

winrt::fire_and_forget SampleWindow::OnSnapshotButtonClicked()
{
    auto file = co_await m_app->TakeSnapshotAsync();
    if (file != nullptr)
    {
        co_await winrt::Launcher::LaunchFileAsync(file);
    }
}

// Not DPI aware but could be by multiplying the constants based on the monitor scale factor
void SampleWindow::CreateControls(HINSTANCE instance)
{
    // Programmatic capture
    auto isWin32ProgrammaticPresent = winrt::ApiInformation::IsApiContractPresent(WindowsUniversalContract, 8);
    auto win32ProgrammaticStyle = isWin32ProgrammaticPresent ? 0 : WS_DISABLED;

    auto controls = util::StackPanel(m_window, instance, 10, 10, 40, 200, 30);

    // Create picker button
    m_pickerButton = controls.CreateControl(util::ControlType::Button, L"Open Picker");

    // Create stop capture button
    m_stopButton = controls.CreateControl(util::ControlType::Button, L"Stop Capture", WS_DISABLED);

    // Create independent snapshot button
    m_snapshotButton = controls.CreateControl(util::ControlType::Button, L"Take Snapshot", WS_DISABLED);
}

void SampleWindow::SetSubTitle(std::wstring const& text)
{
    std::wstring titleText(L"Cap");
    if (!text.empty())
    {
        titleText += (L" - " + text);
    }
    SetWindowTextW(m_window, titleText.c_str());
}

void SampleWindow::StopCapture()
{
    m_app->StopCapture();
    SetSubTitle(L"");
    EnableWindow(m_stopButton, false);
    EnableWindow(m_snapshotButton, false);
}

void SampleWindow::OnCaptureItemClosed(winrt::GraphicsCaptureItem const&, winrt::IInspectable const&)
{
    StopCapture();
}