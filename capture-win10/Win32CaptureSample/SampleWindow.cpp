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
            if (hwnd == m_poseButton)
            {
                OnPickerButtonClicked(CaptureIndex::POSE);
            }
            else if (hwnd == m_countButton)
            {
                OnPickerButtonClicked(CaptureIndex::COUNT);
            }
            else if (hwnd == m_stopButton)
            {
                StopCapture(CaptureIndex::POSE);
                StopCapture(CaptureIndex::COUNT);
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

void SampleWindow::OnCaptureStarted(winrt::GraphicsCaptureItem const& item, CaptureIndex idx)
{
    m_itemClosedRevoker[idx].revoke();
    m_isCap[idx] = true;

    switch (idx)
    {
    case CaptureIndex::POSE:
        m_itemClosedRevoker[idx] = item.Closed(winrt::auto_revoke, { this, &SampleWindow::OnCaptureItemClosedPose });
        break;
    case CaptureIndex::COUNT:
        m_itemClosedRevoker[idx] = item.Closed(winrt::auto_revoke, { this, &SampleWindow::OnCaptureItemClosedCount });
        break;
    }
    SetSubTitle();
    EnableWindow(m_stopButton, true);
}

winrt::fire_and_forget SampleWindow::OnPickerButtonClicked(CaptureIndex idx)
{
    auto selectedItem = co_await m_app->StartCaptureWithPickerAsync(idx);
    if (selectedItem) OnCaptureStarted(selectedItem, idx);
}

// Not DPI aware but could be by multiplying the constants based on the monitor scale factor
void SampleWindow::CreateControls(HINSTANCE instance)
{
    // Programmatic capture
    auto isWin32ProgrammaticPresent = winrt::ApiInformation::IsApiContractPresent(WindowsUniversalContract, 8);
    auto win32ProgrammaticStyle = isWin32ProgrammaticPresent ? 0 : WS_DISABLED;

    auto controls = util::StackPanel(m_window, instance, 10, 10, 40, 200, 30);
    m_poseButton = controls.CreateControl(util::ControlType::Button, L"Face detection / Pose Picker");
    m_countButton = controls.CreateControl(util::ControlType::Button, L"Code detection / Count Picker");
    m_stopButton = controls.CreateControl(util::ControlType::Button, L"Stop Capture", WS_DISABLED);
}

void SampleWindow::SetSubTitle()
{
    std::wstring titleText;
    for (int idx = 0; idx < CaptureIndex::CAPTURE_LEN; ++idx)
    {
        titleText.append(CaptureName[idx]);
        titleText.append(m_isCap[idx] ? L" ON  " : L" OFF  ");
	}
    SetWindowTextW(m_window, titleText.c_str());
}

void SampleWindow::StopCapture(CaptureIndex idx)
{
	m_isCap[idx] = false;
    m_app->StopCapture(idx);
    SetSubTitle();
}

void SampleWindow::OnCaptureItemClosedPose(winrt::GraphicsCaptureItem const&, winrt::IInspectable const&)
{
    StopCapture(CaptureIndex::POSE);
}

void SampleWindow::OnCaptureItemClosedCount(winrt::GraphicsCaptureItem const&, winrt::IInspectable const&)
{
    StopCapture(CaptureIndex::COUNT);
}
