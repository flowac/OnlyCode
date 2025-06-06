#include "pch.h"
#include "App.h"
#include "SampleWindow.h"

extern std::ofstream LOGA;

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
std::once_flag RegisterFonts;

const std::wstring WindowsUniversalContract = L"Windows.Foundation.UniversalApiContract";

// Color scheme
const COLORREF BACKGROUND_COLOR = RGB(32, 32, 36);   // Dark background
const COLORREF BACKGROUND_COLOR_LITE = RGB(45, 45, 48);
const COLORREF ACCENT_COLOR = RGB(0, 120, 215);      // Windows blue
const COLORREF BUTTON_COLOR = RGB(45, 45, 48);       // Dark button
const COLORREF BUTTON_HOVER_COLOR = RGB(62, 62, 66); // Lighter on hover
const COLORREF TEXT_COLOR = RGB(255, 255, 255);      // White text
const COLORREF BORDER_COLOR = RGB(104, 104, 104);    // Gray border

HBRUSH g_buttonBrush;
HBRUSH g_labelBrush;
HBRUSH g_texteditBrush;
HFONT g_hFontTitle = nullptr;
HFONT g_hFontButton = nullptr;
HFONT g_hFontText = nullptr;

void CreateColors()
{
    g_hFontTitle = CreateFontW(-18, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    g_hFontButton = CreateFontW(-14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    g_hFontText = CreateFontW(-12, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    g_buttonBrush = CreateSolidBrush(BUTTON_COLOR);
    g_labelBrush = CreateSolidBrush(BACKGROUND_COLOR);
    g_texteditBrush = CreateSolidBrush(BACKGROUND_COLOR_LITE);
}

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
    wcex.hbrBackground = CreateSolidBrush(BACKGROUND_COLOR);
    wcex.lpszClassName = ClassName.c_str();
    wcex.hIconSm = LoadIconW(wcex.hInstance, IDI_APPLICATION);
    winrt::check_bool(RegisterClassExW(&wcex));
}

SampleWindow::SampleWindow(int width, int height, std::shared_ptr<App> app)
{
    auto instance = winrt::check_pointer(GetModuleHandleW(nullptr));

    std::call_once(SampleWindowClassRegistration, []() { RegisterWindowClass(); });
    std::call_once(RegisterFonts, []() { CreateColors(); });

    // Inactivative options would be useful for the next project
    auto exStyle = WS_EX_TOPMOST | WS_EX_LAYERED;// | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE;
    auto style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;// | WS_THICKFRAME;// WS_OVERLAPPEDWINDOW;

    RECT rect = { 0, 0, width, height };
    winrt::check_bool(AdjustWindowRectEx(&rect, style, false, exStyle));
    auto adjustedWidth = rect.right - rect.left;
    auto adjustedHeight = rect.bottom - rect.top;

    winrt::check_bool(CreateWindowExW(exStyle, ClassName.c_str(), L"Cap", style,
        CW_USEDEFAULT, CW_USEDEFAULT, adjustedWidth, adjustedHeight, nullptr, nullptr, instance, this));
    WINRT_ASSERT(m_window);
    SetLayeredWindowAttributes(m_window, RGB(100, 100, 100), 250, LWA_ALPHA);

    m_app = app;
    m_app->InitializeWithWindow(m_window);
    CreateControls(instance);
    m_app->SetResultPtr(m_result);

    if (bool secure = SetWindowDisplayAffinity(m_window, WDA_EXCLUDEFROMCAPTURE); // Prevents the window from being captured
    secure) LOGA << "Session secured: Successfully configured WDA_EXCLUDEFROMCAPTURE\n";
    else    LOGA << "Session NOT secure: SetWindowDisplayAffinity failed with error: " << GetLastError() << std::endl;

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
                OnPickerButtonClicked(CaptureIndex::CODE);
            }
            else if (hwnd == m_stopButton)
            {
                StopCapture(CaptureIndex::POSE);
                StopCapture(CaptureIndex::CODE);
            }
            break;
        }
    }
    break;
    //case WM_CTLCOLORSTATIC:
    //    return util::StaticControlColorMessageHandler(wparam, lparam);
    case WM_CTLCOLORBTN:  // Button colors
    {
        HDC hdc = (HDC)wparam;
        SetTextColor(hdc, TEXT_COLOR);
        SetBkColor(hdc, BUTTON_COLOR);
        return (LRESULT)g_buttonBrush;
    }
    break;
    case WM_CTLCOLORSTATIC:  // Static/Label colors
    {
        HDC hdc = (HDC)wparam;
        SetTextColor(hdc, TEXT_COLOR);
        SetBkColor(hdc, BACKGROUND_COLOR);
        return (LRESULT)g_labelBrush;
    }
    break;
    case WM_CTLCOLOREDIT:  // Edit control colors
    {
        HDC hdc = (HDC)wparam;
        SetTextColor(hdc, TEXT_COLOR);
        SetBkColor(hdc, BACKGROUND_COLOR_LITE);
        return (LRESULT)g_texteditBrush;
    }
    break;
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
    case CaptureIndex::CODE:
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
    auto isWin32ProgrammaticPresent = winrt::ApiInformation::IsApiContractPresent(WindowsUniversalContract, 8);
    auto win32ProgrammaticStyle = isWin32ProgrammaticPresent ? 0 : WS_DISABLED;

    auto controls = util::StackPanel(m_window, instance, 10, 10, 40, 200, 30);
    m_poseButton = controls.CreateControl(util::ControlType::Button, L"Pose Detection");
    m_countButton = controls.CreateControl(util::ControlType::Button, L"Code Detection");
    m_stopButton = controls.CreateControl(util::ControlType::Button, L"Stop Capture", WS_DISABLED);
	m_label = controls.CreateControl(util::ControlType::Label, L"Analysis");
	m_result = controls.CreateControl(util::ControlType::Edit, L"Ready for capture...", ES_MULTILINE | ES_READONLY);
    SetWindowPos(m_result, nullptr, 0, 0, 200, 140, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);

    SendMessage(m_poseButton, WM_SETFONT, (WPARAM)g_hFontButton, TRUE);
    SendMessage(m_countButton, WM_SETFONT, (WPARAM)g_hFontButton, TRUE);
    SendMessage(m_stopButton, WM_SETFONT, (WPARAM)g_hFontButton, TRUE);
    SendMessage(m_label, WM_SETFONT, (WPARAM)g_hFontText, TRUE);
    SendMessage(m_result, WM_SETFONT, (WPARAM)g_hFontText, TRUE);
}

int SampleWindow::SetSubTitle()
{
	int nActive = 0;
    std::wstring titleText;
    for (int idx = 0; idx < CaptureIndex::CAPTURE_LEN; ++idx)
    {
        titleText.append(CaptureName[idx]);
        titleText.append(m_isCap[idx] ? L" ON  " : L" OFF  ");
        nActive += m_isCap[idx];
	}
    SetWindowTextW(m_window, titleText.c_str());
    return nActive;
}

void SampleWindow::StopCapture(CaptureIndex idx)
{
	m_isCap[idx] = false;
    m_app->StopCapture(idx);
	if (SetSubTitle() == 0) EnableWindow(m_stopButton, false);
}

void SampleWindow::OnCaptureItemClosedPose(winrt::GraphicsCaptureItem const&, winrt::IInspectable const&)
{
    StopCapture(CaptureIndex::POSE);
}

void SampleWindow::OnCaptureItemClosedCount(winrt::GraphicsCaptureItem const&, winrt::IInspectable const&)
{
    StopCapture(CaptureIndex::CODE);
}
