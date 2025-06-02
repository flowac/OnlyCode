#include "pch.h"

#include "App.h"
#include "SampleWindow.h"

namespace winrt
{
    using namespace Windows::Graphics::Capture;
    using namespace Windows::UI::Core;
    using namespace Windows::UI::Composition;
}

namespace util
{
    using namespace robmikh::common::desktop;
}

static int __stdcall WinMain(HINSTANCE, HINSTANCE, PSTR, int)
{
    winrt::init_apartment(winrt::apartment_type::single_threaded);
    auto isCaptureSupported = winrt::Windows::Graphics::Capture::GraphicsCaptureSession::IsSupported();
    if (!isCaptureSupported)
    {
        MessageBoxW(nullptr, L"Screen capture is not supported", L"Cap", MB_OK | MB_ICONERROR);
        return 1;
    }

    auto controller = util::CreateDispatcherQueueControllerForCurrentThread();
    auto compositor = winrt::Compositor();
    auto root = compositor.CreateContainerVisual();
    root.RelativeSizeAdjustment({ 1.0f, 1.0f });
    root.Size({ -220.0f, 0.0f });
    root.Offset({ 220.0f, 0.0f, 0.0f });

    auto app = std::make_shared<App>(root);
    auto window = SampleWindow(300, 300, app);
    auto target = window.CreateWindowTarget(compositor);
    target.Root(root);
    SetWindowPos(window.m_window, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return util::ShutdownDispatcherQueueControllerAndWait(controller, static_cast<int>(msg.wParam));
}