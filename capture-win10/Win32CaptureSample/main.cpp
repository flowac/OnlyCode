#include "pch.h"
#include <filesystem>
#include <iostream>

#include "App.h"
#include "SampleWindow.h"

winrt::hstring VOICE_API = L"http://104.52.63.66:7001";
winrt::hstring POSE_API = L"http://104.52.63.66:7002";
winrt::hstring CODE_API = L"http://104.52.63.66:7003";

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

static void ReadConfig()
{
	std::string filename = "api.conf";
	std::string u8buf;
	if (!std::filesystem::exists(filename)) return;
	if (std::ifstream file(filename); file.is_open()) {
		std::getline(file, u8buf);
		VOICE_API = winrt::to_hstring(u8buf);
		std::getline(file, u8buf);
		POSE_API = winrt::to_hstring(u8buf);
		std::getline(file, u8buf);
		CODE_API = winrt::to_hstring(u8buf);

		file.close();
	}
}

static int __stdcall WinMain(HINSTANCE, HINSTANCE, PSTR, int)
{
	ReadConfig();
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
	auto window = SampleWindow(220, 270, app);
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