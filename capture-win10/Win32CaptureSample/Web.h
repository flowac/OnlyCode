#pragma once

winrt::Windows::Foundation::IAsyncOperation<winrt::hstring> SendHttpPost(
    const WCHAR* fname,
    const winrt::hstring& serverUrl);

winrt::Windows::Foundation::IAsyncOperation<winrt::hstring> SendHttpGet(
	const winrt::hstring& serverUrl);
