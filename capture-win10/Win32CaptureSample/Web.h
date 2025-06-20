#pragma once

winrt::Windows::Foundation::IAsyncOperation<winrt::hstring> SendHttpPost(
    const WCHAR* fname,
    const winrt::hstring& serverUrl);

winrt::Windows::Foundation::IAsyncOperation<winrt::hstring> SendHttpGet(
	const winrt::hstring& serverUrl);

#include <unordered_map>
void ParseJson(winrt::hstring data, std::unordered_map<winrt::hstring, double> &query);
