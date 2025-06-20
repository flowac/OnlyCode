#include "pch.h"
#include <winrt/Windows.Web.Http.h>
#include <winrt/Windows.Web.Http.Headers.h>
#include <iostream>

#include "Web.h"

namespace winrt {
    using namespace winrt::Windows::Foundation;
    using namespace winrt::Windows::Web::Http;
    using namespace winrt::Windows::Web::Http::Headers;
    using namespace winrt::Windows::Storage::Streams;
}

winrt::IBuffer LoadFileBuffer(const std::wstring& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    winrt::Buffer buffer(static_cast<uint32_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    buffer.Length(static_cast<uint32_t>(size));
    return buffer;
}

winrt::IAsyncOperation<winrt::hstring> SendHttpPost(
    const WCHAR* fname,
    const winrt::hstring& serverUrl)
{
    winrt::hstring responseText(L"Local file read error");
    winrt::IBuffer ibuf = LoadFileBuffer(fname);
    winrt::HttpBufferContent fileContent(ibuf);
    WCHAR dispo[256];
	swprintf_s(dispo, L"form-data; name=\"file\"; filename=\"%s\"", fname);
    fileContent.Headers().Append(L"Content-Disposition", dispo);

    winrt::HttpClient httpClient;
    winrt::HttpRequestMessage request(winrt::HttpMethod::Post(), winrt::Uri(serverUrl));
    winrt::HttpMultipartFormDataContent multipartContent(L"form-data");
    multipartContent.Add(fileContent);
    request.Content(multipartContent);

    winrt::HttpResponseMessage response = co_await httpClient.SendRequestAsync(request);
    try {
        response.EnsureSuccessStatusCode();
        winrt::IBuffer responseBuffer = co_await response.Content().ReadAsBufferAsync();
        winrt::DataReader reader = winrt::DataReader::FromBuffer(responseBuffer);
        responseText = reader.ReadString(responseBuffer.Length());
    } catch (const std::exception& e) {
        responseText = winrt::to_hstring(e.what());
    }

    co_return responseText;
}

winrt::IAsyncOperation<winrt::hstring> SendHttpGet(
    const winrt::hstring& serverUrl)
{
    winrt::HttpClient httpClient;
    winrt::HttpResponseMessage response = co_await httpClient.GetAsync(winrt::Uri(serverUrl));

    winrt::hstring responseText;
    try {
        response.EnsureSuccessStatusCode();
        responseText = co_await response.Content().ReadAsStringAsync();
    } catch (const std::exception& e) {
        responseText = winrt::to_hstring(e.what());
    }

    co_return responseText;
}

#include <winrt/Windows.Data.Json.h>
void ParseJson(winrt::Windows::Data::Json::JsonObject jsonObject, std::unordered_map<winrt::hstring, double> &query)
{
    for (const auto& pair : jsonObject) {
        if (query.contains(pair.Key())
            && pair.Value().ValueType() == winrt::Windows::Data::Json::JsonValueType::Number) {
            query[pair.Key()] = pair.Value().GetNumber();
        } else if (pair.Value().ValueType() == winrt::Windows::Data::Json::JsonValueType::Object) {
            ParseJson(pair.Value().GetObject(), query);
        }
    }
}

void ParseJson(winrt::hstring data, std::unordered_map<winrt::hstring, double> &query)
{
    winrt::Windows::Data::Json::JsonObject jsonObject = winrt::Windows::Data::Json::JsonObject::Parse(data);
	ParseJson(jsonObject, query);
}
