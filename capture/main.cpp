#include <windows.h>
#include <winuser.h>

#include <iostream>
#include <string>
#include <vector>

#include "lib.h"

BOOL CALLBACK list_windows(HWND hwnd, LPARAM lParam)
{
    constexpr DWORD kTitleSize{1024};
    WCHAR title[kTitleSize];

    GetWindowTextW(hwnd, title, kTitleSize);

    std::wstring wst(title);
    if (!IsWindowVisible(hwnd) || wst.empty() || title == L"Capper") {
        return TRUE;
    }

    std::vector<std::wstring>& titles =
        *reinterpret_cast<std::vector<std::wstring>*>(lParam);
    titles.push_back(title);

    return TRUE;
}

int main(int argc, char **argv)
{
    std::vector<std::wstring> titles;
    EnumWindows(list_windows, reinterpret_cast<LPARAM>(&titles));

    for (const auto& title : titles)
        std::wcout << L"Title: " << title << std::endl;

    QApplication app(argc, argv);
    App ui;

    return app.exec();
}

