#include <unordered_set>
#include <vector>

#include "lib.h"

std::unordered_set<QString> task_filter({
"Cap",
"Microsoft Text Input Application",
"NVIDIA GeForce Overlay",
"Program Manager",
"Settings"
});

BOOL CALLBACK list_windows(HWND id, LPARAM param)
{
    constexpr int TLIM{60};
    WCHAR title[TLIM];
    if (!IsWindowVisible(id) || GetWindowTextW(id, title, TLIM) == 0) return TRUE;
    QString name = QString::fromWCharArray(title);
    if (task_filter.contains(name)) return TRUE;

    RECT geo;
    HDC wdc = GetWindowDC(id);
    if (!GetClientRect(id, &geo)) return TRUE;
    int width = geo.right - geo.left;
    int height = geo.bottom - geo.top;

    HDC mirror = CreateCompatibleDC(wdc);
    HBITMAP bmp = CreateCompatibleBitmap(wdc, width, height);
    SelectObject(mirror, bmp);
    if (!BitBlt(mirror, 0, 0, width, height, wdc, 0, 0, SRCCOPY)) return TRUE;

    QImage capture = QImage::fromHBITMAP(bmp);
    ReleaseDC(id, wdc);
    DeleteDC(mirror);
    DeleteObject(bmp);

    std::cout << name.u8s << std::endl;
    name += "\n" + QString::number(width) + "x" + QString::number(height);
    auto &wlist = *reinterpret_cast<std::vector<WindowInfo> *>(param);
    wlist.push_back({id, name, QPixmap::fromImage(capture)});

    return TRUE;
}

void App::show_list()
{
    std::vector<WindowInfo> wlist;
    EnumWindows(list_windows, reinterpret_cast<LPARAM>(&wlist));

    for (const auto &winfo : wlist)
    {
        QWidget *element = new QWidget(this);
        QVBoxLayout *vbox = new QVBoxLayout(element);

        QIcon icon(winfo.pix);
        SelectButton *button = new SelectButton(element, winfo.id);
        button->setIcon(icon);
        button->setIconSize(QSize(icon_w, icon_h));
        connect(button, SIGNAL(released()), button, SLOT(clicked()));
        connect(button, SIGNAL(selected(HWND)), this, SLOT(_select(HWND)));

        vbox->addWidget(new QLabel(winfo.name, element), 0, Qt::AlignBottom);
        vbox->addWidget(button, 0, Qt::AlignTop);
        vbox->setSpacing(0);
        vbox->setContentsMargins(0, 0, 0, 0);

        int cnt = lay_select->count();
        lay_select->addWidget(element, cnt / row_lim, cnt % row_lim, Qt::AlignCenter);
    }

    int iw = icon_w + icon_space;
    int ih = icon_h + icon_space;
    QRect geo = geometry();
    geo.setWidth(std::max(1, std::min(lay_select->count(), row_lim)) * iw);
    geo.setHeight(std::max(1, lay_select->count() / row_lim) * ih);
    this->setGeometry(geo);
}

void App::_select(HWND id)
{
    capture_target = id;

    WCHAR title[BUF_200];
    GetWindowTextW(id, title, BUF_200);
    std::cout << "Selected window: ";
    std::wcout << title;
    std::cout << std::endl << std::flush;

    QThread::msleep(100);
    _quit();
}

