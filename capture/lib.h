#ifndef _MAIN_LIB_H_
#define _MAIN_LIB_H_

#include <windows.h>
#include <wingdi.h>
#include <winuser.h>

#include <QApplication>
#include <QDateTime>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QScreen>
#include <QTimer>

#include <QWidget>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include <QLabel>
#include <QPalette>
#include <QPushButton>
#include <QSizePolicy>

#define u8s toUtf8().constData()

#define BUF_80   80
#define BUF_200 200
#define FONT_SZ  24

typedef struct
{
    HWND id;
    QString name;
    QPixmap pix;
} WindowInfo;

class App : public QWidget
{Q_OBJECT
public:
    App();

protected:
    void keyPressEvent(QKeyEvent *evt) override;

private slots:
    void _quit();

private:
    QGridLayout *lay_select{0}; // Window selection panel
    static constexpr int row_lim{5}; // Previews per row
    int icon_w, icon_h, icon_space; // Window preview size

    void show_list();
};
#endif

