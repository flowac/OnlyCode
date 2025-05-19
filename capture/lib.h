#ifndef _MAIN_LIB_H_
#define _MAIN_LIB_H_

#include <windows.h>
#include <wingdi.h>
#include <winuser.h>

#include <iostream>

#include <QApplication>
#include <QDateTime>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QScreen>
#include <QThread>
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

class SelectButton : public QPushButton
{Q_OBJECT
public:
    SelectButton(QWidget *p, HWND id) : QPushButton(p), _id(id) {}
public slots:
    void clicked() {emit selected(_id);}
signals:
    void selected(HWND);
private:
    HWND _id;
};

class App : public QWidget
{Q_OBJECT
public:
    App();

public slots:
    void _quit() {QApplication::quit();}
    void _select(HWND id);

protected:
    void keyPressEvent(QKeyEvent *evt) override;

private:
    static constexpr int row_lim{5}; // Previews per row
    HWND capture_target{0};
    QGridLayout *lay_select; // Window selection panel
    int icon_w, icon_h, icon_space; // Window preview size

    void show_list();
};
#endif

