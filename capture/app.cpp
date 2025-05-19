#include "lib.h"

App::App()
{
    QString base, css, fontsize;
    QPalette cback = QPalette(QPalette::Window, Qt::gray);
    base = "red";
    fontsize = QString::number(FONT_SZ);
    css = "background: black; border:1px solid " + base + "; color: " + base + "; font-size: " + fontsize + "px;";//button

    lay_select = new QGridLayout(this);
    setFocusPolicy(Qt::StrongFocus);
    setPalette(cback);
    setWindowFlags(Qt::FramelessWindowHint | Qt::BypassWindowManagerHint);
//    setAttribute(Qt::WA_TranslucentBackground);
    activateWindow();
    show();

    QPoint qxy(this->x(), this->y());
    QScreen *qscr = QGuiApplication::screenAt(qxy);
    QRect geo = qscr->availableGeometry();
    icon_w = geo.width() * 0.8 / row_lim;
    icon_h = geo.height() * icon_w / geo.width();
    icon_space = geo.width() / 100;
    geo.setX(geo.x() + geo.width() / 16);
    geo.setY(geo.y() + geo.height() / 8);
    geo.setWidth(500);
    geo.setHeight(500);
    setGeometry(geo);

    show_list();
}

void App::keyPressEvent(QKeyEvent *evt)
{
    switch (evt->key())
    {
    case Qt::Key_Escape:
        _quit(); break;
    }
}

