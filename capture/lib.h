#ifndef _MAIN_LIB_H_
#define _MAIN_LIB_H_

#include <cmath>
#include <cstdio>
#include <cstdint>

#include <vector>

#include <QHBoxLayout>
#include <QLabel>
#include <QPalette>
#include <QPushButton>
#include <QSizePolicy>
#include <QWidget>

#include <QApplication>
#include <QDateTime>
#include <QGuiApplication>
#include <QTimer>

#define u8s toUtf8().constData()

#define BUF_80	80
#define BUF_200	200
#define FONT_SZ	24

class App : public QWidget
{
	Q_OBJECT
public:
	App();
private slots:
	void _quit();
};
#endif

