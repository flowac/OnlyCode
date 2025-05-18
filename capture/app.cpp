#include "lib.h"

App::App()
{
	QString base, css, fontsize;
	QBrush brush(QColor(0,0,0), Qt::SolidPattern);
	QPalette cback = QPalette(QPalette::Window, Qt::black);
	QSizePolicy szpol = QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout *hlay1;

	base = "red";
	fontsize = QString::number(FONT_SZ);
	css = "background: black; border:1px solid " + base + "; color: " + base + "; font-size: " + fontsize + "px;";//button

	hlay1 = new QHBoxLayout(this);
        std::vector<QString> plist = {"test.png"};

        for (auto &element : plist)
        {
            QPixmap pixmap(element);
            QIcon ButtonIcon(pixmap);
            QPushButton *button = new QPushButton(this);
            button->setIcon(ButtonIcon);
            button->setIconSize(pixmap.rect().size());
	    hlay1->addWidget(button);
        }

	this->setPalette(cback);
	this->activateWindow();
	this->show();
}

void App::_quit()
{
	QApplication::quit();
}

